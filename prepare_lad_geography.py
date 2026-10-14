"""Derive static local-authority geography attributes from the ONS Postcode Directory.

Per rent local authority: postcode-weighted centroid, distance to central
London, urban postcode share (2011 rural-urban classification), mean IMD rank
percentile, and the k nearest local authorities by centroid distance.
Only live (unterminated) small-user postcodes are used; raw/ is read-only.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from prepare_local_authority import OLD_TO_RENT_LAD
from prepare_reference import write_csv


ROOT = Path(__file__).resolve().parent
CENTRAL_LONDON = (51.5080, -0.1247)
NEIGHBOURS = 5
# 2011 rural-urban classification: A/B/C are urban classes in England and Wales.
URBAN_PREFIXES = ("A", "B", "C")


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def rent_areas():
    with (ROOT / "data/processed/reference/rent_lad_monthly.csv").open(encoding="utf-8") as file:
        return {row["lad_code"]: row["lad_name"] for row in csv.DictReader(file)}


def accumulate(areas):
    stats = defaultdict(lambda: defaultdict(float))
    imd_max = defaultdict(int)
    path = ROOT / "data/raw/Geography Lookup/ONSPD_AUG_2024.zip"
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist()
                       if re.fullmatch(r"Data/multi_csv/ONSPD_AUG_2024_UK_[A-Z]+\.csv", n))
        for name in names:
            with archive.open(name) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                for row in reader:
                    code = OLD_TO_RENT_LAD.get(row["oslaua"], row["oslaua"])
                    if code not in areas or row["doterm"].strip() or row["usertype"] != "0":
                        continue
                    s = stats[code]
                    s["postcodes"] += 1
                    s["country"] = row["ctry"]
                    lat, lon = float(row["lat"]), float(row["long"])
                    if 49 < lat < 61:
                        s["lat_sum"] += lat
                        s["lon_sum"] += lon
                        s["located"] += 1
                    ru = row["ru11ind"].strip()
                    if ru:
                        s["ru_known"] += 1
                        s["urban"] += ru.startswith(URBAN_PREFIXES)
                    imd = row["imd"].strip()
                    if imd.isdigit() and int(imd) > 0:
                        s["imd_sum"] += int(imd)
                        s["imd_known"] += 1
                        imd_max[row["ctry"]] = max(imd_max[row["ctry"]], int(imd))
    return stats, imd_max


def main():
    areas = rent_areas()
    stats, imd_max = accumulate(areas)
    missing = sorted(set(areas) - set(stats))
    if missing:
        raise RuntimeError(f"Local authorities without live postcodes: {missing}")
    rows, centroids = [], {}
    for code in sorted(areas):
        s = stats[code]
        centroid = (s["lat_sum"] / s["located"], s["lon_sum"] / s["located"])
        centroids[code] = centroid
        # IMD ranks are nation-specific (1 = most deprived); express as a 0-1 deprivation score.
        rows.append({"lad_code": code, "lad_name": areas[code],
                     "country": "Wales" if code.startswith("W") else "England",
                     "centroid_lat": round(centroid[0], 6), "centroid_lon": round(centroid[1], 6),
                     "dist_to_london_km": round(haversine_km(centroid, CENTRAL_LONDON), 3),
                     "live_postcodes": int(s["postcodes"]),
                     "urban_postcode_share": round(s["urban"] / s["ru_known"], 6),
                     "imd_deprivation_score": round(
                         1 - s["imd_sum"] / s["imd_known"] / imd_max[s["country"]], 6)})
    write_csv(ROOT / "data/processed/reference/lad_geography.csv", list(rows[0]), rows)
    links = []
    for code, centroid in centroids.items():
        distances = sorted((haversine_km(centroid, point), other)
                           for other, point in centroids.items() if other != code)
        for rank, (distance, other) in enumerate(distances[:NEIGHBOURS], 1):
            links.append({"lad_code": code, "neighbour_lad_code": other,
                          "rank": rank, "distance_km": round(distance, 3)})
    write_csv(ROOT / "data/processed/reference/lad_neighbours.csv",
              ["lad_code", "neighbour_lad_code", "rank", "distance_km"], links)
    result = {"local_authorities": len(rows), "neighbours_per_lad": NEIGHBOURS,
              "imd_max_rank_by_country": dict(imd_max),
              "live_postcodes": sum(r["live_postcodes"] for r in rows)}
    path = ROOT / "data/metadata/lad_geography.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if len(rows) != 316:
        raise RuntimeError("Geography attributes did not cover all 316 local authorities")


if __name__ == "__main__":
    main()
