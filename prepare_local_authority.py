"""Prepare the finest rent geography available in the supplied ONS workbook.

One rent row is a local-authority/month estimate, not an individual tenancy.
Writes only to processed/, failed/, and metadata/; raw/ is read-only.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from openpyxl import load_workbook

from prepare_reference import REGIONS, positive, write_csv


ROOT = Path(__file__).resolve().parent
LA_PREFIXES = ("E06", "E07", "E08", "E09", "W06")
NAME_TO_REGION = {v.casefold(): k for k, v in REGIONS.items()}
# In the September 2026 PIPR workbook these two area identifiers differ from
# the identifiers in the August 2024 ONSPD and 2024 GDHI workbook. Names and
# regions were inspected in both sources; the boundaries need a sensitivity note.
OLD_TO_RENT_LAD = {"E08000016": "E08000038", "E08000019": "E08000039"}
RENAME_EXPECTED = {"E08000016": "Barnsley", "E08000019": "Sheffield"}


def rent_reference():
    path = ROOT / "data/raw/price_index_of_private_rents.xlsx"
    sheet = load_workbook(path, read_only=True, data_only=True)["Table 1"]
    columns = ["month", "lad_code", "lad_name", "region_code", "region_name",
               "rent_gbp_month", "rent_index", "rent_one_bed_gbp",
               "rent_two_bed_gbp", "rent_three_bed_gbp",
               "rent_four_plus_bed_gbp", "rent_flat_gbp"]
    good, bad, area = [], [], {}
    keys = set()
    for row_number, row in enumerate(sheet.iter_rows(min_row=4, values_only=True), 4):
        period, code, name, region = row[:4]
        if not isinstance(code, str) or not code.startswith(LA_PREFIXES):
            continue
        if not isinstance(period, (datetime, date)) or not 2015 <= period.year <= 2024:
            continue
        region_code = NAME_TO_REGION.get(str(region).casefold())
        values = [row[i] for i in (7, 4, 11, 15, 19, 23, 39)]
        reason = None
        if not region_code:
            reason = "unknown_region_name"
        elif not all(positive(v) for v in values):
            reason = "invalid_or_missing_rent_value"
        elif (code, period.year, period.month) in keys:
            reason = "duplicate_lad_month"
        elif code in area and area[code] != (name, region_code):
            reason = "inconsistent_lad_name_or_region"
        if reason:
            bad.append({"source_row": row_number, "lad_code": code,
                        "month": period.strftime("%Y-%m"), "failure_reason": reason})
            continue
        keys.add((code, period.year, period.month))
        area[code] = (name, region_code)
        good.append(dict(zip(columns, [period.strftime("%Y-%m"), code, name,
                                       region_code, REGIONS[region_code], *values])))
    write_csv(ROOT / "data/processed/reference/rent_lad_monthly.csv", columns, good)
    write_csv(ROOT / "data/failed/rent_lad_invalid.csv",
              ["source_row", "lad_code", "month", "failure_reason"], bad)
    return area, {"areas": len(area), "accepted": len(good), "failed": len(bad)}


def income_reference(area):
    path = ROOT / "data/raw/regional_gdhi_local_authorities_1997_2024.xlsx"
    book = load_workbook(path, read_only=True, data_only=True)
    tables = {"Table 1": "gdhi_total_gbp_million",
              "Table 2": "population", "Table 3": "gdhi_per_head_gbp"}
    values = defaultdict(dict)
    bad = []
    for sheet_name, measure in tables.items():
        sheet = book[sheet_name]
        rows = sheet.iter_rows(min_row=4, values_only=True)
        header = next(rows)
        year_cols = [(i, int(v)) for i, v in enumerate(header)
                     if isinstance(v, str) and v.isdigit() and 2014 <= int(v) <= 2024]
        for source_row, row in enumerate(rows, 5):
            original_code = row[1]
            if not isinstance(original_code, str) or not original_code.startswith(LA_PREFIXES):
                continue
            code = OLD_TO_RENT_LAD.get(original_code, original_code)
            if code not in area:
                continue
            if original_code in RENAME_EXPECTED and row[2] != RENAME_EXPECTED[original_code]:
                raise RuntimeError(f"Unexpected renamed LAD: {original_code} / {row[2]}")
            for i, year in year_cols:
                number = row[i]
                if not positive(number):
                    bad.append({"sheet": sheet_name, "source_row": source_row,
                                "lad_code": code, "year": year,
                                "failure_reason": "invalid_or_missing_annual_measure"})
                else:
                    key = (code, year)
                    if measure in values[key] and values[key][measure] != number:
                        raise RuntimeError(f"Conflicting annual measure: {key} / {measure}")
                    values[key][measure] = number
    columns = ["lad_code", "lad_name", "region_code", "year", *tables.values()]
    good = []
    for code in sorted(area):
        for year in range(2014, 2025):
            measures = values.get((code, year), {})
            if not all(measure in measures for measure in tables.values()):
                bad.append({"sheet": "joined", "source_row": "", "lad_code": code,
                            "year": year, "failure_reason": "incomplete_annual_income_or_population"})
                continue
            name, region = area[code]
            good.append({"lad_code": code, "lad_name": name, "region_code": region,
                         "year": year, **measures})
    write_csv(ROOT / "data/processed/reference/gdhi_lad_annual.csv", columns, good)
    write_csv(ROOT / "data/failed/gdhi_lad_invalid.csv",
              ["sheet", "source_row", "lad_code", "year", "failure_reason"], bad)
    return {"accepted": len(good), "failed": len(bad), "expected": len(area) * 11}


def geography_reference(area):
    path = ROOT / "data/raw/Geography Lookup/ONSPD_AUG_2024.zip"
    postcode_path = ROOT / "data/processed/reference/postcode_lad.csv"
    lsoa_path = ROOT / "data/processed/reference/lsoa_lad.csv"
    postcode_path.parent.mkdir(parents=True, exist_ok=True)
    lsoa_counts = defaultdict(Counter)
    seen = {}
    stats = Counter()
    with zipfile.ZipFile(path) as archive, postcode_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["postcode_key", "lad_code"])
        names = sorted(n for n in archive.namelist()
                       if re.fullmatch(r"Data/multi_csv/ONSPD_AUG_2024_UK_[A-Z]+\.csv", n))
        for name in names:
            with archive.open(name) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                for row in reader:
                    code = OLD_TO_RENT_LAD.get(row["oslaua"], row["oslaua"])
                    if code not in area:
                        continue
                    key = re.sub(r"\s+", "", row["pcds"].upper())
                    if not key:
                        continue
                    if key in seen:
                        if seen[key] != code:
                            stats["conflicting_postcodes"] += 1
                        continue
                    seen[key] = code
                    writer.writerow([key, code])
                    stats["mapped_postcodes"] += 1
                    for vintage in ("lsoa11", "lsoa21"):
                        lsoa = row[vintage].strip()
                        if lsoa:
                            lsoa_counts[lsoa][code] += 1
    with lsoa_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["lsoa_code", "lad_code", "postcode_support", "total_support",
                         "dominance_fraction", "ambiguous"])
        for lsoa, counts in sorted(lsoa_counts.items()):
            code, support = counts.most_common(1)[0]
            total = sum(counts.values())
            ambiguous = len(counts) > 1
            writer.writerow([lsoa, code, support, total, round(support / total, 6),
                             str(ambiguous).lower()])
            stats["mapped_lsoa"] += 1
            if ambiguous:
                stats["ambiguous_lsoa"] += 1
            if support / total < 0.95:
                stats["low_confidence_lsoa"] += 1
    return dict(stats)


def main():
    area, rent = rent_reference()
    income = income_reference(area)
    geography = geography_reference(area)
    result = {"grain": "local_authority_month", "rent": rent,
              "income": income, "geography": geography,
              "explicit_code_crosswalk": OLD_TO_RENT_LAD,
              "income_source": "regional_gdhi_local_authorities_1997_2024.xlsx"}
    path = ROOT / "data/metadata/local_authority_reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if rent["areas"] != 316 or rent["accepted"] != 37920 or income["accepted"] != 3476:
        raise RuntimeError("Reference coverage did not match the verified source profile")


if __name__ == "__main__":
    main()
