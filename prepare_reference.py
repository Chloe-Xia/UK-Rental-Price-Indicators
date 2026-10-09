"""Prepare small, auditable reference tables from the original ONS files.

No source in data/raw is modified. This script can be rerun independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from openpyxl import load_workbook


REGIONS = {
    "E12000001": "North East", "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber", "E12000004": "East Midlands",
    "E12000005": "West Midlands", "E12000006": "East of England",
    "E12000007": "London", "E12000008": "South East",
    "E12000009": "South West", "W92000004": "Wales",
}
ITL_TO_REGION = dict(zip(
    ["TLC", "TLD", "TLE", "TLF", "TLG", "TLH", "TLI", "TLJ", "TLK", "TLL"],
    REGIONS,
))


def write_csv(path: Path, columns: list[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def rent_file(root: Path, start_year: int, end_year: int):
    source = root / "data/raw/price_index_of_private_rents.xlsx"
    sheet = load_workbook(source, read_only=True, data_only=True)["Table 1"]
    cols = ["month", "region_code", "region_name", "rent_gbp_month", "rent_index",
            "rent_one_bed_gbp", "rent_two_bed_gbp", "rent_three_bed_gbp",
            "rent_four_plus_bed_gbp", "rent_flat_gbp"]
    good, bad = [], []
    for line_no, row in enumerate(sheet.iter_rows(min_row=4, values_only=True), 4):
        code = row[1]
        if code not in REGIONS:
            continue
        month = row[0]
        if not isinstance(month, (date, datetime)) or not start_year <= month.year <= end_year:
            continue
        values = [row[i] for i in (7, 4, 11, 15, 19, 23, 39)]
        if not all(positive(v) for v in values):
            bad.append({"source": str(source), "source_row": line_no,
                        "region_code": code, "failure_reason": "invalid_or_missing_rent_value"})
            continue
        good.append(dict(zip(cols, [month.strftime("%Y-%m"), code, REGIONS[code], *values])))
    write_csv(root / "data/processed/reference/rent_monthly.csv", cols, good)
    write_csv(root / "data/failed/rent_invalid.csv",
              ["source", "source_row", "region_code", "failure_reason"], bad)
    return {"accepted": len(good), "failed": len(bad)}


def gdhi_file(root: Path, start_year: int, end_year: int):
    source = root / "data/raw/regional_gross_disposable_house_hold_income.xlsx"
    book = load_workbook(source, read_only=True, data_only=True)
    tables = {"Table 1": "gdhi_total_gbp_million",
              "Table 2": "population", "Table 3": "gdhi_per_head_gbp"}
    records = defaultdict(dict)
    bad = []
    for sheet_name, measure in tables.items():
        sheet = book[sheet_name]
        rows = sheet.iter_rows(min_row=4, values_only=True)
        headers = next(rows)
        year_columns = [(i, int(v)) for i, v in enumerate(headers)
                        if isinstance(v, str) and v.isdigit() and start_year <= int(v) <= end_year]
        for line_no, row in enumerate(rows, 5):
            if row[0] != "ITL1" or row[1] not in ITL_TO_REGION:
                continue
            code = ITL_TO_REGION[row[1]]
            for i, year in year_columns:
                value = row[i]
                if not positive(value):
                    bad.append({"source": str(source), "sheet": sheet_name,
                                "source_row": line_no, "region_code": code,
                                "year": year, "failure_reason": "invalid_or_missing_measure"})
                else:
                    records[(code, year)][measure] = value
    cols = ["region_code", "region_name", "year", *tables.values()]
    good = []
    for (code, year), measures in sorted(records.items()):
        if all(k in measures for k in tables.values()):
            good.append({"region_code": code, "region_name": REGIONS[code], "year": year, **measures})
        else:
            bad.append({"source": str(source), "sheet": "joined", "source_row": "",
                        "region_code": code, "year": year,
                        "failure_reason": "incomplete_annual_income_or_population"})
    write_csv(root / "data/processed/reference/gdhi_annual.csv", cols, good)
    write_csv(root / "data/failed/gdhi_invalid.csv",
              ["source", "sheet", "source_row", "region_code", "year", "failure_reason"], bad)
    return {"accepted": len(good), "failed": len(bad)}


def geography_file(root: Path):
    source = root / "data/raw/Geography Lookup/ONSPD_AUG_2024.zip"
    postcode_path = root / "data/processed/reference/postcode_region.csv"
    lsoa_path = root / "data/processed/reference/lsoa_region.csv"
    postcode_path.parent.mkdir(parents=True, exist_ok=True)
    lsoa_counts = defaultdict(Counter)
    postcode_regions = {}
    stats = Counter()
    with zipfile.ZipFile(source) as archive, postcode_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["postcode_key", "region_code"])
        members = sorted(n for n in archive.namelist()
                         if re.fullmatch(r"Data/multi_csv/ONSPD_AUG_2024_UK_[A-Z]+\.csv", n))
        for member in members:
            with archive.open(member) as stream:
                import io
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                for row in reader:
                    stats["raw_postcodes"] += 1
                    code = "W92000004" if row["ctry"] == "W92000004" else row["rgn"]
                    if code not in REGIONS:
                        continue
                    key = re.sub(r"\s+", "", row["pcds"].upper())
                    if not key:
                        continue
                    previous = postcode_regions.get(key)
                    if previous is not None:
                        if previous != code:
                            stats["conflicting_postcodes"] += 1
                        continue
                    postcode_regions[key] = code
                    writer.writerow([key, code])
                    stats["accepted_postcodes"] += 1
                    for vintage in ("lsoa11", "lsoa21"):
                        lsoa = row[vintage].strip()
                        if lsoa:
                            lsoa_counts[lsoa][code] += 1
    with lsoa_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["lsoa_code", "region_code", "postcode_support", "ambiguous"])
        for lsoa, counts in sorted(lsoa_counts.items()):
            code, support = counts.most_common(1)[0]
            ambiguous = len(counts) > 1
            if ambiguous:
                stats["ambiguous_lsoa"] += 1
            writer.writerow([lsoa, code, support, str(ambiguous).lower()])
            stats["accepted_lsoa"] += 1
    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = {"rent": rent_file(root, args.start_year, args.end_year),
               "gdhi": gdhi_file(root, args.start_year - 1, args.end_year),
               "geography": geography_file(root)}
    output = root / "data/metadata/reference_preparation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
