"""Independent reproducible checks for the final regional-month dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    spark = SparkSession.builder.appName("UKRentalPanelValidation").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        panel = spark.read.parquet(str(root / "data/processed/regional_rent_panel"))
        rows = panel.count()
        duplicates = panel.groupBy("region_code", "month").count().filter("count != 1").count()
        checks = {
            "exactly_1200_region_months": rows == 1200,
            "no_duplicate_region_month": duplicates == 0,
            "ten_regions": panel.select("region_code").distinct().count() == 10,
            "all_months_2015_to_2024": panel.select("month").distinct().count() == 120,
            "rent_positive": panel.filter(F.col("rent_gbp_month").isNull() |
                                          (F.col("rent_gbp_month") <= 0)).count() == 0,
            "annual_feature_lagged": panel.filter(F.col("feature_year") !=
                 F.substring("month", 1, 4).cast("int") - 1).count() == 0,
            "crime_missing_when_incomplete": panel.filter(
                 (~F.col("crime_coverage_complete")) & F.col("crime_count").isNotNull()).count() == 0,
            "sales_nonnegative": panel.filter(F.col("sales_count") < 0).count() == 0,
            "north_west_2024_crime_incomplete": panel.filter(
                 (F.col("region_code") == "E12000002") & F.col("month").startswith("2024") &
                 (~F.col("crime_coverage_complete"))).count() == 12,
        }
        split_counts = {r["split"]: r["count"] for r in panel.groupBy("split").count().collect()}
        checks["expected_split_counts"] = split_counts == {
            "history": 120, "train": 840, "validation": 120, "test": 120}
        force_counts = spark.read.parquet(*[str(root / f"data/processed/crime_force_monthly/year={y}")
                                            for y in range(2015, 2025)])
        checks["btp_and_northern_ireland_excluded"] = force_counts.filter(
            F.col("force").isin("btp", "northern-ireland")).count() == 0
        result = {"passed": all(checks.values()), "row_count": rows,
                  "split_counts": split_counts, "checks": checks}
        path = root / "data/metadata/independent_validation.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
