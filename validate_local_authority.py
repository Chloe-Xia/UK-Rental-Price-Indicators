"""Independent validation of the local-authority/month modeling table."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


ROOT = Path(__file__).resolve().parent


def main():
    spark = SparkSession.builder.appName("UKRentalLocalAuthorityValidation").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        path = ROOT / "data/processed/local_authority/rent_panel"
        panel = spark.read.parquet(str(path))
        splits = {r["split"]: r["count"] for r in panel.groupBy("split").count().collect()}
        rows = panel.count()
        csv_path = ROOT / "data/processed/local_authority/rent_panel.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            csv_rows = sum(1 for _ in csv.DictReader(file))
        region_rows = Counter()
        for path in (ROOT / "data/processed/local_authority/by_region").glob("*.csv"):
            with path.open(newline="", encoding="utf-8") as file:
                region_rows[path.stem] = sum(1 for _ in csv.DictReader(file))
        checks = {
            "exactly_37920_rows": rows == 37920,
            "316_local_authorities": panel.select("lad_code").distinct().count() == 316,
            "120_months": panel.select("month").distinct().count() == 120,
            "no_duplicate_lad_month": panel.groupBy("lad_code", "month").count()
                .filter("count != 1").count() == 0,
            "expected_splits": splits == {"history": 3792, "train": 26544,
                                           "validation": 3792, "test": 3792},
            "all_rents_positive": panel.filter(F.col("rent_gbp_month").isNull() |
                (F.col("rent_gbp_month") <= 0)).count() == 0,
            "annual_income_lagged": panel.filter(F.col("feature_year") !=
                F.substring("month", 1, 4).cast("int") - 1).count() == 0,
            "all_income_population_present": panel.filter(
                F.col("gdhi_per_head_gbp").isNull() | F.col("population").isNull()).count() == 0,
            "incomplete_crime_not_zero_filled": panel.filter(
                (~F.col("crime_coverage_complete")) & F.col("crime_count").isNotNull()).count() == 0,
            "partial_uploads_not_counted": panel.filter(
                F.col("crime_partial_upload") & F.col("crime_count").isNotNull()).count() == 0,
            "recorded_crime_present_and_positive": panel.filter(
                F.col("recorded_crime_quarter_count").isNull() |
                (F.col("recorded_crime_quarter_count") <= 0)).count() == 0,
            "recorded_crime_constant_within_quarter": panel.groupBy("lad_code", "recorded_crime_quarter")
                .agg(F.countDistinct("recorded_crime_quarter_count").alias("n"), F.count("*").alias("months"))
                .filter("n != 1 or months != 3").count() == 0,
            "csv_matches_parquet": csv_rows == rows,
            "ten_region_csvs_match": len(region_rows) == 10 and sum(region_rows.values()) == rows,
            "renamed_lads_have_income": panel.filter(
                F.col("lad_code").isin("E08000038", "E08000039") &
                F.col("gdhi_per_head_gbp").isNotNull()).count() == 240,
        }
        result = {"passed": all(checks.values()), "rows": rows,
                  "split_counts": splits, "region_csv_rows": dict(region_rows), "checks": checks}
        output = ROOT / "data/metadata/local_authority_independent_validation.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
