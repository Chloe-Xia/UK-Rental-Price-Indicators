"""Rebuild the modeling dataset at local-authority × month grain.

Consumes validated event-level Parquet from run_pipeline.py, maps each event to
an ONS local authority, aggregates to the same grain as PIPR, then joins rent,
sales, crime, and lagged GDHI/population. Raw source files are never altered.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

from prepare_reference import REGIONS


ROOT = Path(__file__).resolve().parent
YEAR_RANGE = range(2015, 2025)
# A published file can still be a partial upload: a force-month with fewer than
# half the force's median monthly records (e.g. Greater Manchester 2022-08), or
# an authority with under a quarter of its median count because the source left
# most of its records without an LSOA (Leicestershire 2020-03).
PARTIAL_FORCE_RATIO = 0.5
PARTIAL_LAD_RATIO = 0.25


def save(df, path: Path):
    df.write.mode("overwrite").parquet(str(path))


def json_write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def failure_counts(spark, path: Path):
    return {r["failure_reason"]: r["count"] for r in
            spark.read.parquet(str(path)).groupBy("failure_reason").count().collect()}


def sales_year(spark, year: int, postcode, areas):
    source = spark.read.parquet(str(ROOT / f"data/processed/sales_transactions/year={year}"))
    source = source.withColumnRenamed("region_code", "source_region_code")
    mapped = (source.join(F.broadcast(postcode), "postcode_key", "left")
              .join(F.broadcast(areas), "lad_code", "left"))
    reason = (F.when(F.col("lad_code").isNull(), "postcode_not_mapped_to_rent_lad")
              .when(F.col("target_region_code").isNull(), "lad_missing_in_rent_source")
              .when(F.col("source_region_code") != F.col("target_region_code"),
                    "lad_region_mismatch"))
    checked = mapped.withColumn("failure_reason", reason)
    accepted = checked.filter(F.col("failure_reason").isNull())
    rejected = checked.filter(F.col("failure_reason").isNotNull())
    valid_path = ROOT / f"data/processed/local_authority/sales_events/year={year}"
    failed_path = ROOT / f"data/failed/local_authority/sales_year={year}"
    save(accepted.select("transaction_id", "sale_date", "month", "lad_code",
                         F.col("target_region_code").alias("region_code"), "price_gbp",
                         "postcode_key", "property_type", "category_type", "record_status",
                         "source_file"), valid_path)
    save(rejected.select("transaction_id", "sale_date", "month", "postcode_key",
                         "source_region_code", "lad_code", "target_region_code",
                         "source_file", "failure_reason"), failed_path)
    valid = spark.read.parquet(str(valid_path))
    baseline = valid.filter((F.col("category_type") == "A") & (F.col("record_status") == "A"))
    monthly = (baseline.groupBy("lad_code", "month")
               .agg(F.count("*").alias("sales_count"),
                    F.expr("percentile_approx(price_gbp, 0.5, 10000)").alias("sales_median_gbp"),
                    F.sum(F.when(F.col("property_type") == "F", 1).otherwise(0)).alias("flat_sales_count"),
                    F.expr("percentile_approx(case when property_type='F' then price_gbp end, 0.5, 10000)")
                    .alias("flat_sales_median_gbp")))
    save(monthly, ROOT / f"data/processed/local_authority/sales_monthly/year={year}")
    return {"accepted": valid.count(), "failed_by_reason": failure_counts(spark, failed_path),
            "baseline_standard_sales": baseline.count()}


def crime_year(spark, year: int, lsoa, areas):
    source = spark.read.parquet(str(ROOT / f"data/processed/crime_events/year={year}"))
    source = source.withColumnRenamed("region_code", "source_region_code")
    mapped = (source.join(F.broadcast(lsoa), "lsoa_code", "left")
              .join(F.broadcast(areas), "lad_code", "left"))
    reason = (F.when(F.col("lad_code").isNull(), "lsoa_not_mapped_to_rent_lad")
              .when(F.col("dominance_fraction") < 0.95, "ambiguous_lsoa_to_lad")
              .when(F.col("target_region_code").isNull(), "lad_missing_in_rent_source")
              .when(F.col("source_region_code") != F.col("target_region_code"),
                    "lad_region_mismatch"))
    checked = mapped.withColumn("failure_reason", reason)
    accepted = checked.filter(F.col("failure_reason").isNull())
    rejected = checked.filter(F.col("failure_reason").isNotNull())
    valid_path = ROOT / f"data/processed/local_authority/crime_events/year={year}"
    failed_path = ROOT / f"data/failed/local_authority/crime_year={year}"
    save(accepted.select("crime_id", "month", "force", "lad_code",
                         F.col("target_region_code").alias("region_code"), "lsoa_code",
                         "crime_type", "source_file"), valid_path)
    save(rejected.select("crime_id", "month", "force", "lsoa_code", "lad_code",
                         "dominance_fraction", "source_region_code", "target_region_code",
                         "source_file", "failure_reason"), failed_path)
    valid = spark.read.parquet(str(valid_path))
    force_monthly = (valid.groupBy("force", "lad_code", "month")
                     .agg(F.count("*").alias("crime_count_observed")))
    save(force_monthly, ROOT / f"data/processed/local_authority/crime_force_monthly/year={year}")
    return {"accepted": valid.count(), "failed_by_reason": failure_counts(spark, failed_path)}


def partial_uploads(by_force, home_map, present):
    force_totals = defaultdict(dict)
    for r in by_force.groupBy("force", "month").agg(F.sum("crime_count_observed").alias("n")).collect():
        force_totals[r["force"]][r["month"]] = r["n"]
    partial = {(force, month) for force, months in force_totals.items()
               for month, n in months.items()
               if n < PARTIAL_FORCE_RATIO * statistics.median(months.values())}
    lad_totals = defaultdict(dict)
    for r in by_force.groupBy("lad_code", "month").agg(F.sum("crime_count_observed").alias("n")).collect():
        lad_totals[r["lad_code"]][r["month"]] = r["n"]
    for lad, force in home_map.items():
        published = {month: n for month, n in lad_totals[lad].items()
                     if (month, force) in present and (month, force) not in partial}
        typical = statistics.median(published.values())
        partial |= {(lad, month) for month, n in published.items() if n < PARTIAL_LAD_RATIO * typical}
    return partial


def crime_coverage(spark, areas):
    paths = [str(ROOT / f"data/processed/local_authority/crime_force_monthly/year={y}")
             for y in YEAR_RANGE]
    by_force = spark.read.parquet(*paths)
    home = (by_force.groupBy("force", "lad_code")
            .agg(F.sum("crime_count_observed").alias("events"))
            .withColumn("rank", F.row_number().over(
                Window.partitionBy("lad_code").orderBy(F.desc("events"), F.asc("force"))))
            .filter(F.col("rank") == 1).select("lad_code", "force"))
    home_map = {r["lad_code"]: r["force"] for r in home.collect()}
    present = set()
    for year in YEAR_RANGE:
        for path in (ROOT / "data/raw/Crime Data").glob(f"{year}-??/*-street.csv"):
            match = re.fullmatch(r"(\d{4}-\d{2})-([a-z-]+)-street\.csv", path.name)
            if match:
                present.add((match.group(1), match.group(2)))
    partial = partial_uploads(by_force, home_map, present)
    area_rows = [(r["lad_code"], r["target_region_code"]) for r in areas.collect()]
    rows = []
    for year in YEAR_RANGE:
        for month_number in range(1, 13):
            month = f"{year}-{month_number:02d}"
            for lad, region in area_rows:
                force = home_map.get(lad, "")
                published = bool(force) and (month, force) in present
                incomplete = published and ((force, month) in partial or (lad, month) in partial)
                rows.append((lad, region, month, force, published and not incomplete, incomplete))
    schema = ["lad_code", "region_code", "month", "dominant_force",
              "crime_coverage_complete", "crime_partial_upload"]
    coverage = spark.createDataFrame(rows, schema)
    save(coverage, ROOT / "data/processed/local_authority/crime_coverage")
    with (ROOT / "data/metadata/local_authority_crime_coverage.csv").open(
            "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(schema)
        writer.writerows(rows)
    missing = Counter(region for _, region, _, _, complete, _ in rows if not complete)
    return by_force, coverage, {"lads_with_dominant_force": len(home_map),
                                "incomplete_lad_months": sum(missing.values()),
                                "partial_upload_lad_months": sum(row[-1] for row in rows),
                                "incomplete_by_region": dict(missing)}


def build_panel(spark, areas):
    by_force, coverage, coverage_stats = crime_coverage(spark, areas)
    crime = (by_force.groupBy("lad_code", "month")
             .agg(F.sum("crime_count_observed").alias("crime_count_observed")))
    sales = spark.read.parquet(*[str(ROOT / f"data/processed/local_authority/sales_monthly/year={y}")
                                 for y in YEAR_RANGE])
    rent = spark.read.option("header", True).csv(
        str(ROOT / "data/processed/reference/rent_lad_monthly.csv"))
    income = (spark.read.option("header", True).csv(
        str(ROOT / "data/processed/reference/gdhi_lad_annual.csv"))
              .select("lad_code", F.col("year").cast("int").alias("feature_year"),
                      "gdhi_total_gbp_million", "gdhi_per_head_gbp", "population"))
    recorded = (spark.read.option("header", True).csv(
        str(ROOT / "data/processed/reference/recorded_crime_lad_quarterly.csv"))
                .select("lad_code", F.col("quarter").alias("recorded_crime_quarter"),
                        F.col("recorded_crime_total").cast("double").alias("recorded_crime_quarter_count"),
                        F.col("force_unassigned_share").cast("double").alias("recorded_crime_unassigned_share"),
                        (F.col("allocated") == "True").alias("recorded_crime_allocated")))
    quarter = F.concat(F.substring("month", 1, 4), F.lit("-Q"),
                       F.ceil(F.substring("month", 6, 2).cast("int") / 3).cast("string"))
    panel = (rent.join(sales, ["lad_code", "month"], "left")
             .join(crime, ["lad_code", "month"], "left")
             .join(coverage.select("lad_code", "month", "dominant_force", "crime_coverage_complete",
                                   "crime_partial_upload"), ["lad_code", "month"], "left")
             .withColumn("recorded_crime_quarter", quarter)
             .join(recorded, ["lad_code", "recorded_crime_quarter"], "left")
             .withColumn("feature_year", F.substring("month", 1, 4).cast("int") - 1)
             .join(income, ["lad_code", "feature_year"], "left"))
    numeric = ["rent_gbp_month", "rent_index", "rent_one_bed_gbp",
               "rent_two_bed_gbp", "rent_three_bed_gbp", "rent_four_plus_bed_gbp",
               "rent_flat_gbp", "gdhi_total_gbp_million", "gdhi_per_head_gbp", "population"]
    for column in numeric:
        panel = panel.withColumn(column, F.col(column).cast("double"))
    panel = (panel.withColumn("crime_count_observed", F.coalesce("crime_count_observed", F.lit(0)))
             .withColumn("crime_count", F.when(F.col("crime_coverage_complete"),
                                                 F.col("crime_count_observed")))
             .withColumn("crime_per_1000_residents", F.when(
                 F.col("crime_coverage_complete") & (F.col("population") > 0),
                 F.col("crime_count") / F.col("population") * 1000))
             .withColumn("sales_data_present", F.col("sales_count").isNotNull())
             .withColumn("split", F.when(F.substring("month", 1, 4).cast("int") == 2023, "validation")
                         .when(F.substring("month", 1, 4).cast("int") == 2024, "test")
                         .when(F.substring("month", 1, 4).cast("int") == 2015, "history")
                         .otherwise("train")))
    columns = ["region_code", "region_name", "lad_code", "lad_name", "month", "split",
               "rent_gbp_month", "rent_index", "rent_one_bed_gbp", "rent_two_bed_gbp",
               "rent_three_bed_gbp", "rent_four_plus_bed_gbp", "rent_flat_gbp",
               "sales_median_gbp", "flat_sales_median_gbp", "sales_count", "flat_sales_count",
               "sales_data_present", "crime_count", "crime_count_observed",
               "crime_per_1000_residents", "crime_coverage_complete", "crime_partial_upload",
               "dominant_force", "recorded_crime_quarter", "recorded_crime_quarter_count",
               "recorded_crime_unassigned_share", "recorded_crime_allocated",
               "feature_year", "gdhi_total_gbp_million", "gdhi_per_head_gbp", "population"]
    output = ROOT / "data/processed/local_authority/rent_panel"
    panel.select(*columns).write.mode("overwrite").partitionBy("region_code").parquet(str(output))
    result = spark.read.parquet(str(output))
    csv_dir = ROOT / "data/processed/local_authority/rent_panel_csv_parts"
    (result.select(*columns).orderBy("region_code", "lad_code", "month")
     .coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_dir)))
    csv_path = ROOT / "data/processed/local_authority/rent_panel.csv"
    shutil.copyfile(next(csv_dir.glob("part-*.csv")), csv_path)
    region_dir = ROOT / "data/processed/local_authority/by_region"
    region_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        handles = {}
        try:
            for row in reader:
                code = row["region_code"]
                if code not in handles:
                    handle = (region_dir / f"{code}.csv").open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    handles[code] = (handle, writer)
                handles[code][1].writerow(row)
        finally:
            for handle, _ in handles.values():
                handle.close()
    row_count = result.count()
    duplicates = result.groupBy("lad_code", "month").count().filter("count != 1").count()
    splits = {r["split"]: r["count"] for r in result.groupBy("split").count().collect()}
    nulls = {column: result.filter(F.col(column).isNull()).count()
             for column in ["rent_gbp_month", "sales_median_gbp", "crime_count",
                            "recorded_crime_quarter_count", "gdhi_per_head_gbp", "population"]}
    metrics = {"rows": row_count, "unique_lads": result.select("lad_code").distinct().count(),
               "duplicate_lad_month_keys": duplicates, "splits": splits,
               "null_counts": nulls, "crime_coverage": coverage_stats}
    json_write(ROOT / "data/metadata/local_authority_panel_quality.json", metrics)
    if row_count != 37920 or duplicates != 0 or metrics["unique_lads"] != 316 or \
       nulls["recorded_crime_quarter_count"] or \
       splits != {"history": 3792, "train": 26544, "validation": 3792, "test": 3792}:
        raise RuntimeError(f"Local-authority panel grain failed: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip years with completed LA aggregates")
    parser.add_argument("--only-year", type=int, help="Process one year without assembling full panel")
    args = parser.parse_args()
    spark = (SparkSession.builder.appName("UKRentalLocalAuthorityPanel")
             .master("local[4]").config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "16").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    started = datetime.now(timezone.utc).isoformat()
    try:
        postcode = (spark.read.option("header", True).csv(
            str(ROOT / "data/processed/reference/postcode_lad.csv"))
            .select("postcode_key", "lad_code"))
        lsoa = (spark.read.option("header", True).csv(
            str(ROOT / "data/processed/reference/lsoa_lad.csv"))
            .select("lsoa_code", "lad_code", F.col("dominance_fraction").cast("double")))
        areas = (spark.read.option("header", True).csv(
            str(ROOT / "data/processed/reference/rent_lad_monthly.csv"))
            .select("lad_code", F.col("region_code").alias("target_region_code"))
            .dropDuplicates(["lad_code"]))
        years = [args.only_year] if args.only_year else YEAR_RANGE
        for year in years:
            done = (ROOT / f"data/processed/local_authority/sales_monthly/year={year}").exists() and \
                   (ROOT / f"data/processed/local_authority/crime_force_monthly/year={year}").exists()
            if args.resume and done:
                print(f"SKIP {year}", flush=True)
                continue
            print(f"START {year} LA sales", flush=True)
            sales = sales_year(spark, year, postcode, areas)
            print(f"START {year} LA crime", flush=True)
            crime = crime_year(spark, year, lsoa, areas)
            json_write(ROOT / f"data/metadata/local_authority_year_{year}.json",
                       {"year": year, "sales": sales, "crime": crime})
            print(f"DONE {year}", flush=True)
        if not args.only_year:
            quality = build_panel(spark, areas)
            json_write(ROOT / "data/metadata/local_authority_run_manifest.json",
                       {"started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "grain": "local_authority_month", "years": [2015, 2024],
                        "quality": quality,
                        "source_events": ["processed/sales_transactions", "processed/crime_events"],
                        "reference_files": ["rent_lad_monthly.csv", "gdhi_lad_annual.csv",
                                            "postcode_lad.csv", "lsoa_lad.csv",
                                            "recorded_crime_lad_quarterly.csv"]})
            print(json.dumps(quality, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
