"""Ingest UK housing/crime source files and build an England/Wales monthly panel.

Requirements: Python 3.10+, pyspark 3.5, Java 17, openpyxl (for prepare_reference.py).
All writes stay under --root/data/{processed,failed,metadata}; raw is read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F, types as T

from prepare_reference import REGIONS


PPD_COLUMNS = ["transaction_id", "price_raw", "transfer_date_raw", "postcode_raw",
               "property_type", "old_new", "tenure", "paon", "saon", "street",
               "locality", "town_city", "district", "county", "category_type", "record_status"]
CRIME_COLUMNS = ["crime_id", "month_raw", "reported_by", "falls_within", "longitude",
                 "latitude", "location", "lsoa_code", "lsoa_name", "crime_type",
                 "last_outcome_category", "context"]


def schema(columns):
    return T.StructType([T.StructField(c, T.StringType(), True) for c in columns] +
                        [T.StructField("_corrupt_record", T.StringType(), True)])


def csv_source(spark, paths, columns, header):
    return (spark.read.option("header", header).option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .option("quote", '"').option("escape", '"').schema(schema(columns))
            .csv([str(p) for p in paths]))


def save(df, path: Path, fmt="parquet"):
    df.write.mode("overwrite").format(fmt).save(str(path))


def write_stats(root, name, data):
    path = root / "data/metadata" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def price_year(spark, root, year, postcode):
    source_dir = root / "data/raw/Price Paid Data"
    paths = sorted(source_dir.glob(f"pp-{year}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No price paid CSVs for {year}")
    raw = csv_source(spark, paths, PPD_COLUMNS, False).withColumn("source_file", F.input_file_name())
    cleaned = (raw.withColumn("price_gbp", F.col("price_raw").cast("double"))
               .withColumn("sale_date", F.to_date(F.substring("transfer_date_raw", 1, 10), "yyyy-MM-dd"))
               .withColumn("postcode_key", F.regexp_replace(F.upper(F.trim("postcode_raw")), r"\s+", ""))
               .join(F.broadcast(postcode), "postcode_key", "left")
               .withColumn("month", F.date_format("sale_date", "yyyy-MM")))
    reason = (F.when(F.col("_corrupt_record").isNotNull(), "malformed_csv")
              .when(F.col("transaction_id").isNull() | (F.trim("transaction_id") == ""), "missing_transaction_id")
              .when(F.col("sale_date").isNull() | (F.year("sale_date") != year), "invalid_or_wrong_year_date")
              .when(F.col("price_gbp").isNull() | (F.col("price_gbp") <= 0), "invalid_price")
              .when(F.col("postcode_key").isNull() | (F.col("postcode_key") == ""), "missing_postcode")
              .when(F.col("region_code").isNull(), "postcode_not_mapped_to_england_wales")
              .when(~F.col("property_type").isin("D", "S", "T", "F", "O"), "invalid_property_type")
              .when(~F.col("category_type").isin("A", "B"), "invalid_category")
              .when(~F.col("record_status").isin("A", "C", "D"), "invalid_record_status"))
    checked = cleaned.withColumn("failure_reason", reason)
    valid = checked.filter(F.col("failure_reason").isNull())
    failed = checked.filter(F.col("failure_reason").isNotNull())
    base = root / "data/processed/sales_transactions" / f"year={year}"
    save(valid.select("transaction_id", "sale_date", "month", "region_code", "price_gbp",
                      "postcode_key", "property_type", "old_new", "tenure", "category_type",
                      "record_status", "source_file"), base)
    save(failed.select(*PPD_COLUMNS, "source_file", "failure_reason"),
         root / "data/failed/price_paid" / f"year={year}")
    accepted = spark.read.parquet(str(base))
    # Only standard-market additions (A/A) are comparable for the baseline.
    baseline = accepted.filter((F.col("category_type") == "A") & (F.col("record_status") == "A"))
    aggregate = (baseline.groupBy("region_code", "month")
                 .agg(F.count("*").alias("sales_count"),
                      F.expr("percentile_approx(price_gbp, 0.5, 10000)").alias("sales_median_gbp"),
                      F.sum(F.when(F.col("property_type") == "F", 1).otherwise(0)).alias("flat_sales_count"),
                      F.expr("percentile_approx(case when property_type='F' then price_gbp end, 0.5, 10000)")
                      .alias("flat_sales_median_gbp")))
    save(aggregate, root / "data/processed/sales_monthly" / f"year={year}")
    counts = accepted.groupBy("category_type", "record_status").count().collect()
    failed_counts = (spark.read.parquet(str(root / "data/failed/price_paid" / f"year={year}"))
                     .groupBy("failure_reason").count().collect())
    return {"source_files": len(paths), "accepted_by_category_status":
            {f"{r['category_type']}/{r['record_status']}": r['count'] for r in counts},
            "failed_by_reason": {r['failure_reason']: r['count'] for r in failed_counts}}


def crime_year(spark, root, year, lsoa):
    source_dir = root / "data/raw/Crime Data"
    paths = sorted(p for p in source_dir.glob(f"{year}-??/*-street.csv")
                   if "-northern-ireland-" not in p.name and "-btp-" not in p.name)
    if not paths:
        raise FileNotFoundError(f"No street-crime CSVs for {year}")
    raw = csv_source(spark, paths, CRIME_COLUMNS, True).withColumn("source_file", F.input_file_name())
    cleaned = (raw.withColumn("source_month", F.regexp_extract("source_file", r"/(\d{4}-\d{2})-[^/]*-street\.csv$", 1))
               .withColumn("force", F.regexp_extract("source_file", r"/\d{4}-\d{2}-([a-z-]+)-street\.csv$", 1))
               .withColumnRenamed("month_raw", "month")
               .join(F.broadcast(lsoa), "lsoa_code", "left"))
    reason = (F.when(F.col("_corrupt_record").isNotNull(), "malformed_csv")
              .when(~F.col("month").rlike(r"^\d{4}-\d{2}$") | (F.col("month") != F.col("source_month")),
                    "invalid_or_mismatched_month")
              .when(F.col("crime_type").isNull() | (F.trim("crime_type") == ""), "missing_crime_type")
              .when(F.col("lsoa_code").isNull() | (F.trim("lsoa_code") == ""), "missing_lsoa")
              .when(F.col("region_code").isNull(), "lsoa_not_mapped_to_england_wales"))
    checked = cleaned.withColumn("failure_reason", reason)
    valid = checked.filter(F.col("failure_reason").isNull())
    failed = checked.filter(F.col("failure_reason").isNotNull())
    base = root / "data/processed/crime_events" / f"year={year}"
    save(valid.select("crime_id", "month", "force", "region_code", "lsoa_code", "crime_type",
                      "longitude", "latitude", "source_file"), base)
    fail_path = root / "data/failed/crime" / f"year={year}"
    save(failed.select(*CRIME_COLUMNS[0:1], "month", *CRIME_COLUMNS[2:],
                       "source_file", "failure_reason"), fail_path)
    accepted = spark.read.parquet(str(base))
    force_counts = (accepted.groupBy("force", "month", "region_code")
                    .agg(F.count("*").alias("crime_count")))
    save(force_counts, root / "data/processed/crime_force_monthly" / f"year={year}")
    failed_counts = (spark.read.parquet(str(fail_path)).groupBy("failure_reason").count().collect())
    return {"source_files": len(paths),
            "accepted": accepted.count(),
            "failed_by_reason": {r['failure_reason']: r['count'] for r in failed_counts}}


def build_coverage(spark, root, start_year, end_year):
    paths = [str(root / "data/processed/crime_force_monthly" / f"year={year}")
             for year in range(start_year, end_year + 1)]
    counts = spark.read.parquet(*paths)
    homes = (counts.groupBy("force", "region_code").agg(F.sum("crime_count").alias("n"))
             .withColumn("rank", F.row_number().over(Window.partitionBy("force").orderBy(F.desc("n"))))
             .filter(F.col("rank") == 1).select("force", "region_code"))
    home_map = {r["force"]: r["region_code"] for r in homes.collect()}
    present = set()
    for year in range(start_year, end_year + 1):
        for path in (root / "data/raw/Crime Data").glob(f"{year}-??/*-street.csv"):
            match = re.fullmatch(r"(\d{4}-\d{2})-([a-z-]+)-street\.csv", path.name)
            if match and match.group(2) in home_map:
                present.add((match.group(1), match.group(2)))
    expected = defaultdict(list)
    for force, code in home_map.items():
        expected[code].append(force)
    rows = []
    for year in range(start_year, end_year + 1):
        for month_number in range(1, 13):
            month = f"{year}-{month_number:02d}"
            for code in REGIONS:
                missing = sorted(force for force in expected[code] if (month, force) not in present)
                rows.append((code, month, len(expected[code]), len(missing),
                             ";".join(missing), len(missing) == 0))
    coverage = spark.createDataFrame(rows, ["region_code", "month", "expected_forces",
                                            "missing_force_count", "missing_forces",
                                            "crime_coverage_complete"])
    save(coverage, root / "data/processed/crime_coverage")
    # Small human-readable inventory, including missing force months.
    with (root / "data/metadata/crime_force_coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["region_code", "month", "expected_forces", "missing_force_count",
                         "missing_forces", "crime_coverage_complete"])
        writer.writerows(rows)
    return counts, coverage, {"forces": len(home_map), "incomplete_region_months":
                              sum(not row[-1] for row in rows)}


def build_panel(spark, root, start_year, end_year):
    counts, coverage, coverage_stats = build_coverage(spark, root, start_year, end_year)
    crime = counts.groupBy("region_code", "month").agg(F.sum("crime_count").alias("crime_count_observed"))
    sales = spark.read.parquet(*[str(root / "data/processed/sales_monthly" / f"year={y}")
                                for y in range(start_year, end_year + 1)])
    rent = (spark.read.option("header", True).csv(str(root / "data/processed/reference/rent_monthly.csv"))
            .filter((F.substring("month", 1, 4).cast("int") >= start_year) &
                    (F.substring("month", 1, 4).cast("int") <= end_year)))
    annual = (spark.read.option("header", True)
              .csv(str(root / "data/processed/reference/gdhi_annual.csv"))
              .withColumnRenamed("year", "feature_year")
              .withColumnRenamed("region_name", "income_region_name"))
    panel = (rent.join(sales, ["region_code", "month"], "left")
             .join(crime, ["region_code", "month"], "left")
             .join(coverage, ["region_code", "month"], "left")
             .withColumn("feature_year", F.substring("month", 1, 4).cast("int") - 1)
             .join(annual, ["region_code", "feature_year"], "left")
             .withColumn("crime_count", F.when(F.col("crime_coverage_complete"), F.col("crime_count_observed")))
             .withColumn("sales_coverage_complete", F.col("sales_count").isNotNull())
             .withColumn("split", F.when(F.substring("month", 1, 4).cast("int") == 2023, "validation")
                         .when(F.substring("month", 1, 4).cast("int") == 2024, "test")
                         .when(F.substring("month", 1, 4).cast("int") <= 2015, "history")
                         .otherwise("train")))
    numeric = ["rent_gbp_month", "rent_index", "rent_one_bed_gbp", "rent_two_bed_gbp",
               "rent_three_bed_gbp", "rent_four_plus_bed_gbp", "rent_flat_gbp",
               "gdhi_total_gbp_million", "population", "gdhi_per_head_gbp"]
    for col in numeric:
        panel = panel.withColumn(col, F.col(col).cast("double"))
    panel = panel.withColumn("crime_per_1000_residents", F.when(
        F.col("crime_coverage_complete") & (F.col("population") > 0),
        F.col("crime_count") / F.col("population") * 1000))
    # The sales feature is contemporaneous; models that forecast ahead should lag it explicitly.
    panel = panel.select("region_code", "region_name", "month", "split", "rent_gbp_month",
                         "rent_index", "rent_one_bed_gbp", "rent_two_bed_gbp",
                         "rent_three_bed_gbp", "rent_four_plus_bed_gbp", "rent_flat_gbp",
                         "sales_median_gbp", "flat_sales_median_gbp", "sales_count",
                         "flat_sales_count", "sales_coverage_complete", "crime_count",
                         "crime_count_observed", "crime_per_1000_residents", "crime_coverage_complete",
                         "expected_forces", "missing_force_count", "missing_forces", "feature_year",
                         "gdhi_total_gbp_million", "gdhi_per_head_gbp", "population")
    save(panel, root / "data/processed/regional_rent_panel")
    result = spark.read.parquet(str(root / "data/processed/regional_rent_panel"))
    csv_dir = root / "data/processed/regional_rent_panel_csv"
    (result.orderBy("region_code", "month").coalesce(1).write.mode("overwrite")
     .option("header", True).csv(str(csv_dir)))
    shutil.copyfile(next(csv_dir.glob("part-*.csv")),
                    root / "data/processed/regional_rent_panel.csv")
    n = result.count()
    dup = result.groupBy("region_code", "month").count().filter("count != 1").count()
    split_counts = {r["split"]: r["count"] for r in result.groupBy("split").count().collect()}
    null_counts = {col: result.filter(F.col(col).isNull()).count()
                   for col in ["rent_gbp_month", "sales_median_gbp", "crime_count",
                               "gdhi_per_head_gbp", "population"]}
    write_stats(root, "panel_quality.json", {"rows": n, "duplicate_region_month_keys": dup,
                                              "split_counts": split_counts, "null_counts": null_counts,
                                              "coverage": coverage_stats})
    if dup or n != len(REGIONS) * 12 * (end_year - start_year + 1):
        raise RuntimeError(f"Panel grain check failed: {n} rows, {dup} duplicate keys")
    return {"rows": n, "split_counts": split_counts, "null_counts": null_counts,
            "coverage": coverage_stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--resume", action="store_true", help="Skip years whose processed outputs exist")
    parser.add_argument("--only-year", type=int, help="Process one year for a smoke test, without panel")
    parser.add_argument("--crime-only", action="store_true", help="Rebuild crime years and panel without redoing sales")
    args = parser.parse_args()
    root = args.root.resolve()
    spark = (SparkSession.builder.appName("UKRentalPriceIndicatorsIngestion")
             .master("local[4]").config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.sql.files.maxPartitionBytes", "134217728")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    started = datetime.now(timezone.utc).isoformat()
    try:
        postcode = (spark.read.option("header", True)
                    .csv(str(root / "data/processed/reference/postcode_region.csv"))
                    .select("postcode_key", "region_code"))
        lsoa = (spark.read.option("header", True)
                .csv(str(root / "data/processed/reference/lsoa_region.csv"))
                .select("lsoa_code", "region_code"))
        years = [args.only_year] if args.only_year else list(range(args.start_year, args.end_year + 1))
        for year in years:
            if args.resume and (root / f"data/processed/sales_monthly/year={year}").exists() and \
               (root / f"data/processed/crime_force_monthly/year={year}").exists():
                print(f"SKIP {year}: processed outputs already exist", flush=True)
                continue
            if args.crime_only:
                prior = root / "data/metadata" / f"year_{year}.json"
                sales_stats = json.loads(prior.read_text())["sales"]
            else:
                print(f"START {year} price paid", flush=True)
                sales_stats = price_year(spark, root, year, postcode)
            print(f"START {year} street crime", flush=True)
            crime_stats = crime_year(spark, root, year, lsoa)
            write_stats(root, f"year_{year}.json", {"year": year, "sales": sales_stats, "crime": crime_stats})
            print(f"DONE {year}", flush=True)
        if not args.only_year:
            panel_stats = build_panel(spark, root, args.start_year, args.end_year)
            write_stats(root, "run_manifest.json", {"started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "years": [args.start_year, args.end_year], "panel": panel_stats,
                        "raw_unchanged": True, "excluded_sources": ["house_hold_debt.xlsx", "total_wealth.xlsx"],
                        "excluded_crime_types": ["outcomes CSV", "stop-and-search CSV",
                                                 "British Transport Police", "Northern Ireland"]})
            print(json.dumps(panel_stats, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
