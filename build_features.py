"""Engineer modeling features on the local-authority × month rent panel.

Inputs are the outputs of build_local_authority_panel.py and
prepare_lad_geography.py. A feature at month t uses information up to t only;
rolling windows are trailing and annual measures are already lagged to Y-1.
No imputation, scaling, or encoding is fitted here: those must be fitted on
training rows model-side. data/metadata/feature_dictionary.csv states which
columns suit an association study and which suit forward prediction.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parent
LA = ROOT / "data/processed/local_authority"
CRIME_GROUPS = {
    "violence": ["Violence and sexual offences", "Robbery"],
    "asb": ["Anti-social behaviour"],
    "burglary": ["Burglary"],
    "theft": ["Other theft", "Shoplifting", "Bicycle theft", "Theft from the person",
              "Vehicle crime"],
    "damage": ["Criminal damage and arson"],
    "drugs": ["Drugs"],
    "public_order": ["Public order", "Possession of weapons"],
    "other": ["Other crime"],
}
PROPERTY_TYPES = {"D": "detached", "S": "semi", "T": "terraced", "F": "flat"}
LAD_TYPES = {"E06": "unitary", "E07": "district", "E08": "metropolitan",
             "E09": "london_borough", "W06": "welsh_unitary"}
MIN_CRIME_MONTHS = 6
# Street-crime gaps (missing or partial months) of up to this many consecutive
# months are filled from the same month a year earlier, scaled by the regional
# year-on-year change. Longer gaps (Greater Manchester 2019-07 to 2024-12) stay
# null; the Home Office recorded-crime features cover them.
MAX_IMPUTED_GAP = 12
RECORDED_GROUPS = ["violence", "sexual", "robbery", "burglary", "theft", "damage", "drugs",
                   "public_order", "misc"]

# role -> (use_association, use_forecast)
ROLE_USE = {"id": (False, False), "target": (False, False), "target_variant": (False, False),
            "predictor": (True, True), "contemporaneous": (True, False),
            "lagged": (False, True), "quality_flag": (True, True)}
FEATURES = [
    ("region_code", "id", "id", "ONS region code; also usable as a categorical region effect"),
    ("region_name", "id", "id", "Region name"),
    ("lad_code", "id", "id", "Local authority code (primary key with month)"),
    ("lad_name", "id", "id", "Local authority name"),
    ("country", "id", "predictor", "England or Wales"),
    ("dominant_force", "id", "id", "police.uk force supplying most of the authority's street crime"),
    ("month", "id", "id", "YYYY-MM (primary key with lad_code)"),
    ("year", "time", "predictor", "Calendar year"),
    ("split", "id", "id", "history (2015) / train (2016-2022) / validation (2023) / test (2024)"),
    ("rent_gbp_month", "rent", "target", "Published ONS PIPR mean monthly private rent, nominal GBP"),
    ("log_rent", "rent", "target_variant", "Natural log of rent_gbp_month"),
    ("rent_yoy_log_change", "rent", "target_variant", "log rent(t) - log rent(t-12)"),
    ("rent_relative_to_national", "rent", "target_variant",
     "rent / unweighted mean rent of all 316 local authorities in the same month"),
    ("rent_relative_to_region", "rent", "target_variant",
     "rent / unweighted mean rent of local authorities in the same region and month"),
    ("t_index", "time", "predictor", "Months since 2015-01 (0-119)"),
    ("month_of_year", "time", "predictor", "1-12"),
    ("month_sin", "time", "predictor", "sin(2*pi*(month_of_year-1)/12)"),
    ("month_cos", "time", "predictor", "cos(2*pi*(month_of_year-1)/12)"),
    ("covid_restrictions", "time", "predictor", "1 for 2020-03 to 2021-03"),
    ("transaction_tax_holiday", "time", "predictor",
     "1 during the SDLT holiday (England 2020-07 to 2021-09) or LTT holiday (Wales 2020-07 to 2021-06)"),
    ("lad_type", "geography", "predictor",
     "unitary / district / metropolitan / london_borough / welsh_unitary, from the code prefix"),
    ("is_london", "geography", "predictor", "1 for London boroughs"),
    ("centroid_lat", "geography", "predictor", "Mean latitude of live residential postcodes (ONSPD Aug 2024)"),
    ("centroid_lon", "geography", "predictor", "Mean longitude of live residential postcodes"),
    ("dist_to_london_km", "geography", "predictor", "Great-circle distance from centroid to Charing Cross"),
    ("log_dist_to_london_km", "geography", "predictor", "log(1 + dist_to_london_km)"),
    ("urban_postcode_share", "geography", "predictor",
     "Share of live residential postcodes in an urban 2011 rural-urban class"),
    ("imd_deprivation_score", "geography", "predictor",
     "1 - mean postcode IMD rank / national max rank; higher = more deprived. "
     "England (IMD 2019) and Wales (WIMD) are separate indices"),
    ("gdhi_per_head_gbp", "income_population", "predictor", "GDHI per head in year Y-1"),
    ("log_gdhi_per_head", "income_population", "predictor", "log gdhi_per_head_gbp"),
    ("population", "income_population", "predictor", "Mid-year population in year Y-1"),
    ("log_population", "income_population", "predictor", "log population"),
    ("population_growth", "income_population", "predictor", "log population(Y-1) - log population(Y-2)"),
    ("gdhi_per_head_growth", "income_population", "predictor",
     "log GDHI per head(Y-1) - log GDHI per head(Y-2)"),
    ("sales_median_gbp", "housing_market", "contemporaneous",
     "Median standard-market (A/A) sale price in month t"),
    ("log_sales_median", "housing_market", "contemporaneous", "log sales_median_gbp"),
    ("sales_median_roll3", "housing_market", "contemporaneous", "Mean of monthly sale medians, t-2..t"),
    ("sales_median_roll12", "housing_market", "contemporaneous", "Mean of monthly sale medians, t-11..t"),
    ("log_sales_median_roll12", "housing_market", "contemporaneous", "log sales_median_roll12"),
    ("sales_median_yoy_log_change", "housing_market", "contemporaneous",
     "log sales_median_roll3(t) - log sales_median_roll3(t-12)"),
    ("flat_sales_median_roll12", "housing_market", "contemporaneous",
     "Mean of monthly flat sale medians, t-11..t (at least 6 months with flat sales)"),
    ("sales_count", "housing_market", "contemporaneous", "Standard-market sales in month t"),
    ("sales_per_1000_residents_roll12", "housing_market", "contemporaneous",
     "Standard-market sales over t-11..t per 1000 residents"),
    *[(f"{name}_share_roll12", "housing_market", "contemporaneous",
       f"Share of standard-market sales over t-11..t that are {name}") for name in PROPERTY_TYPES.values()],
    ("new_build_share_roll12", "housing_market", "contemporaneous",
     "Share of standard-market sales over t-11..t that are new builds"),
    ("leasehold_share_roll12", "housing_market", "contemporaneous",
     "Share of standard-market sales over t-11..t that are leasehold"),
    ("price_to_income_ratio", "housing_market", "contemporaneous",
     "sales_median_roll12 / gdhi_per_head_gbp (per-head, not per-household, income)"),
    ("crime_per_1000", "crime", "contemporaneous",
     "police.uk street crimes in month t per 1000 residents (observed, or imputed if crime_imputed)"),
    ("crime_per_1000_roll12", "crime", "contemporaneous",
     "Mean monthly street crimes over valid months in t-11..t per 1000 residents"),
    ("crime_yoy_log_change", "crime", "contemporaneous",
     "log mean crime count t-2..t minus the same 12 months earlier"),
    *[(f"crime_{group}_per_1000_roll12", "crime", "contemporaneous",
       f"Mean monthly {' / '.join(types)} over valid months in t-11..t per 1000 residents")
      for group, types in CRIME_GROUPS.items()],
    ("recorded_crime_per_1000_roll4q", "recorded_crime", "contemporaneous",
     "Home Office police recorded crime (excl. fraud) over the 4 quarters ending with the quarter "
     "containing t, as a monthly average per 1000 residents"),
    ("recorded_crime_yoy_log_change", "recorded_crime", "contemporaneous",
     "log recorded crime in the quarter containing t minus the same quarter a year earlier"),
    *[(f"recorded_{group}_per_1000_roll4q", "recorded_crime", "contemporaneous",
       f"Recorded {group.replace('_', ' ')} offences, 4 quarters ending with the quarter containing t, "
       "monthly average per 1000 residents") for group in RECORDED_GROUPS],
    ("neighbour_log_sales_median_roll12", "spatial", "contemporaneous",
     "Mean log_sales_median_roll12 of the 5 nearest local authorities"),
    ("neighbour_crime_per_1000_roll12", "spatial", "contemporaneous",
     "Mean crime_per_1000_roll12 of the 5 nearest local authorities"),
    ("rent_lag1", "rent_history", "lagged", "rent_gbp_month at t-1"),
    ("rent_lag3", "rent_history", "lagged", "rent_gbp_month at t-3"),
    ("rent_lag12", "rent_history", "lagged", "rent_gbp_month at t-12"),
    ("log_rent_lag1", "rent_history", "lagged", "log rent at t-1"),
    ("rent_mom_log_change_lag1", "rent_history", "lagged", "log rent(t-1) - log rent(t-2)"),
    ("rent_yoy_log_change_lag1", "rent_history", "lagged", "log rent(t-1) - log rent(t-13)"),
    ("rent_to_income_lag1", "rent_history", "lagged", "12 * rent_lag1 / gdhi_per_head_gbp"),
    ("neighbour_rent_lag1", "spatial", "lagged", "Mean rent_lag1 of the 5 nearest local authorities"),
    ("neighbour_rent_yoy_log_change_lag1", "spatial", "lagged",
     "Mean rent_yoy_log_change_lag1 of the 5 nearest local authorities"),
    ("rent_lag1_vs_neighbours", "spatial", "lagged", "log rent_lag1 - log neighbour_rent_lag1"),
    ("crime_data_ok", "crime", "quality_flag",
     "1 if the dominant force published month t and the upload is not partial"),
    ("crime_partial_upload", "crime", "quality_flag",
     "1 if the dominant force published < 50% of its median monthly records, or the local "
     "authority received < 25% of its median monthly count"),
    ("crime_imputed", "crime", "quality_flag",
     "1 if street crime for month t is imputed: same month a year earlier x regional "
     "year-on-year change, for gaps of at most 12 consecutive months"),
    ("crime_months_in_roll12", "crime", "quality_flag",
     "Observed or imputed street-crime months within t-11..t"),
    ("recorded_crime_allocated", "recorded_crime", "quality_flag",
     "1 if the CSP covers several local authorities and was split by population"),
    ("recorded_crime_unassigned_share", "recorded_crime", "quality_flag",
     "Share of the force's offences in that quarter with no CSP, redistributed to CSPs"),
    *[(f"{name}_prevq", "recorded_crime", "lagged",
       f"{name} for the last quarter completed before month t")
      for name in ["recorded_crime_per_1000_roll4q", "recorded_crime_yoy_log_change",
                   *[f"recorded_{group}_per_1000_roll4q" for group in RECORDED_GROUPS]]],
]
LAG1_SOURCES = ["log_sales_median_roll12", "sales_median_yoy_log_change",
                "sales_per_1000_residents_roll12", "flat_share_roll12", "new_build_share_roll12",
                "leasehold_share_roll12", "price_to_income_ratio", "crime_per_1000_roll12",
                "crime_yoy_log_change", *[f"crime_{g}_per_1000_roll12" for g in CRIME_GROUPS],
                "neighbour_log_sales_median_roll12", "neighbour_crime_per_1000_roll12"]
FEATURES += [(f"{name}_lag1", group, "lagged", f"{name} at t-1")
             for name, group, _, _ in FEATURES if name in LAG1_SOURCES]
# Year-on-year changes of 3-month windows start at 2016-03, so they are not listed.
REQUIRED_OUTSIDE_HISTORY = ["rent_lag12", "rent_yoy_log_change", "log_sales_median_roll12",
                            "population_growth", "neighbour_rent_lag1", "price_to_income_ratio",
                            "recorded_crime_yoy_log_change_prevq"]
REQUIRED_ALL_ROWS = ["recorded_crime_per_1000_roll4q", "recorded_crime_yoy_log_change",
                     "recorded_crime_per_1000_roll4q_prevq",
                     *[f"recorded_{group}_per_1000_roll4q" for group in RECORDED_GROUPS]]


def crime_by_group(spark):
    lookup = {crime_type: group for group, types in CRIME_GROUPS.items() for crime_type in types}
    mapping = F.create_map(*[F.lit(value) for pair in lookup.items() for value in pair])
    counts = (spark.read.parquet(str(LA / "crime_events"))
              .withColumn("crime_group", mapping[F.col("crime_type")])
              .groupBy("lad_code", "month").pivot("crime_group", list(CRIME_GROUPS)).count())
    return counts.select("lad_code", "month", *[F.coalesce(F.col(g), F.lit(0)).alias(f"crime_{g}_count")
                                                for g in CRIME_GROUPS])


def recorded_quarterly(spark):
    measures = {"recorded_crime": "recorded_crime_total",
                **{f"recorded_{g}": f"recorded_{g}" for g in RECORDED_GROUPS}}
    frame = (reference_csv(spark, "recorded_crime_lad_quarterly.csv", measures.values())
             .withColumn("quarter_index", F.col("year").cast("int") * 4 + F.col("quarter_number").cast("int")))
    by_quarter = Window.partitionBy("lad_code").orderBy("quarter_index")
    last4 = by_quarter.rowsBetween(-3, 0)
    for name, source in measures.items():
        frame = frame.withColumn(f"{name}_roll4q",
                                 F.when(F.count(source).over(last4) == 4, F.sum(source).over(last4)))
    frame = frame.withColumn("recorded_crime_yoy_log_change", F.log("recorded_crime_total") -
                             F.lag(F.log("recorded_crime_total"), 4).over(by_quarter))
    current = [f"{name}_roll4q" for name in measures] + ["recorded_crime_yoy_log_change"]
    for name in current:
        frame = frame.withColumn(f"{name}_prevq", F.lag(name, 1).over(by_quarter))
    return frame.select("lad_code", F.col("quarter").alias("recorded_crime_quarter"),
                        *current, *[f"{name}_prevq" for name in current])


def sales_mix(spark):
    events = (spark.read.parquet(str(LA / "sales_events"))
              .filter((F.col("category_type") == "A") & (F.col("record_status") == "A"))
              .select("transaction_id", "year", "lad_code", "month", "property_type"))
    attributes = (spark.read.parquet(str(ROOT / "data/processed/sales_transactions"))
                  .select("transaction_id", "year", "old_new", "tenure"))
    counts = [F.count("*").alias("mix_sales_count"),
              *[F.sum((F.col("property_type") == code).cast("int")).alias(f"{name}_mix_count")
                for code, name in PROPERTY_TYPES.items()],
              F.sum((F.col("old_new") == "Y").cast("int")).alias("new_build_mix_count"),
              F.sum((F.col("tenure") == "L").cast("int")).alias("leasehold_mix_count")]
    return (events.join(attributes, ["transaction_id", "year"], "left")
            .groupBy("lad_code", "month").agg(*counts))


def prior_income(spark):
    return (spark.read.option("header", True)
            .csv(str(ROOT / "data/processed/reference/gdhi_lad_annual.csv"))
            .select("lad_code", (F.col("year").cast("int") + 1).alias("feature_year"),
                    F.col("population").cast("double").alias("population_prev"),
                    F.col("gdhi_per_head_gbp").cast("double").alias("gdhi_per_head_prev")))


def reference_csv(spark, name, numeric):
    frame = spark.read.option("header", True).csv(str(ROOT / "data/processed/reference" / name))
    for column in numeric:
        frame = frame.withColumn(column, F.col(column).cast("double"))
    return frame


def trailing_mean(column, window, periods):
    return F.when(F.count(column).over(window) >= periods, F.avg(column).over(window))


def trailing_share(part, total, window):
    return F.when(F.count(total).over(window) == 12,
                  F.sum(part).over(window) / F.sum(total).over(window))


def impute_street_crime(df, by_lad):
    """Fill short street-crime gaps from t-12, scaled by the regional change since t-12."""
    measures = {"crime": "crime_count_observed",
                **{f"crime_{g}": f"crime_{g}_count" for g in CRIME_GROUPS}}
    covered = F.col("crime_coverage_complete")
    for name, source in measures.items():
        df = (df.withColumn(f"{name}_obs", F.when(covered, F.coalesce(F.col(source), F.lit(0)).cast("double")))
              .withColumn(f"{name}_obs_lag12", F.lag(f"{name}_obs", 12).over(by_lad)))
    runs = Window.partitionBy("lad_code", "crime_coverage_complete").orderBy("t_index")
    df = df.withColumn("gap_id", F.col("t_index") - F.row_number().over(runs))
    df = df.withColumn("gap_length", F.count("*").over(
        Window.partitionBy("lad_code", "crime_coverage_complete", "gap_id")))
    pairs = df.filter(F.col("crime_obs").isNotNull() & F.col("crime_obs_lag12").isNotNull())

    def ratios(keys, suffix):
        return pairs.groupBy(*keys).agg(*[(F.sum(f"{n}_obs") / F.sum(f"{n}_obs_lag12")).alias(f"{n}_{suffix}")
                                          for n in measures])

    df = (df.join(ratios(["region_code", "month"], "region_ratio"), ["region_code", "month"], "left")
          .join(ratios(["month"], "nation_ratio"), "month", "left"))
    can_impute = (~covered & (F.col("gap_length") <= MAX_IMPUTED_GAP) &
                  F.col("crime_obs_lag12").isNotNull())
    for name in measures:
        ratio = F.coalesce(f"{name}_region_ratio", f"{name}_nation_ratio")
        df = df.withColumn(f"{name}_clean", F.coalesce(
            F.col(f"{name}_obs"), F.when(can_impute, F.col(f"{name}_obs_lag12") * ratio)))
    return df.withColumn("crime_imputed", (can_impute & F.col("crime_clean").isNotNull()).cast("int"))


def build(spark):
    by_lad = Window.partitionBy("lad_code").orderBy("t_index")
    last3, last12 = by_lad.rowsBetween(-2, 0), by_lad.rowsBetween(-11, 0)
    geography = reference_csv(spark, "lad_geography.csv",
                              ["centroid_lat", "centroid_lon", "dist_to_london_km",
                               "urban_postcode_share", "imd_deprivation_score"]).drop("lad_name")
    df = (spark.read.parquet(str(LA / "rent_panel"))
          .join(crime_by_group(spark), ["lad_code", "month"], "left")
          .join(recorded_quarterly(spark), ["lad_code", "recorded_crime_quarter"], "left")
          .join(sales_mix(spark), ["lad_code", "month"], "left")
          .join(prior_income(spark), ["lad_code", "feature_year"], "left")
          .join(F.broadcast(geography), "lad_code", "left"))
    lad_type = F.create_map(*[F.lit(value) for pair in LAD_TYPES.items() for value in pair])
    angle = 2 * math.pi * (F.col("month_of_year") - 1) / 12
    holiday_end = F.when(F.col("country") == "Wales", "2021-06").otherwise("2021-09")
    df = (df.withColumn("year", F.substring("month", 1, 4).cast("int"))
          .withColumn("month_of_year", F.substring("month", 6, 2).cast("int"))
          .withColumn("t_index", (F.col("year") - 2015) * 12 + F.col("month_of_year") - 1)
          .withColumn("month_sin", F.sin(angle))
          .withColumn("month_cos", F.cos(angle))
          .withColumn("covid_restrictions", F.col("month").between("2020-03", "2021-03").cast("int"))
          .withColumn("transaction_tax_holiday",
                      ((F.col("month") >= "2020-07") & (F.col("month") <= holiday_end)).cast("int"))
          .withColumn("lad_type", lad_type[F.substring("lad_code", 1, 3)])
          .withColumn("is_london", (F.col("region_code") == "E12000007").cast("int"))
          .withColumn("log_dist_to_london_km", F.log1p("dist_to_london_km")))

    df = (df.withColumn("log_rent", F.log("rent_gbp_month"))
          .withColumn("rent_yoy_log_change", F.col("log_rent") - F.lag("log_rent", 12).over(by_lad))
          .withColumn("rent_relative_to_national", F.col("rent_gbp_month") /
                      F.avg("rent_gbp_month").over(Window.partitionBy("month")))
          .withColumn("rent_relative_to_region", F.col("rent_gbp_month") /
                      F.avg("rent_gbp_month").over(Window.partitionBy("region_code", "month")))
          .withColumn("rent_lag1", F.lag("rent_gbp_month", 1).over(by_lad))
          .withColumn("rent_lag3", F.lag("rent_gbp_month", 3).over(by_lad))
          .withColumn("rent_lag12", F.lag("rent_gbp_month", 12).over(by_lad))
          .withColumn("log_rent_lag1", F.lag("log_rent", 1).over(by_lad))
          .withColumn("rent_mom_log_change_lag1",
                      F.col("log_rent_lag1") - F.lag("log_rent", 2).over(by_lad))
          .withColumn("rent_yoy_log_change_lag1", F.lag("rent_yoy_log_change", 1).over(by_lad))
          .withColumn("rent_to_income_lag1", 12 * F.col("rent_lag1") / F.col("gdhi_per_head_gbp")))

    df = (df.withColumn("log_gdhi_per_head", F.log("gdhi_per_head_gbp"))
          .withColumn("log_population", F.log("population"))
          .withColumn("population_growth", F.log(F.col("population") / F.col("population_prev")))
          .withColumn("gdhi_per_head_growth",
                      F.log(F.col("gdhi_per_head_gbp") / F.col("gdhi_per_head_prev"))))

    df = (df.withColumn("log_sales_median", F.log("sales_median_gbp"))
          .withColumn("sales_median_roll3", trailing_mean("sales_median_gbp", last3, 3))
          .withColumn("sales_median_roll12", trailing_mean("sales_median_gbp", last12, 12))
          .withColumn("log_sales_median_roll12", F.log("sales_median_roll12"))
          .withColumn("log_sales_median_roll3", F.log("sales_median_roll3"))
          .withColumn("sales_median_yoy_log_change",
                      F.col("log_sales_median_roll3") - F.lag("log_sales_median_roll3", 12).over(by_lad))
          .withColumn("flat_sales_median_roll12", trailing_mean("flat_sales_median_gbp", last12, 6))
          .withColumn("sales_per_1000_residents_roll12",
                      F.when(F.count("sales_count").over(last12) == 12,
                             F.sum("sales_count").over(last12)) / F.col("population") * 1000)
          .withColumn("new_build_share_roll12",
                      trailing_share("new_build_mix_count", "mix_sales_count", last12))
          .withColumn("leasehold_share_roll12",
                      trailing_share("leasehold_mix_count", "mix_sales_count", last12))
          .withColumn("price_to_income_ratio", F.col("sales_median_roll12") / F.col("gdhi_per_head_gbp")))
    for name in PROPERTY_TYPES.values():
        df = df.withColumn(f"{name}_share_roll12",
                           trailing_share(f"{name}_mix_count", "mix_sales_count", last12))

    per_1000 = 1000 / F.col("population")
    df = (impute_street_crime(df, by_lad)
          .withColumn("crime_data_ok", F.col("crime_coverage_complete").cast("int"))
          .withColumn("crime_partial_upload", F.col("crime_partial_upload").cast("int"))
          .withColumn("crime_months_in_roll12", F.count("crime_clean").over(last12))
          .withColumn("crime_per_1000", F.col("crime_clean") * per_1000)
          .withColumn("crime_per_1000_roll12",
                      trailing_mean("crime_clean", last12, MIN_CRIME_MONTHS) * per_1000)
          .withColumn("log_crime_roll3", F.log(trailing_mean("crime_clean", last3, 3)))
          .withColumn("crime_yoy_log_change",
                      F.col("log_crime_roll3") - F.lag("log_crime_roll3", 12).over(by_lad)))
    for group in CRIME_GROUPS:
        df = df.withColumn(f"crime_{group}_per_1000_roll12",
                           trailing_mean(f"crime_{group}_clean", last12, MIN_CRIME_MONTHS) * per_1000)

    monthly_per_1000 = per_1000 / 12
    df = (df.withColumn("recorded_crime_allocated", F.col("recorded_crime_allocated").cast("int"))
          .withColumn("recorded_crime_per_1000_roll4q", F.col("recorded_crime_roll4q") * monthly_per_1000)
          .withColumn("recorded_crime_per_1000_roll4q_prevq",
                      F.col("recorded_crime_roll4q_prevq") * monthly_per_1000))
    for group in RECORDED_GROUPS:
        df = (df.withColumn(f"recorded_{group}_per_1000_roll4q",
                            F.col(f"recorded_{group}_roll4q") * monthly_per_1000)
              .withColumn(f"recorded_{group}_per_1000_roll4q_prevq",
                          F.col(f"recorded_{group}_roll4q_prevq") * monthly_per_1000))

    neighbours = spark.read.option("header", True).csv(
        str(ROOT / "data/processed/reference/lad_neighbours.csv")).select("lad_code", "neighbour_lad_code")
    values = df.select(F.col("lad_code").alias("neighbour_lad_code"), "month", "rent_lag1",
                       "rent_yoy_log_change_lag1", "log_sales_median_roll12", "crime_per_1000_roll12")
    spatial = (F.broadcast(neighbours).join(values, "neighbour_lad_code").groupBy("lad_code", "month")
               .agg(F.avg("rent_lag1").alias("neighbour_rent_lag1"),
                    F.avg("rent_yoy_log_change_lag1").alias("neighbour_rent_yoy_log_change_lag1"),
                    F.avg("log_sales_median_roll12").alias("neighbour_log_sales_median_roll12"),
                    F.avg("crime_per_1000_roll12").alias("neighbour_crime_per_1000_roll12")))
    df = (df.join(spatial, ["lad_code", "month"], "left")
          .withColumn("rent_lag1_vs_neighbours",
                      F.log(F.col("rent_lag1") / F.col("neighbour_rent_lag1"))))
    for name in LAG1_SOURCES:
        df = df.withColumn(f"{name}_lag1", F.lag(name, 1).over(by_lad))
    checks = df.select("lad_code", "month", "crime_count_observed", "sales_count", "mix_sales_count",
                       sum(F.col(f"crime_{g}_count") for g in CRIME_GROUPS).alias("crime_group_total"))
    return df.select(*[name for name, *_ in FEATURES]), checks


def write_outputs(spark, features):
    output = LA / "features"
    features.orderBy("lad_code", "month").coalesce(1).write.mode("overwrite").parquet(str(output))
    result = spark.read.parquet(str(output))
    csv_dir = LA / "features_csv_parts"
    result.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_dir))
    shutil.copyfile(next(csv_dir.glob("part-*.csv")), LA / "features.csv")
    with (ROOT / "data/metadata/feature_dictionary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "group", "role", "use_association", "use_forecast", "description"])
        for name, group, role, description in FEATURES:
            writer.writerow([name, group, role, *ROLE_USE[role], description])
    return result


def quality(result, checks):
    rows = result.count()
    duplicates = result.groupBy("lad_code", "month").count().filter("count != 1").count()
    mismatches = {
        "crime_groups_vs_observed": checks.filter(
            F.coalesce("crime_group_total", F.lit(0)) != F.col("crime_count_observed")).count(),
        "sales_mix_vs_sales_count": checks.filter(
            F.col("mix_sales_count") != F.col("sales_count")).count()}
    later = result.filter(F.col("split") != "history")
    missing_required = {c: later.filter(F.col(c).isNull()).count() for c in REQUIRED_OUTSIDE_HISTORY}
    missing_required.update({c: result.filter(F.col(c).isNull()).count() for c in REQUIRED_ALL_ROWS})
    crime_rows = {
        "observed": result.filter(F.col("crime_data_ok") == 1).count(),
        "imputed": result.filter(F.col("crime_imputed") == 1).count(),
        "partial_upload": result.filter(F.col("crime_partial_upload") == 1).count(),
        "still_missing": result.filter(F.col("crime_per_1000").isNull()).count(),
        "still_missing_by_force": {r["dominant_force"]: r["count"] for r in result.filter(
            F.col("crime_per_1000").isNull()).groupBy("dominant_force").count().collect()},
        "imputed_months_by_force": {r["dominant_force"]: r["months"] for r in result.filter(
            F.col("crime_imputed") == 1).groupBy("dominant_force").agg(
            F.countDistinct("month").alias("months")).collect()}}
    lag_check = (result.alias("a").join(result.alias("b"), F.col("a.lad_code") == F.col("b.lad_code"))
                 .filter(F.col("a.t_index") == F.col("b.t_index") + 12)
                 .filter(F.col("a.rent_lag12") != F.col("b.rent_gbp_month")).count())
    numeric = [name for name, dtype in result.dtypes if dtype in ("double", "int", "bigint")]
    nulls = result.groupBy("split").agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric])
    null_by_split = {r["split"]: {c: r[c] for c in numeric if r[c]} for r in nulls.collect()}
    metrics = {"rows": rows, "columns": len(result.columns), "duplicate_keys": duplicates,
               "source_mismatches": mismatches, "rent_lag12_mismatches": lag_check,
               "required_nulls": missing_required, "street_crime_rows": crime_rows,
               "null_counts_by_split": null_by_split}
    metrics["passed"] = (rows == 37920 and duplicates == 0 and lag_check == 0 and
                         not any(mismatches.values()) and not any(missing_required.values()))
    path = ROOT / "data/metadata/feature_quality.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    spark = (SparkSession.builder.appName("UKRentalFeatures")
             .master("local[4]").config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "16").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        features, checks = build(spark)
        result = write_outputs(spark, features)
        metrics = quality(result, checks)
        summary = {k: v for k, v in metrics.items() if k != "null_counts_by_split"}
        print(json.dumps(summary, indent=2), flush=True)
        if not metrics["passed"]:
            raise RuntimeError("Feature quality checks failed; see data/metadata/feature_quality.json")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
