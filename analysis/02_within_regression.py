"""Analysis 2 — Do rents move with house prices, crime and income *within* an area?

A pooled regression across all local authorities mostly rediscovers "London is
expensive": places with high rents also have high prices, high incomes and more
recorded crime. Here every variable is a year-on-year change, and the model is
run three ways so the reader can see what controlling for area and year does:

  (1) pooled         no fixed effects
  (2) LAD FE         each authority's average growth removed
  (3) LAD + year FE  also removes national shocks shared by every area
                     (COVID, inflation, interest-rate cycle)  <- main result

Model (annual panel, 316 authorities x 2016-2024 = 2,844 rows):

  dlog_rent_it = b1 dlog_sales_it + b2 dlog_crime_it
               + b3 gdhi_growth_i,Y-1 + b4 pop_growth_i,Y-1 + FE + e_it

All growth terms are log differences, so b1 and b2 read as elasticities: a 1%
faster rise in local house prices goes with a b1% faster rise in rent.

Why annual and not monthly? rent_yoy_log_change in consecutive months shares
11 of its 12 months, so 34k monthly rows would hold roughly 3k independent
changes and every standard error would look ~3x too precise.

Writes reports/tables/within_regression.csv and prints the table.
Run from the project root:   uv run python analysis/02_within_regression.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, Window, functions as F

os.environ["PYSPARK_PYTHON"] = sys.executable

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/processed/local_authority/features.csv"
OUT = ROOT / "reports/tables/within_regression.csv"

FIRST_YEAR, LAST_YEAR = 2016, 2024   # years of *change*; 2015 serves as the base
TARGET = "dlog_rent"
X = ["dlog_sales", "dlog_crime", "gdhi_growth", "pop_growth"]
LABELS = {"dlog_sales": "House prices (dlog)", "dlog_crime": "Recorded crime (dlog)",
          "gdhi_growth": "GDHI per head growth (Y-1)",
          "pop_growth": "Population growth (Y-1)"}


# ---------------------------------------------------------------------------
# Step 1. Monthly panel -> one row per local authority x year, in changes
# ---------------------------------------------------------------------------
def annual_changes(df):
    annual = (df.groupBy("lad_code", "lad_name", "region_name", "year")
                .agg(F.avg("rent_gbp_month").alias("rent"),
                     F.avg("sales_median_gbp").alias("sales"),
                     # December's 4-quarter window = the whole calendar year
                     F.max(F.when(F.col("month").endswith("-12"),
                                  F.col("recorded_crime_per_1000_roll4q"))).alias("crime"),
                     # annual inputs are already lagged to Y-1 in the panel
                     F.first("gdhi_per_head_growth").alias("gdhi_growth"),
                     F.first("population_growth").alias("pop_growth")))

    prev = Window.partitionBy("lad_code").orderBy("year")
    for level, change in [("rent", "dlog_rent"), ("sales", "dlog_sales"), ("crime", "dlog_crime")]:
        annual = annual.withColumn(change, F.log(level) - F.log(F.lag(level).over(prev)))
    return (annual.filter(F.col("year").between(FIRST_YEAR, LAST_YEAR))
                  .select("lad_code", "lad_name", "region_name", "year", TARGET, *X))


# ---------------------------------------------------------------------------
# Step 2. Fixed effects by demeaning (the "within" transformation)
# ---------------------------------------------------------------------------
def demean(df, cols, by_lad: bool, by_year: bool):
    """Subtract group means so only within-group variation is left.

    Regressing demeaned y on demeaned x gives exactly the same slopes as adding
    one dummy per LAD (and per year) — without 325 extra columns. Two-way
    demeaning (x - LAD mean - year mean + grand mean) is exact here because the
    panel is balanced: every LAD has every year.
    """
    out = df
    lad_w, year_w = Window.partitionBy("lad_code"), Window.partitionBy("year")
    all_w = Window.partitionBy()  # one window over everything (fine for 2.8k rows)
    for c in cols:
        expr = F.col(c)
        if by_lad:
            expr = expr - F.avg(c).over(lad_w)
        if by_year:
            expr = expr - F.avg(c).over(year_w)
        if by_lad and by_year:
            expr = expr + F.avg(c).over(all_w)
        out = out.withColumn(c, expr)
    return out


# ---------------------------------------------------------------------------
# Step 3. Fit with MLlib, then make the standard errors honest
# ---------------------------------------------------------------------------
def fit(df, n_fe: int):
    data = VectorAssembler(inputCols=X, outputCol="features").transform(df)
    # solver="normal" solves the normal equations exactly (OLS) and is what
    # unlocks coefficientStandardErrors / pValues in the training summary.
    model = LinearRegression(featuresCol="features", labelCol=TARGET, solver="normal",
                             regParam=0.0, fitIntercept=(n_fe == 0)).fit(data)
    summary = model.summary
    n, k = summary.numInstances, len(X) + (n_fe == 0)
    beta = list(model.coefficients)
    mllib_se = list(summary.coefficientStandardErrors)[:len(X)]

    # MLlib assumes independent rows. Two corrections:
    # (a) its degrees of freedom ignore the fixed effects we absorbed by demeaning;
    # (b) residuals of the same LAD are correlated over years -> cluster by LAD.
    dof_fix = math.sqrt((n - k) / (n - k - n_fe)) if n_fe else 1.0
    cluster_se = clustered_se(model.transform(data), beta, n, k + n_fe)
    return {"beta": beta, "se_mllib": [s * dof_fix for s in mllib_se],
            "se_cluster": cluster_se, "n": n, "r2": summary.r2}


def clustered_se(pred, beta, n, k):
    """Cluster-robust (by LAD) standard errors: (X'X)^-1 (sum_g u_g u_g') (X'X)^-1.

    Spark does the heavy lifting (sums over rows and per-LAD score vectors);
    only K x K matrices (K = 4) come back to the driver.
    """
    import numpy as np
    resid = F.col(TARGET) - F.col("prediction")
    xtx = pred.select(*[F.sum(F.col(a) * F.col(b)).alias(f"{a}*{b}")
                        for a in X for b in X]).first()
    bread = np.linalg.inv(np.array([[xtx[f"{a}*{b}"] for b in X] for a in X]))
    scores = (pred.groupBy("lad_code")
                  .agg(*[F.sum(F.col(a) * resid).alias(a) for a in X]).collect())
    meat = sum(np.outer(u, u) for u in (np.array([row[a] for a in X]) for row in scores))
    g = len(scores)
    small_sample = g / (g - 1) * (n - 1) / (n - k)        # usual Stata-style factor
    return list(np.sqrt(np.diag(bread @ meat @ bread) * small_sample))


def stars(beta, se):
    t = abs(beta / se)
    return "***" if t > 3.29 else "**" if t > 2.58 else "*" if t > 1.96 else ""


def main():
    spark = (SparkSession.builder.appName("UKRentalAnalysis2")
             .master("local[4]").config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark.read.csv(str(FEATURES), header=True, inferSchema=True)
        df = df.withColumn("month", F.date_format("month", "yyyy-MM"))
        panel = annual_changes(df).cache()

        rows = panel.count()
        complete = panel.dropna().count()
        n_lad = panel.select("lad_code").distinct().count()
        n_year = panel.select("year").distinct().count()
        print(f"\nannual panel: {rows:,} rows = {n_lad} LADs x {n_year} years; "
              f"complete cases: {complete:,}")
        print("\nwhat the changes look like (log points; 0.05 ~ +5%):")
        panel.select(TARGET, *X).summary("mean", "stddev", "min", "50%", "max").show()

        print("raw correlations with rent growth (before any controls):")
        for c in X:
            print(f"  {LABELS[c]:<28} r = {panel.stat.corr(TARGET, c):+.2f}")

        panel = panel.dropna()
        specs = [("(1) pooled", panel, 0),
                 ("(2) LAD FE", demean(panel, [TARGET, *X], True, False), n_lad - 1),
                 ("(3) LAD + year FE", demean(panel, [TARGET, *X], True, True),
                  n_lad + n_year - 2)]
        results = {name: fit(data, n_fe) for name, data, n_fe in specs}

        # ---- print and save the table (coefficient, clustered SE in brackets)
        names = list(results)
        print("\n" + "Within-LAD regression of annual rent growth".center(94))
        print(f"{'':<28}" + "".join(f"{n:>22}" for n in names))
        table = []
        for i, c in enumerate(X):
            line, row = f"{LABELS[c]:<28}", {"term": LABELS[c]}
            for n in names:
                b, s = results[n]["beta"][i], results[n]["se_cluster"][i]
                line += f"{b:+.3f}{stars(b, s):<3} ({s:.3f})".rjust(22)
                row[f"{n} coef"], row[f"{n} se_cluster"] = round(b, 4), round(s, 4)
                row[f"{n} se_mllib"] = round(results[n]["se_mllib"][i], 4)
            print(line)
            table.append(row)
        print(f"{'R^2 (within for FE models)':<28}"
              + "".join(f"{results[n]['r2']:>22.3f}" for n in names))
        print(f"{'N':<28}" + "".join(f"{results[n]['n']:>22,}" for n in names))
        print("clustered-by-LAD SE in brackets; * p<.05  ** p<.01  *** p<.001")

        print("\nhow much clustering matters (model 3, SE: MLlib -> clustered):")
        for i, c in enumerate(X):
            m, s = results[names[2]]["se_mllib"][i], results[names[2]]["se_cluster"][i]
            print(f"  {LABELS[c]:<28} {m:.4f} -> {s:.4f}  (x{s / m:.1f})")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
            writer.writerow({"term": "R2", **{f"{n} coef": round(results[n]["r2"], 4)
                                              for n in names}})
            writer.writerow({"term": "N", **{f"{n} coef": results[n]["n"] for n in names}})
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
