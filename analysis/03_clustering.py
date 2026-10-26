"""Analysis 3 — Rental-market profiles of local authorities (KMeans clustering).

Answers the proposal's "regional rental-market profiles": which local
authorities look alike once rent level, rent growth, affordability (price/income),
crime and urbanity are considered together?

Steps
  1. Spark: collapse the monthly panel to one profile row per authority (316).
  2. MLlib: standardise the five profile features, run KMeans for k = 2..8 and
     score each k with the silhouette (how well each LAD sits in its cluster).
  3. Profile the chosen clusters and draw them on the map.

Writes
  reports/tables/cluster_k_selection.csv   silhouette and cost for every k
  reports/tables/cluster_profiles.csv      cluster means in original units
  reports/tables/lad_clusters.csv          cluster of every authority
  reports/figures/fig3_rent_clusters.png

Run from the project root:   uv run python analysis/03_clustering.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, functions as F

os.environ["PYSPARK_PYTHON"] = sys.executable

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/processed/local_authority/features.csv"
BOUNDARIES = (ROOT / "data/raw/Boundaries/"
              "Local_Authority_Districts_December_2024_Boundaries_UK_BUC_2466110883082851160.geojson")
TABLE_DIR, FIG_DIR = ROOT / "reports/tables", ROOT / "reports/figures"

BASE_YEAR, END_YEAR = 2016, 2024
K_RANGE = range(2, 9)
MIN_K = 3            # k = 2 is too coarse to describe anything
TOLERANCE = 0.01     # silhouettes closer than this count as a tie
SEED = 42            # KMeans starts from random centres; a fixed seed makes runs repeatable

PROFILE = {          # column -> label (all are period averages unless noted)
    "rent": "Mean rent (GBP/month)",
    "rent_growth": f"Rent growth {BASE_YEAR}-{END_YEAR} (%)",
    "price_to_income": "House price / income",
    "crime": "Recorded crime per 1000 / month",
    "urban": "Urban postcode share",
}
# KMeans measures straight-line distance, so right-skewed values are logged first
# (a 400 -> 800 rent jump should count like 1600 -> 3200; Westminster's crime
# rate, 4x the median, should not form a cluster on its own).
LOGGED = {"rent", "crime"}
# GDHI per head is left out on purpose: it correlates 0.81 with rent, so it would
# count "expensive area" a second time (price/income already carries income).


# ---------------------------------------------------------------------------
# Step 1. One profile row per local authority
# ---------------------------------------------------------------------------
def lad_profiles(df):
    period = df.filter(F.col("year").between(BASE_YEAR, END_YEAR))
    profiles = (period.groupBy("lad_code", "lad_name", "region_name")
                .agg(F.avg("rent_gbp_month").alias("rent"),
                     F.avg("price_to_income_ratio").alias("price_to_income"),
                     F.avg("recorded_crime_per_1000_roll4q").alias("crime"),
                     F.first("urban_postcode_share").alias("urban"),   # static
                     F.avg("population").alias("population"),
                     F.avg(F.when(F.col("year") == BASE_YEAR, F.col("rent_gbp_month"))).alias("r0"),
                     F.avg(F.when(F.col("year") == END_YEAR, F.col("rent_gbp_month"))).alias("r1")))
    profiles = (profiles.withColumn("rent_growth", (F.col("r1") / F.col("r0") - 1) * 100)
                        .drop("r0", "r1"))
    for c in LOGGED:
        profiles = profiles.withColumn(f"{c}_in", F.log(c))
    return profiles


def model_inputs():
    return [f"{c}_in" if c in LOGGED else c for c in PROFILE]


# ---------------------------------------------------------------------------
# Step 2. Standardise + KMeans, choose k by silhouette
# ---------------------------------------------------------------------------
def build_pipeline(k):
    return Pipeline(stages=[
        VectorAssembler(inputCols=model_inputs(), outputCol="raw_features"),
        # Without scaling, urban share (0..1) and rent growth (~30) would dominate the
        # distance just because of their units. Scaled: each feature has sd = 1.
        StandardScaler(inputCol="raw_features", outputCol="features",
                       withMean=True, withStd=True),
        KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=SEED),
    ])


def choose_k(profiles):
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                    metricName="silhouette")
    scores = []
    for k in K_RANGE:
        model = build_pipeline(k).fit(profiles)
        silhouette = evaluator.evaluate(model.transform(profiles))
        cost = model.stages[-1].summary.trainingCost    # within-cluster sum of squares
        scores.append({"k": k, "silhouette": round(silhouette, 4), "wcss": round(cost, 1)})
    print("\nchoosing k  (silhouette: -1 bad .. +1 perfect;  WCSS: lower = tighter)")
    for s in scores:
        print(f"  k={s['k']}  silhouette={s['silhouette']:.3f}  WCSS={s['wcss']:8.1f}")
    eligible = [s for s in scores if s["k"] >= MIN_K]
    top = max(s["silhouette"] for s in eligible)
    # Silhouettes within TOLERANCE are a tie; among ties take the larger k, which
    # describes the map in more detail at no real cost in cluster quality.
    best = max((s for s in eligible if s["silhouette"] >= top - TOLERANCE), key=lambda s: s["k"])
    print(f"-> k = {best['k']} (largest k >= {MIN_K} within {TOLERANCE} "
          f"of the best silhouette {top:.3f})")
    return best["k"], scores


# ---------------------------------------------------------------------------
# Step 3. Describe and map the clusters
# ---------------------------------------------------------------------------
def describe(clustered):
    """Cluster means in original units, ordered by mean rent (0 = cheapest)."""
    stats = (clustered.groupBy("cluster")
             .agg(F.count("*").alias("n_lads"), F.sum("population").alias("population"),
                  *[F.avg(c).alias(c) for c in PROFILE])
             .orderBy("rent").collect())
    order = {row.cluster: i for i, row in enumerate(stats)}   # relabel 0..k-1 by rent
    print("\ncluster profiles (means; ordered from cheapest to most expensive):")
    header = f"{'':<34}" + "".join(f"{'C' + str(order[r.cluster]):>11}" for r in stats)
    print(header)
    print(f"{'Local authorities':<34}" + "".join(f"{r.n_lads:>11}" for r in stats))
    for c, label in PROFILE.items():
        fmt = "{:>11,.0f}" if c == "rent" else "{:>11.2f}"
        print(f"{label:<34}" + "".join(fmt.format(r[c]) for r in stats))

    # name a few members of each cluster (largest by population) and its regions
    members = clustered.select("lad_name", "region_name", "population", "cluster").collect()
    for r in stats:
        rows = sorted((m for m in members if m.cluster == r.cluster), key=lambda m: -m.population)
        regions = {}
        for m in rows:
            regions[m.region_name] = regions.get(m.region_name, 0) + 1
        top_regions = ", ".join(f"{name} {n}" for name, n in
                                sorted(regions.items(), key=lambda x: -x[1])[:3])
        print(f"\nC{order[r.cluster]}: e.g. {', '.join(m.lad_name for m in rows[:6])}")
        print(f"     regions: {top_regions}")
    return stats, order


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#c3c2b7"


def plot(lad_cluster, k, stats, order, out):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    shapes = gpd.read_file(BOUNDARIES)
    shapes["lad_code"] = shapes["LAD24CD"].replace({"E08000016": "E08000038",
                                                     "E08000019": "E08000039"})
    shapes = shapes[shapes["lad_code"].str[0].isin(["E", "W"])].copy()
    shapes["cluster"] = shapes["lad_code"].map(lad_cluster)

    cmap = ListedColormap(COLORS[:k])
    style = dict(column="cluster", cmap=cmap, vmin=-0.5, vmax=k - 0.5,
                 edgecolor="white", linewidth=0.2,
                 missing_kwds={"color": "#f0efec", "edgecolor": "white", "linewidth": 0.2})
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    shapes.plot(ax=ax, **style)
    ax.set_axis_off()
    ax.set_title("Rental-market profiles of local authorities (KMeans)",
                 loc="left", color=INK, fontsize=9)

    london = shapes[shapes["lad_code"].str.startswith("E09")]
    inset = ax.inset_axes([-0.02, 0.64, 0.34, 0.34])
    london.plot(ax=inset, **style)
    inset.set_xticks([]); inset.set_yticks([]); inset.set_xlabel(""); inset.set_ylabel("")
    for side in inset.spines.values():
        side.set_edgecolor(GREY)
    inset.set_title("London", fontsize=7, color=MUTED, pad=2)
    xmin, ymin, xmax, ymax = london.total_bounds
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor=MUTED, linewidth=0.6))

    # Legend carries each cluster's headline numbers, so it reads without the table.
    by_label = {order[r.cluster]: r for r in stats}
    handles = [Patch(facecolor=COLORS[c], edgecolor="white",
                     label=(f"C{c}  \u00a3{by_label[c].rent:,.0f}/mo, "
                            f"+{by_label[c].rent_growth:.0f}%, {by_label[c].n_lads} LADs"))
               for c in range(k)]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.16, 1.03),
              frameon=False, fontsize=6.5, alignment="left",
              title=f"mean rent, growth {BASE_YEAR}\u2013{END_YEAR}", title_fontsize=6.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"\nwrote {out.relative_to(ROOT)}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    spark = (SparkSession.builder.appName("UKRentalAnalysis3")
             .master("local[4]").config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark.read.csv(str(FEATURES), header=True, inferSchema=True)
        profiles = lad_profiles(df).cache()
        print(f"\nprofiles: {profiles.count()} local authorities, "
              f"missing values: {profiles.count() - profiles.dropna().count()}")
        profiles.select(*PROFILE).summary("mean", "stddev", "min", "max").show()

        print("correlation between profile features (highly correlated = double counted):")
        cols = list(PROFILE)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = profiles.stat.corr(a, b)
                if abs(r) >= 0.6:
                    print(f"  {a} ~ {b}: r = {r:+.2f}")

        k, scores = choose_k(profiles)
        clustered = build_pipeline(k).fit(profiles).transform(profiles).cache()
        stats, order = describe(clustered)

        lad_rows = [{"lad_code": r.lad_code, "lad_name": r.lad_name,
                     "region_name": r.region_name, "cluster": order[r.cluster]}
                    for r in clustered.select("lad_code", "lad_name", "region_name",
                                              "cluster").collect()]
        write_csv(TABLE_DIR / "cluster_k_selection.csv", scores)
        write_csv(TABLE_DIR / "cluster_profiles.csv",
                  [{"cluster": f"C{order[r.cluster]}", "n_lads": r.n_lads,
                    **{c: round(r[c], 3) for c in PROFILE}} for r in stats])
        write_csv(TABLE_DIR / "lad_clusters.csv", lad_rows)
        plot({r["lad_code"]: r["cluster"] for r in lad_rows}, k, stats, order,
             FIG_DIR / "fig3_rent_clusters.png")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
