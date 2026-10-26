"""Analysis 1 — How rents vary across space and time (exploratory).

Answers the proposal's first question: where and when did private rents change,
and by how much? Reads the feature table built by build_features.py and writes

* reports/figures/fig1_rent_space_time.png   (the figure for the report)
  (needs the ONS LAD Dec-2024 BUC boundary GeoJSON in data/raw/Boundaries/)
* reports/tables/lad_rent_growth.csv          (one row per local authority)
* reports/tables/region_rent_monthly.csv      (one row per region x month)

Run from the project root:   uv run python analysis/01_spatiotemporal.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

# Same fix as the pipeline scripts: Spark's Python workers use this interpreter.
os.environ["PYSPARK_PYTHON"] = sys.executable

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/processed/local_authority/features.csv"
FIG_DIR = ROOT / "reports/figures"
TABLE_DIR = ROOT / "reports/tables"

BOUNDARIES = (ROOT / "data/raw/Boundaries/"
              "Local_Authority_Districts_December_2024_Boundaries_UK_BUC_2466110883082851160.geojson")

BASE_YEAR, END_YEAR = 2016, 2024   # 2015 is the "history" warm-up year; skip it

# The rent workbook uses the 2024 recodes for Barnsley and Sheffield; the
# December 2024 boundary file still carries the old codes (same areas, same
# mapping as prepare_local_authority.py).
BOUNDARY_RECODE = {"E08000016": "E08000038",   # Barnsley
                   "E08000019": "E08000039"}   # Sheffield


def load_boundaries():
    """England + Wales local-authority polygons, keyed by the panel's lad_code.

    The file is in British National Grid (EPSG:27700, metres), so the map is
    drawn without distortion and needs no reprojection.
    """
    import geopandas as gpd
    shapes = gpd.read_file(BOUNDARIES)
    shapes["lad_code"] = shapes["LAD24CD"].replace(BOUNDARY_RECODE)
    return shapes[shapes["lad_code"].str[0].isin(["E", "W"])].copy()


# ---------------------------------------------------------------------------
# Step 0. Get to know the table
# ---------------------------------------------------------------------------
def explore(df):
    """Print the things worth knowing before any analysis."""
    print(f"\nrows = {df.count():,}   columns = {len(df.columns)}")
    print("\nrows per split (history = 2015 warm-up, mostly null rolling features):")
    df.groupBy("split").count().orderBy("split").show()

    print("one row = one local authority (lad) x one month:")
    df.select("lad_code", "lad_name", "region_name", "month", "split",
              "rent_gbp_month", "sales_median_gbp", "recorded_crime_per_1000_roll4q",
              "gdhi_per_head_gbp").orderBy("lad_code", "month").show(5, truncate=False)

    print("rent_gbp_month summary (GBP per month, nominal):")
    df.select("rent_gbp_month").summary("min", "25%", "50%", "75%", "max").show()

    print("local authorities per region:")
    (df.select("region_name", "lad_code").distinct()
       .groupBy("region_name").count().orderBy(F.desc("count")).show(truncate=False))


# ---------------------------------------------------------------------------
# Step 1. Cumulative rent growth per local authority, 2016 -> 2024
# ---------------------------------------------------------------------------
def lad_growth(df):
    """Annual mean rent in BASE_YEAR and END_YEAR, then the % change.

    Annual means instead of a single month smooth out month-to-month noise.
    """
    annual = (df.filter(F.col("year").isin(BASE_YEAR, END_YEAR))
                .groupBy("lad_code", "lad_name", "region_name", "year")
                .agg(F.avg("rent_gbp_month").alias("rent"),
                     F.first("centroid_lat").alias("lat"),
                     F.first("centroid_lon").alias("lon")))
    # Long -> wide: one row per LAD with a column per year.
    wide = (annual.groupBy("lad_code", "lad_name", "region_name")
                  .pivot("year", [BASE_YEAR, END_YEAR]).agg(F.first("rent"))
                  .withColumnRenamed(str(BASE_YEAR), "rent_base")
                  .withColumnRenamed(str(END_YEAR), "rent_end"))
    coords = annual.groupBy("lad_code").agg(F.first("lat").alias("lat"),
                                            F.first("lon").alias("lon"))
    return (wide.join(coords, "lad_code")
                .withColumn("growth_pct", (F.col("rent_end") / F.col("rent_base") - 1) * 100)
                .orderBy(F.desc("growth_pct")))


# ---------------------------------------------------------------------------
# Step 2. Monthly rent per region, indexed to BASE_YEAR = 100
# ---------------------------------------------------------------------------
def region_series(df):
    """Population-weighted mean rent per region and month.

    A plain mean would let a small rural authority count as much as Birmingham.
    Weighting by (prior-year) population gives the rent the average resident faces.
    """
    monthly = (df.filter(F.col("year") >= BASE_YEAR)
                 .groupBy("region_name", "month")
                 .agg((F.sum(F.col("rent_gbp_month") * F.col("population"))
                       / F.sum("population")).alias("rent")))
    base = (monthly.filter(F.col("month").startswith(f"{BASE_YEAR}-"))
                   .groupBy("region_name").agg(F.avg("rent").alias("rent_base")))
    return (monthly.join(base, "region_name")
                   .withColumn("index", F.col("rent") / F.col("rent_base") * 100)
                   .orderBy("region_name", "month"))


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows([[row[c] for c in columns] for row in rows])


# ---------------------------------------------------------------------------
# Step 3. Figure (results are tiny now, so plotting happens on the driver)
# ---------------------------------------------------------------------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GREY, INK, MUTED = "#c3c2b7", "#0b0b0b", "#52514e"
HIGHLIGHT = {"London": "#2a78d6"}          # second highlight is picked from the data
SECOND = "#eb6834"


def plot(lads, regions, out: Path):
    import matplotlib
    matplotlib.use("Agg")                   # no window; just write the file
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GREY, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9})
    fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                        gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) choropleth: each local authority coloured by its rent growth
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    shapes = load_boundaries()
    growth = {r.lad_code: r.growth_pct for r in lads}
    shapes["growth_pct"] = shapes["lad_code"].map(growth)
    unmatched = sorted(set(growth) - set(shapes["lad_code"]))
    print(f"boundaries: {shapes['growth_pct'].notna().sum()} of {len(growth)} panel "
          f"authorities matched; unmatched: {unmatched or 'none'}")
    no_data = shapes[shapes["growth_pct"].isna()]   # City of London, Isles of Scilly
    print(f"no rent data (drawn grey): {', '.join(no_data['LAD24NM'])}")

    vmin, vmax = min(growth.values()), max(growth.values())
    style = dict(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.2)
    shapes.plot(column="growth_pct", ax=ax_map, **style,
                missing_kwds={"color": "#f0efec", "edgecolor": "white", "linewidth": 0.2})
    ax_map.set_axis_off()
    ax_map.set_title(f"(a) Rent growth {BASE_YEAR}→{END_YEAR} by local authority",
                     loc="left", color=INK)

    # London's 32 boroughs are a speck at national scale, so zoom in on them.
    london = shapes[shapes["lad_code"].str.startswith("E09")]
    inset = ax_map.inset_axes([-0.02, 0.62, 0.36, 0.36])   # top-left, over the Irish Sea
    london.plot(column="growth_pct", ax=inset, **style,
                missing_kwds={"color": "#f0efec", "edgecolor": "white", "linewidth": 0.2})
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_xlabel(""); inset.set_ylabel("")   # geopandas adds CRS axis labels
    for side in inset.spines.values():
        side.set_edgecolor(GREY)
    inset.set_title("London", fontsize=7, color=MUTED, pad=2)
    xmin, ymin, xmax, ymax = london.total_bounds
    # Outline London on the main map (no connector lines: they would cross England).
    ax_map.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor=MUTED, linewidth=0.6))

    scale = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    bar = fig.colorbar(scale, ax=ax_map, shrink=0.7, pad=0.02)
    bar.set_label("% change in mean monthly rent")
    bar.outline.set_visible(False)

    # (b) regions indexed to BASE_YEAR = 100; grey context, two highlighted lines
    by_region = {}
    for r in regions:
        by_region.setdefault(r.region_name, []).append(r)
    final = {name: rows[-1]["index"] for name, rows in by_region.items()}
    fastest = max((n for n in final if n not in HIGHLIGHT), key=final.get)
    colors = {**HIGHLIGHT, fastest: SECOND}

    months = sorted({r.month for r in regions})
    x_of = {m: i for i, m in enumerate(months)}
    for name, rows in by_region.items():
        if name in colors:
            continue
        ax_ts.plot([x_of[r.month] for r in rows], [r["index"] for r in rows],
                   color=GREY, linewidth=1)
    for name, color in colors.items():
        rows = by_region[name]
        xs, ys = [x_of[r.month] for r in rows], [r["index"] for r in rows]
        ax_ts.plot(xs, ys, color=color, linewidth=2)
        ax_ts.annotate(f"{name} ({ys[-1] - 100:+.0f}%)", (xs[-1], ys[-1]),
                       xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=7, color=INK)
    ax_ts.annotate("other regions", (x_of[months[-1]], sorted(final.values())[len(final) // 2]),
                   xytext=(4, 0), textcoords="offset points", va="center",
                   fontsize=7, color=MUTED)
    years = [m for m in months if m.endswith("-01") and int(m[:4]) % 2 == 0]
    ax_ts.set_xticks([x_of[m] for m in years], [m[:4] for m in years])
    ax_ts.set_xlim(0, len(months) + 26)     # room for the end labels
    ax_ts.axhline(100, color=GREY, linewidth=0.8, linestyle=":")
    ax_ts.grid(axis="y", color="#f0efec", linewidth=0.8)
    ax_ts.spines[["top", "right"]].set_visible(False)
    ax_ts.set_ylabel(f"Rent index ({BASE_YEAR} mean = 100)")
    ax_ts.set_title("(b) Population-weighted rent by region", loc="left", color=INK)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"\nwrote {out.relative_to(ROOT)}")


def main():
    spark = (SparkSession.builder.appName("UKRentalAnalysis1")
             .master("local[4]").config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        # inferSchema: let Spark guess types (fine for a 64 MB file; for big data
        # you would pass an explicit schema to avoid an extra pass over the file).
        df = spark.read.csv(str(FEATURES), header=True, inferSchema=True)
        # inferSchema turns "2015-01" into a timestamp (2015-01-01 00:00:00);
        # put it back to the panel's "YYYY-MM" key so it sorts and labels cleanly.
        df = df.withColumn("month", F.date_format("month", "yyyy-MM")).cache()
        explore(df)

        lads = lad_growth(df).collect()          # 316 rows -> safe to collect
        regions = region_series(df).collect()    # 10 regions x 108 months

        print(f"\nfastest-growing local authorities {BASE_YEAR}->{END_YEAR}:")
        for r in lads[:5]:
            print(f"  {r.lad_name:<28} {r.region_name:<26} {r.growth_pct:6.1f}%")
        print("slowest:")
        for r in lads[-5:]:
            print(f"  {r.lad_name:<28} {r.region_name:<26} {r.growth_pct:6.1f}%")

        write_csv(TABLE_DIR / "lad_rent_growth.csv", lads,
                  ["lad_code", "lad_name", "region_name", "lat", "lon",
                   "rent_base", "rent_end", "growth_pct"])
        write_csv(TABLE_DIR / "region_rent_monthly.csv", regions,
                  ["region_name", "month", "rent", "index"])
        plot(lads, regions, FIG_DIR / "fig1_rent_space_time.png")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
