"""Combined three-panel figure for the 2-page report.

Reads only the small tables written by analyses 01 and 03 (no Spark needed):
  reports/tables/lad_rent_growth.csv, region_rent_monthly.csv, lad_clusters.csv,
  cluster_profiles.csv
and writes reports/figures/fig_report.png, sized for the full text width.

Run after 01 and 03:   uv run python analysis/04_report_figure.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
OUT = ROOT / "reports/figures/fig_report.png"
BOUNDARIES = (ROOT / "data/raw/Boundaries/"
              "Local_Authority_Districts_December_2024_Boundaries_UK_BUC_2466110883082851160.geojson")
RECODE = {"E08000016": "E08000038", "E08000019": "E08000039"}   # Barnsley, Sheffield

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GREY, NODATA = "#0b0b0b", "#52514e", "#c3c2b7", "#f0efec"


def read(name):
    with (TABLES / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def map_panel(ax, shapes, column, title, **style):
    shapes.plot(column=column, ax=ax, edgecolor="white", linewidth=0.1,
                missing_kwds={"color": NODATA, "edgecolor": "white", "linewidth": 0.1}, **style)
    london = shapes[shapes["lad_code"].str.startswith("E09")]
    inset = ax.inset_axes([-0.04, 0.60, 0.40, 0.40])
    london.plot(column=column, ax=inset, edgecolor="white", linewidth=0.15,
                missing_kwds={"color": NODATA}, **style)
    inset.set_xticks([]); inset.set_yticks([]); inset.set_xlabel(""); inset.set_ylabel("")
    for side in inset.spines.values():
        side.set_edgecolor(GREY); side.set_linewidth(0.5)
    inset.set_title("London", fontsize=5.5, color=MUTED, pad=1)
    x0, y0, x1, y1 = london.total_bounds
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=MUTED, linewidth=0.5))
    ax.set_axis_off()
    ax.set_title(title, loc="left", color=INK)


def main():
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.edgecolor": GREY,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED})
    shapes = gpd.read_file(BOUNDARIES)
    shapes["lad_code"] = shapes["LAD24CD"].replace(RECODE)
    shapes = shapes[shapes["lad_code"].str[0].isin(["E", "W"])].copy()

    growth = {r["lad_code"]: float(r["growth_pct"]) for r in read("lad_rent_growth.csv")}
    cluster = {r["lad_code"]: int(r["cluster"]) for r in read("lad_clusters.csv")}
    profiles = {int(r["cluster"][1:]): r for r in read("cluster_profiles.csv")}
    shapes["growth"] = shapes["lad_code"].map(growth)
    shapes["cluster"] = shapes["lad_code"].map(cluster)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(6.5, 2.2),
                                           gridspec_kw={"width_ratios": [1, 1.15, 1]})

    # (a) rent growth choropleth
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    lo, hi = min(growth.values()), max(growth.values())
    map_panel(ax_a, shapes, "growth", "(a) Rent growth 2016–2024",
              cmap=cmap, vmin=lo, vmax=hi)
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)),
                       ax=ax_a, orientation="horizontal", fraction=0.05, pad=0.02, aspect=25)
    bar.set_label("% change in mean monthly rent", fontsize=6)
    bar.outline.set_visible(False)

    # (b) regional index lines
    series = {}
    for r in read("region_rent_monthly.csv"):
        series.setdefault(r["region_name"], []).append((r["month"], float(r["index"])))
    months = sorted({m for rows in series.values() for m, _ in rows})
    x_of = {m: i for i, m in enumerate(months)}
    final = {name: rows[-1][1] for name, rows in series.items()}
    fastest = max((n for n in final if n != "London"), key=final.get)
    colors = {"London": "#2a78d6", fastest: "#eb6834"}
    for name, rows in series.items():
        if name not in colors:
            ax_b.plot([x_of[m] for m, _ in rows], [v for _, v in rows], color=GREY, lw=0.8)
    for name, color in colors.items():
        rows = series[name]
        ax_b.plot([x_of[m] for m, _ in rows], [v for _, v in rows], color=color, lw=1.6)
        ax_b.annotate(f"{name}\n{rows[-1][1] - 100:+.0f}%", (x_of[rows[-1][0]], rows[-1][1]),
                      xytext=(3, 0), textcoords="offset points", va="center",
                      fontsize=6, color=INK)
    ticks = [m for m in months if m.endswith("-01") and int(m[:4]) % 2 == 0]
    ax_b.set_xticks([x_of[m] for m in ticks], [m[:4] for m in ticks])
    ax_b.set_xlim(0, len(months) + 30)
    ax_b.axhline(100, color=GREY, lw=0.6, ls=":")
    ax_b.grid(axis="y", color=NODATA, lw=0.6)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.set_ylabel("Rent index (2016 = 100)")
    ax_b.set_title("(b) Population-weighted rent by region", loc="left", color=INK)
    ax_b.text(0.02, 0.97, "grey: other regions", transform=ax_b.transAxes,
              va="top", fontsize=5.5, color=MUTED)

    # (c) KMeans clusters
    k = len(profiles)
    map_panel(ax_c, shapes, "cluster", f"(c) KMeans market profiles (k={k})",
              cmap=ListedColormap(CLUSTER_COLORS[:k]), vmin=-0.5, vmax=k - 0.5)
    handles = [Patch(facecolor=CLUSTER_COLORS[c], edgecolor="white",
                     label=(f"C{c} £{float(profiles[c]['rent']):,.0f}, "
                            f"+{float(profiles[c]['rent_growth']):.0f}%"))
               for c in range(k)]
    ax_c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                ncol=2, frameon=False, fontsize=5.8, handlelength=1, columnspacing=0.8)

    fig.tight_layout(w_pad=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
