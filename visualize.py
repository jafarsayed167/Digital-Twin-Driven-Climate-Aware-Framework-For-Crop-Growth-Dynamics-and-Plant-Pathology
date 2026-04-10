import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os

# -------- LOAD DATA --------
df = pd.read_csv("data/climate_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -------- 3-HOUR RESAMPLE --------
df_3h = df.set_index("Date").resample("3h").agg({
    "Temperature":    "mean",
    "Humidity":       "mean",
    "Rainfall":       "sum",
    "WindSpeed":      "mean",
    "SolarRadiation": "mean"
}).dropna(how="all").reset_index()

for col in ["Temperature","Humidity","Rainfall","WindSpeed","SolarRadiation"]:
    if col in df_3h.columns:
        df_3h[col] = df_3h[col].round(2)

# -------- IDEAL RANGES --------
RANGES = {
    "Temperature":    (20,   35,   False),
    "Humidity":       (60,   90,   False),
    "Rainfall":       (None, 20,   True),
    "WindSpeed":      (None, 5,    True),
    "SolarRadiation": (200,  1000, False),
}

COLORS = {
    "Temperature":    "#f97316",
    "Humidity":       "#22d3ee",
    "Rainfall":       "#6366f1",
    "WindSpeed":      "#a78bfa",
    "SolarRadiation": "#fbbf24",
}

UNITS = {
    "Temperature":    "°C",
    "Humidity":       "%",
    "Rainfall":       "mm",
    "WindSpeed":      "m/s",
    "SolarRadiation": "W/m²",
}

TITLES = {
    "Temperature":    "Temperature",
    "Humidity":       "Humidity",
    "Rainfall":       "Rainfall",
    "WindSpeed":      "Wind Speed",
    "SolarRadiation": "Solar Radiation",
}

os.makedirs("data/plots", exist_ok=True)

# -------- GLOBAL STYLE --------
BG      = "#080d14"
PANEL   = "#0f1923"
GRID    = "#1a2535"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
SAFE    = "#4ade80"
DANGER  = "#f87171"
WARN    = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   SUBTEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
})

# -------- HELPER: DOT COLORS --------
def dot_colors(series, mn, mx, only_max):
    out = []
    for v in series:
        if only_max:
            out.append(DANGER if v > mx else SAFE)
        else:
            if mn is not None and (v < mn or v > mx):
                out.append(DANGER)
            else:
                out.append(SAFE)
    return out

# ================================================================
# INDIVIDUAL PLOTS (5 files)
# ================================================================
def plot_single(col):
    mn, mx, only_max = RANGES[col]
    color  = COLORS[col]
    unit   = UNITS[col]
    title  = TITLES[col]
    x      = df_3h["Date"]
    y      = df_3h[col]
    d_cols = dot_colors(y, mn, mx, only_max)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Fill area under line
    if col != "Rainfall":
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.plot(x, y, color=color, linewidth=2.2, zorder=4)
        ax.scatter(x, y, c=d_cols, s=45, zorder=5, edgecolors="none")
    else:
        # Bar for rainfall
        bar_colors = [DANGER if v > mx else color for v in y]
        ax.bar(x, y, color=bar_colors, width=0.1, alpha=0.85, zorder=4)

    # Safe zone band
    if mn is not None and mx is not None:
        ax.axhspan(mn, mx, alpha=0.05, color=SAFE, zorder=1)
        ax.axhline(mn, color=WARN,   linewidth=1,   linestyle="--", alpha=0.6, zorder=3)
        ax.axhline(mx, color=DANGER, linewidth=1,   linestyle="--", alpha=0.6, zorder=3)
    elif mx is not None:
        ax.axhline(mx, color=DANGER, linewidth=1.2, linestyle="--", alpha=0.6, zorder=3,
                   label=f"Max safe ({mx})")

    # Value annotations on every point
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=6.5,
            color=TEXT,
            alpha=0.7
        )

    # X axis — 3h ticks, date+time
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
    plt.xticks(rotation=0, ha="center", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", length=0, pad=6)

    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(GRID)

    # Legend patches
    legend_items = []
    if mn:
        legend_items.append(mpatches.Patch(color=WARN,   alpha=0.8, label=f"Min safe: {mn}"))
    if mx:
        legend_items.append(mpatches.Patch(color=DANGER, alpha=0.8, label=f"Max safe: {mx}"))
    legend_items.append(mpatches.Patch(color=SAFE,   alpha=0.8, label="In range"))
    legend_items.append(mpatches.Patch(color=DANGER, alpha=0.8, label="Out of range"))
    ax.legend(handles=legend_items, loc="upper right",
              fontsize=8, framealpha=0.15, edgecolor=GRID,
              labelcolor=TEXT, facecolor=PANEL)

    # Title + label
    ax.set_title(f"{title}  ·  3-Hour Intervals  ·  Guntur",
                 fontsize=13, fontweight="bold", color=TEXT,
                 pad=14, loc="left")
    ax.set_ylabel(unit, fontsize=9, labelpad=8)
    ax.set_xlabel("Date & Time (IST)", fontsize=9, labelpad=8)

    plt.tight_layout()
    out_path = f"data/plots/{col}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close()
    print(f"✅ {title} saved → {out_path}")

for c in ["Temperature","Humidity","Rainfall","WindSpeed","SolarRadiation"]:
    if c in df_3h.columns:
        plot_single(c)

# ================================================================
# COMBINED OVERVIEW — 5 subplots in 1 image
# ================================================================
fig = plt.figure(figsize=(16, 22), facecolor=BG)
fig.suptitle(
    "📊  Complete Climate Overview  ·  Guntur, Andhra Pradesh  ·  3-Hour Intervals",
    fontsize=14, fontweight="bold", color=TEXT,
    y=0.995
)

gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.45)

for idx, col in enumerate(["Temperature","Humidity","Rainfall","WindSpeed","SolarRadiation"]):
    if col not in df_3h.columns:
        continue

    mn, mx, only_max = RANGES[col]
    color  = COLORS[col]
    unit   = UNITS[col]
    title  = TITLES[col]
    x      = df_3h["Date"]
    y      = df_3h[col]
    d_cols = dot_colors(y, mn, mx, only_max)

    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    if col == "Rainfall":
        bar_colors = [DANGER if v > mx else color for v in y]
        ax.bar(x, y, color=bar_colors, width=0.1, alpha=0.85)
    else:
        ax.fill_between(x, y, alpha=0.1, color=color)
        ax.plot(x, y, color=color, linewidth=1.8, zorder=4)
        ax.scatter(x, y, c=d_cols, s=25, zorder=5, edgecolors="none")

    if mn:
        ax.axhline(mn, color=WARN,   linestyle="--", linewidth=0.9, alpha=0.5)
    if mx:
        ax.axhline(mx, color=DANGER, linestyle="--", linewidth=0.9, alpha=0.5)
    if mn and mx:
        ax.axhspan(mn, mx, alpha=0.04, color=SAFE)

    ax.grid(axis="y", zorder=0, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=SUBTEXT, labelsize=7.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))

    # Title inside plot
    ax.set_title(f"{title} ({unit})", fontsize=10, fontweight="bold",
                 color=TEXT, loc="left", pad=6)

    # X ticks only on last subplot
    if idx < 4:
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
        plt.setp(ax.get_xticklabels(), ha="center", fontsize=7.5)
        ax.set_xlabel("Date & Time (IST)", fontsize=9, color=SUBTEXT, labelpad=6)

plt.savefig("data/plots/climate_overview.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
plt.close()
print("✅ Combined Overview saved → data/plots/climate_overview.png")