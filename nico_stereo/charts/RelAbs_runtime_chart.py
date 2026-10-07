from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go

from nico_stereo.config import ROOT
from nico_stereo.prepare_paths import (
    get_depth_estimation_network,
    prepare_depth_comparison_paths,
)

# -----------------------------------------------------------------------------
# Konfigurácia
# -----------------------------------------------------------------------------
parent_dir = ROOT
date = "24042026"
rgbd_suffix = "zed"
metric_col = "all_AbsRel"

(_, _, _, _, _, depth_comparison_dir) = prepare_depth_comparison_paths(
    parent_dir, date, rgbd_suffix
)
metrics_out_dir = depth_comparison_dir / "metrics_cauchy"


def display_name(nn: str) -> str:
    return nn.split("_")[0] if "_" in nn else nn


# -----------------------------------------------------------------------------
# Zber agregovaných štatistík
# -----------------------------------------------------------------------------
nn_names = get_depth_estimation_network()
records = []

for nn_key, nn_value in nn_names.items():
    csv_path = metrics_out_dir / f"{nn_key}_per_image.csv"
    if not csv_path.exists():
        print(f"Preskakujem {nn_key}: chýba {csv_path}")
        continue

    df_metrics = pd.read_csv(csv_path)

    if metric_col not in df_metrics.columns:
        print(f"Preskakujem {nn_key}: stĺpec '{metric_col}' nebol nájdený")
        continue

    stats_path = (
        parent_dir
        / "out"
        / f"out_{date}"
        / "depth_estimation"
        / nn_key
        / "run_stats.json"
    )

    if not stats_path.exists():
        print(f"Preskakujem {nn_key}: chýba {stats_path}")
        continue

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    per_image_stats = stats.get("per_image_stats", [])
    if not per_image_stats:
        print(f"Preskakujem {nn_key}: neboli nájdené štatistiky pre jednotlivé snímky")
        continue

    # -------------------------------------------------------------------------
    # Kontrola kompatibility dĺžok
    # -------------------------------------------------------------------------
    if len(df_metrics) != len(per_image_stats):
        print(
            f"Upozornenie pre {nn_key}: počet riadkov metrík ({len(df_metrics)}) "
            f"sa nerovná počtu časových záznamov ({len(per_image_stats)}). "
            f"Použije sa kratšia dĺžka."
        )

    n = min(len(df_metrics), len(per_image_stats))
    df_metrics = df_metrics.iloc[:n].copy()
    per_image_stats = per_image_stats[:n]

    # -------------------------------------------------------------------------
    # Spojenie podľa poradia snímok
    # -------------------------------------------------------------------------
    df_metrics["runtime_s"] = [e["time_s"] for e in per_image_stats]

    label = display_name(nn_value)

    records.append(
        {
            "label": label,
            "mean_runtime": df_metrics["runtime_s"].mean(),
            "std_runtime": df_metrics["runtime_s"].std(),
            "mean_absrel": df_metrics[metric_col].mean(),
            "std_absrel": df_metrics[metric_col].std(),
            "n": n,
        }
    )

    print(
        f"{label}: "
        f"čas = {records[-1]['mean_runtime']:.4f} ± {records[-1]['std_runtime']:.4f} s, "
        f"AbsRel = {records[-1]['mean_absrel']:.4f} ± {records[-1]['std_absrel']:.4f}"
    )

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("Neboli nájdené žiadne dáta na vykreslenie grafu.")

# Ak je len 1 vzorka, std môže byť NaN -> nahradíme nulou
df[["std_runtime", "std_absrel"]] = df[["std_runtime", "std_absrel"]].fillna(0)

# -----------------------------------------------------------------------------
# Farby pre jednotlivé metódy
# -----------------------------------------------------------------------------
colors = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# -----------------------------------------------------------------------------
# Vytvorenie grafu — jeden obdĺžnik na sieť + viditeľný marker v strede
# -----------------------------------------------------------------------------
fig = go.Figure()

for i, (_, row) in enumerate(df.iterrows()):
    color = colors[i % len(colors)]

    x_center = row["mean_runtime"]
    y_center = row["mean_absrel"]

    x0 = x_center - row["std_runtime"]
    x1 = x_center + row["std_runtime"]
    y0 = y_center - row["std_absrel"]
    y1 = y_center + row["std_absrel"]

    # Aby obdĺžnik nezasahoval do záporných hodnôt času
    x0 = max(0, x0)

    customdata = [[
        row["label"],
        row["mean_runtime"],
        row["std_runtime"],
        row["mean_absrel"],
        row["std_absrel"],
        row["n"],
    ]] * 5

    # -------------------------------------------------------------------------
    # Obdĺžnik reprezentujúci rozsah mean ± std
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode="lines",
            fill="toself",
            fillcolor=color,
            opacity=0.65,
            name=row["label"],
            line=dict(width=1.5, color=color),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Priemerný čas: %{customdata[1]:.4f} ± %{customdata[2]:.4f} s"
                "<br>Priemerný AbsRel: %{customdata[3]:.4f} ± %{customdata[4]:.4f}"
                "<br>Počet snímok: %{customdata[5]}"
                "<extra></extra>"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Viditeľný marker presne v strede obdĺžnika
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[x_center],
            y=[y_center],
            mode="markers",
            marker=dict(
                symbol="cross",
                size=9,
                color=color,
                opacity=1.0,
                line=dict(width=2, color="black"),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['label']}</b>"
                "<br>Stred obdĺžnika"
                f"<br>Priemerný čas: {x_center:.4f} s"
                f"<br>Priemerný AbsRel: {y_center:.4f}"
                "<extra></extra>"
            ),
        )
    )

# -----------------------------------------------------------------------------
# Nastavenie vzhľadu grafu
# -----------------------------------------------------------------------------
fig.update_layout(
    title=dict(
        text="Priemerný čas inferencie vs. priemerná relatívna absolútna chyba",
        font=dict(size=23),
    ),
    xaxis_title="Priemerný čas inferencie [s]",
    yaxis_title="Priemerný AbsRel",
    template="plotly_white",
    width=1100,
    height=650,
    font=dict(size=19),
    showlegend=True,
    legend=dict(
        font=dict(size=18),
    ),
)

fig.update_xaxes(
    tickformat=".2f",
    title_font=dict(size=18),
    tickfont=dict(size=17),
    rangemode="tozero",
)

fig.update_yaxes(
    tickformat=".2f",
    title_font=dict(size=18),
    tickfont=dict(size=17),
)

fig.show()