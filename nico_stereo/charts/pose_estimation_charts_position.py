from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nico_stereo.config import ROOT


OBJECT_TRANSLATIONS_SK = {
    "apple": "jablko",
    "chips_box": "krabica čipsov",
    "lemon": "citrón",
    "orange": "pomaranč",
    "rubiks_cube": "Rubikova kocka",
    "scissors": "nožnice",
    "wood_block": "drevený kváder",
}

OBJECT_ORDER_SK = [
    "jablko",
    "citrón",
    "pomaranč",
    "Rubikova kocka",
    "krabica čipsov",
    "drevený kváder",
    "nožnice",
]

SCENE_TRANSLATIONS_SK = {
    "scene_001": "scéna 1",
    "scene_002": "scéna 2",
    "scene_003": "scéna 3",
    "scene_004": "scéna 4",
    "scene_005": "scéna 5",
    "scene_006": "scéna 6",
    "scene_007": "scéna 7",
    "scene_008": "scéna 8",
    "scene_009": "scéna 9",
    "scene_010": "scéna 10",
}

SCENE_ORDER_SK = [
    "scéna 1",
    "scéna 2",
    "scéna 3",
    "scéna 4",
    "scéna 5",
    "scéna 6",
    "scéna 7",
    "scéna 8",
    "scéna 9",
    "scéna 10",
]

FONT_SIZE = 22
TITLE_FONT_SIZE = 28
AXIS_TITLE_FONT_SIZE = 24
TICK_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 20


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chýba vstupný súbor: {path}")
    return pd.read_csv(path)


def _add_slovak_object_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["object_sk"] = df["object"].map(OBJECT_TRANSLATIONS_SK).fillna(df["object"])
    df["object_sk"] = pd.Categorical(
        df["object_sk"],
        categories=OBJECT_ORDER_SK,
        ordered=True,
    )
    return df


def _add_slovak_scene_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scene_sk"] = df["scene"].map(SCENE_TRANSLATIONS_SK).fillna(df["scene"])
    df["scene_sk"] = pd.Categorical(
        df["scene_sk"],
        categories=SCENE_ORDER_SK,
        ordered=True,
    )
    return df


def build_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    colorbar_title: str,
    scale_factor: float = 1.0,
    text_format: str = ".2f",
) -> None:
    summary_df = _add_slovak_object_names(summary_df)
    summary_df = _add_slovak_scene_names(summary_df)

    if metric not in summary_df.columns:
        raise ValueError(f"Stĺpec '{metric}' neexistuje v summary dataframe.")

    heat = summary_df.pivot(
        index="object_sk",
        columns="scene_sk",
        values=metric,
    )

    heat = heat.dropna(how="all")

    fig = px.imshow(
        heat * scale_factor,
        labels={
            "color": colorbar_title,
            "x": "Scéna",
            "y": "Objekt",
        },
        text_auto=text_format,
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=TITLE_FONT_SIZE),
        ),
        template="plotly_white",
        width=1100,
        height=700,
        font=dict(size=FONT_SIZE),
        coloraxis_colorbar=dict(
            title=dict(
                text=colorbar_title,
                font=dict(size=AXIS_TITLE_FONT_SIZE),
            ),
            tickfont=dict(size=TICK_FONT_SIZE),
        ),
    )

    fig.update_xaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_yaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_traces(
        textfont=dict(size=ANNOTATION_FONT_SIZE),
    )

    fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vizualizácie pre compare_6D_pose výstupy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/out_24042026/pose_estimation/results_comparison"),
    )

    args = parser.parse_args()

    parent_dir = ROOT
    camera = "zed"

    root = parent_dir / args.root / camera
    summary_df = _load_csv(root / "summary_per_scene_object.csv")

    # Heatmapa chyby translácie
    build_heatmap(
        summary_df=summary_df,
        metric="err_translation_mean_m",
        title="Heatmapa priemernej chyby translácie (objekt × scéna)",
        colorbar_title="Chyba translácie [cm]",
        scale_factor=100.0,
        text_format=".1f",
    )

    # Heatmapa chyby rotácie
    build_heatmap(
        summary_df=summary_df,
        metric="err_rotation_mean_deg",
        title="Heatmapa priemernej chyby rotácie (objekt × scéna)",
        colorbar_title="Chyba rotácie [°]",
        scale_factor=1.0,
        text_format=".2f",
    )


if __name__ == "__main__":
    main()