#!/usr/bin/env python3
"""
Create a combined world map from the walking and crash mapping.csv files.

Walking points are drawn AFTER crash points, so walking points appear on top
where the two datasets overlap.

Default input paths:
    /Users/alam/Repos/walking-yt/mapping.csv
    /Users/alam/Repos/car-crash-yt/mapping.csv

Outputs:
    figures/global_mobility_datasets.html
    figures/global_mobility_datasets.png
    figures/global_mobility_datasets.svg

Requirements:
    plotly
    kaleido

Install:
    pip install plotly kaleido

Run:
    python3 plot_global_mobility_map.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

DEFAULT_WALKING_MAPPING = Path(
    "/Users/alam/Repos/walking-yt/mapping.csv"
)

DEFAULT_CRASH_MAPPING = Path(
    "/Users/alam/Repos/car-crash-yt/mapping.csv"
)


# ---------------------------------------------------------------------
# Figure styling
# ---------------------------------------------------------------------

WALKING_COLOUR = "#1473E6"
CRASH_COLOUR = "#FF5A00"

LAND_COLOUR = "#DDE3E8"
COUNTRY_LINE_COLOUR = "#FFFFFF"
OCEAN_COLOUR = "#FFFFFF"
TEXT_COLOUR = "#10223A"

WALKING_OPACITY = 0.88
CRASH_OPACITY = 0.70


@dataclass(frozen=True)
class LocalityPoint:
    dataset: str
    locality: str
    state: str
    country: str
    continent: str
    lat: float
    lon: float
    video_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot walking and crash mapping.csv locality coverage "
            "on one world map."
        )
    )

    parser.add_argument(
        "--walking-mapping",
        type=Path,
        default=DEFAULT_WALKING_MAPPING,
        help="Path to walking-yt/mapping.csv",
    )

    parser.add_argument(
        "--crash-mapping",
        type=Path,
        default=DEFAULT_CRASH_MAPPING,
        help="Path to car-crash-yt/mapping.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figure files",
    )

    parser.add_argument(
        "--filename",
        default="global_mobility_datasets",
        help="Output filename without extension",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1800,
        help="Static figure width in pixels",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1000,
        help="Static figure height in pixels",
    )

    parser.add_argument(
        "--flat-marker-size",
        action="store_true",
        help=(
            "Use the same marker size for every locality instead of "
            "scaling by number of associated videos."
        ),
    )

    parser.add_argument(
        "--walking-size",
        type=float,
        default=6.0,
        help=(
            "Base walking marker size when using flat marker sizes."
        ),
    )

    parser.add_argument(
        "--crash-size",
        type=float,
        default=5.0,
        help=(
            "Base crash marker size when using flat marker sizes."
        ),
    )

    return parser.parse_args()


def validate_mapping_path(
    path: Path,
    label: str,
) -> Path:
    path = path.expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"{label} mapping file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{label} mapping path is not a file: {path}"
        )

    return path


def parse_video_count(value: str | None) -> int:
    """
    Count video IDs from the repository bracket encoded list.

    Examples:
        [abc123] -> 1
        [abc123,def456] -> 2
        [] -> 0
    """
    text = str(value or "").strip()

    if not text:
        return 0

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if not text:
        return 0

    return sum(
        1
        for item in text.split(",")
        if item.strip().strip("'").strip('"')
    )


def parse_coordinate(
    value: str | None,
    minimum: float,
    maximum: float,
) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if not minimum <= number <= maximum:
        return None

    return number


def load_mapping(
    path: Path,
    dataset: str,
) -> list[LocalityPoint]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "locality",
            "country",
            "continent",
            "lat",
            "lon",
            "videos",
        }

        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(missing)
            )

        points: list[LocalityPoint] = []

        for row in reader:
            lat = parse_coordinate(
                row.get("lat"),
                -90.0,
                90.0,
            )

            lon = parse_coordinate(
                row.get("lon"),
                -180.0,
                180.0,
            )

            if lat is None or lon is None:
                continue

            video_count = parse_video_count(
                row.get("videos")
            )

            if video_count < 1:
                video_count = 1

            points.append(
                LocalityPoint(
                    dataset=dataset,
                    locality=str(
                        row.get("locality") or ""
                    ).strip(),
                    state=str(
                        row.get("state") or ""
                    ).strip(),
                    country=str(
                        row.get("country") or ""
                    ).strip(),
                    continent=str(
                        row.get("continent") or ""
                    ).strip(),
                    lat=lat,
                    lon=lon,
                    video_count=video_count,
                )
            )

    return points


def scaled_marker_sizes(
    points: Iterable[LocalityPoint],
    *,
    dataset: str,
) -> list[float]:
    """
    Use mild square root scaling so locations with more associated videos
    are visible without dominating the map.
    """
    sizes: list[float] = []

    for point in points:
        if dataset == "walking":
            base = 4.7
            scale = 1.8
            maximum = 16.0
        else:
            base = 4.0
            scale = 1.6
            maximum = 14.0

        size = (
            base
            + scale * math.sqrt(point.video_count)
        )

        sizes.append(
            min(size, maximum)
        )

    return sizes


def hover_text(
    point: LocalityPoint,
) -> str:
    location_parts = [
        point.locality,
        point.state,
        point.country,
    ]

    location = ", ".join(
        value
        for value in location_parts
        if value
    )

    return (
        f"<b>{location or 'Unknown locality'}</b>"
        f"<br>{point.dataset}"
        f"<br>Videos: {point.video_count}"
        f"<br>Continent: "
        f"{point.continent or 'Unknown'}"
        f"<br>Reference coordinate: "
        f"{point.lat:.5f}, {point.lon:.5f}"
    )


def make_trace(
    points: list[LocalityPoint],
    *,
    name: str,
    colour: str,
    opacity: float,
    sizes: list[float],
    legend_rank: int,
) -> go.Scattergeo:
    return go.Scattergeo(
        lon=[
            point.lon
            for point in points
        ],
        lat=[
            point.lat
            for point in points
        ],
        text=[
            hover_text(point)
            for point in points
        ],
        hoverinfo="text",
        name=name,
        mode="markers",
        legendrank=legend_rank,
        marker=dict(
            size=sizes,
            color=colour,
            opacity=opacity,
            line=dict(
                width=0.35,
                color="rgba(255,255,255,0.85)",
            ),
        ),
    )


def build_figure(
    walking_points: list[LocalityPoint],
    crash_points: list[LocalityPoint],
    *,
    width: int,
    height: int,
    flat_marker_size: bool,
    walking_size: float,
    crash_size: float,
) -> go.Figure:
    figure = go.Figure()

    if flat_marker_size:
        crash_sizes = [
            crash_size
            for _ in crash_points
        ]

        walking_sizes = [
            walking_size
            for _ in walking_points
        ]

    else:
        crash_sizes = scaled_marker_sizes(
            crash_points,
            dataset="crash",
        )

        walking_sizes = scaled_marker_sizes(
            walking_points,
            dataset="walking",
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    # Crash points are added FIRST.
    # Walking points are added SECOND.
    #
    # Plotly draws later traces on top of earlier traces, therefore
    # walking points remain visible when both datasets share a locality.
    # -------------------------------------------------------------

    figure.add_trace(
        make_trace(
            crash_points,
            name="Crash videos",
            colour=CRASH_COLOUR,
            opacity=CRASH_OPACITY,
            sizes=crash_sizes,
            legend_rank=2,
        )
    )

    figure.add_trace(
        make_trace(
            walking_points,
            name="Walking videos",
            colour=WALKING_COLOUR,
            opacity=WALKING_OPACITY,
            sizes=walking_sizes,
            legend_rank=1,
        )
    )

    figure.update_geos(
        projection_type="equirectangular",
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor=COUNTRY_LINE_COLOUR,
        countrywidth=0.6,
        showland=True,
        landcolor=LAND_COLOUR,
        showocean=True,
        oceancolor=OCEAN_COLOUR,
        bgcolor=OCEAN_COLOUR,
        lataxis_range=[
            -60,
            85,
        ],
        lonaxis_range=[
            -180,
            180,
        ],
    )

    figure.update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        title=dict(
            text="",
            x=0.025,
            y=0.965,
            xanchor="left",
            yanchor="top",
            font=dict(
                size=34,
                color=TEXT_COLOUR,
                family="Arial",
            ),
        ),
        legend=dict(
            x=0.985,
            y=0.975,
            xanchor="right",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(
                size=19,
                color=TEXT_COLOUR,
                family="Arial",
            ),
            itemsizing="constant",
            traceorder="normal",
        ),
        hoverlabel=dict(
            bgcolor="white",
            font=dict(
                size=14,
                family="Arial",
            ),
        ),
    )

    return figure


def print_summary(
    walking_points: list[LocalityPoint],
    crash_points: list[LocalityPoint],
    walking_path: Path,
    crash_path: Path,
) -> None:
    walking_video_refs = sum(
        point.video_count
        for point in walking_points
    )

    crash_video_refs = sum(
        point.video_count
        for point in crash_points
    )

    walking_countries = {
        point.country
        for point in walking_points
        if point.country
    }

    crash_countries = {
        point.country
        for point in crash_points
        if point.country
    }

    print()
    print(
        "=== Combined mobility map ==="
    )
    print(
        f"Walking mapping:       "
        f"{walking_path}"
    )
    print(
        f"Crash mapping:         "
        f"{crash_path}"
    )
    print()

    print(
        f"Walking locality rows: "
        f"{len(walking_points):,}"
    )
    print(
        f"Walking video refs:    "
        f"{walking_video_refs:,}"
    )
    print(
        f"Walking countries:     "
        f"{len(walking_countries):,}"
    )
    print()

    print(
        f"Crash locality rows:   "
        f"{len(crash_points):,}"
    )
    print(
        f"Crash video refs:      "
        f"{crash_video_refs:,}"
    )
    print(
        f"Crash countries:       "
        f"{len(crash_countries):,}"
    )
    print()

    print(
        "Drawing order:         "
        "Crash underneath, walking on top"
    )
    print()


def write_outputs(
    figure: go.Figure,
    output_dir: Path,
    filename: str,
    width: int,
    height: int,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        output_dir
        / f"{filename}.html"
    )

    png_path = (
        output_dir
        / f"{filename}.png"
    )

    svg_path = (
        output_dir
        / f"{filename}.svg"
    )

    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
    )

    print(
        f"Interactive HTML:      "
        f"{html_path.resolve()}"
    )

    try:
        figure.write_image(
            png_path,
            width=width,
            height=height,
            scale=2,
        )

        print(
            f"Publication PNG:       "
            f"{png_path.resolve()}"
        )

        figure.write_image(
            svg_path,
            width=width,
            height=height,
        )

        print(
            f"Vector SVG:            "
            f"{svg_path.resolve()}"
        )

    except Exception as exc:
        print()
        print(
            "Static PNG and SVG export "
            "could not be completed."
        )
        print(
            "The interactive HTML file "
            "was still created."
        )
        print()
        print(
            "Install or update Kaleido with:"
        )
        print(
            "    pip install -U kaleido"
        )
        print()
        print(
            f"Static export error: {exc}"
        )


def main() -> int:
    args = parse_args()

    walking_path = validate_mapping_path(
        args.walking_mapping,
        "Walking",
    )

    crash_path = validate_mapping_path(
        args.crash_mapping,
        "Crash",
    )

    walking_points = load_mapping(
        walking_path,
        "Walking videos",
    )

    crash_points = load_mapping(
        crash_path,
        "Crash videos",
    )

    if not walking_points:
        raise RuntimeError(
            "No valid walking locality coordinates "
            "were found in the walking mapping file."
        )

    if not crash_points:
        raise RuntimeError(
            "No valid crash locality coordinates "
            "were found in the crash mapping file."
        )

    print_summary(
        walking_points,
        crash_points,
        walking_path,
        crash_path,
    )

    figure = build_figure(
        walking_points,
        crash_points,
        width=args.width,
        height=args.height,
        flat_marker_size=args.flat_marker_size,
        walking_size=args.walking_size,
        crash_size=args.crash_size,
    )

    write_outputs(
        figure,
        args.output_dir,
        args.filename,
        args.width,
        args.height,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nCancelled."
        )
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
