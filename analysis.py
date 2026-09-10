"""Generate paper-facing Results values and plots for the car-crash corpus.

The current car-crash pipeline stores the authoritative segment records in
``state.json`` and a geographically resolved, locality-aggregated public view in
``mapping.csv``. This script uses both sources with their correct denominators.

Examples
--------
Run with the repository-root files::

    python analysis.py

Use explicit paths::

    python analysis.py --state state.json --mapping mapping.csv

Record a longitudinal snapshot only when it represents a meaningful collection
or processing checkpoint::

    python analysis.py --record-snapshot --snapshot-label taxonomy_round_1
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.analytics.crash_results import (
    CRASH_TAXONOMY_VERSION,
    CrashAnalysisResult,
    CrashResultsAnalysis,
)
from utils.plotting.crash_results import CrashResultsPlotter


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Optional synthetic MMUCC manner completion for draft visualisation only
# ---------------------------------------------------------------------
#
# IMPORTANT:
# False = publication-safe behaviour. Only measured/currently classified
#         MMUCC manner values are used in fig_world_manner_of_collision.
#
# True  = keep the actual figure AND additionally create a clearly labelled
#         synthetic projection:
#         fig_world_manner_of_collision_synthetic
#
# The synthetic figure never overwrites the actual figure and should not be
# reported as an empirical result. It is useful only for layout/prototyping
# while the taxonomy backfill is still running.
ENABLE_SYNTHETIC_MANNER_COMPLETION = True

# Deterministic seed. Keeping this fixed makes the synthetic projection stable
# across runs for the same segment identifiers.
SYNTHETIC_MANNER_SEED = "chi_draft_manner_v1"

# Draft-only assumed distribution for pending collision segments.
# These are synthetic assumptions, not measured corpus statistics.
# Canonical names match car_crash_pipeline.crash_taxonomy.MANNER_OF_COLLISION_VALUES.
SYNTHETIC_MANNER_WEIGHTS = {
    "front_to_rear_or_rear_to_front": 0.3472,
    "angle": 0.2241,
    "sideswipe_same_direction": 0.1386,
    "front_to_front": 0.0908,
    "sideswipe_opposite_direction": 0.0617,
    "other": 0.1376,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse the current car-crash state and mapping outputs for the CHI Results section."
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "state.json",
        help="Path to authoritative state.json. Default: ./state.json",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "mapping.csv",
        help="Path to mapping.csv. Default: ./mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_output" / "chi_crash",
        help="Directory for tables, JSON, HTML, and figure exports.",
    )
    parser.add_argument(
        "--publication-dir",
        type=Path,
        default=ROOT / "figures" / "chi_crash",
        help="Directory receiving publication figure copies.",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=15,
        help="Number of countries shown in the top-country figure.",
    )
    parser.add_argument(
        "--map-top-road-users",
        type=int,
        default=4,
        help=(
            "Number of most frequent road-user categories shown in the "
            "world-map small multiples. Default: 4."
        ),
    )
    parser.add_argument(
        "--map-top-manners",
        type=int,
        default=4,
        help=(
            "Number of most frequent classified collision manners shown in "
            "the world-map small multiples. Default: 4."
        ),
    )
    parser.add_argument(
        "--map-top-first-harmful",
        type=int,
        default=4,
        help=(
            "Number of most frequent first-harmful-event categories shown "
            "in the world-map small multiples. Default: 4."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute tables and summary only.",
    )
    parser.add_argument(
        "--record-snapshot",
        action="store_true",
        help="Append one explicit longitudinal snapshot to snapshots.csv.",
    )
    parser.add_argument(
        "--snapshot-label",
        default="",
        help="Optional human-readable label for --record-snapshot.",
    )
    return parser


# ---------------------------------------------------------------------
# Paper-facing road-user normalisation
# ---------------------------------------------------------------------

def _normalise_road_user_value(value: object) -> str:
    """Return the controlled paper-facing road-user category.

    The visual model's ``road_users`` field is intentionally open vocabulary.
    For analysis we merge obvious spelling variants and, as requested for the
    paper figure, combine vans and pickup trucks into one category.

    Raw ``road_users`` values are retained in the segment analysis output.
    """
    text = str(value or "").strip().casefold()
    if not text:
        return ""

    key = (
        text.replace("-", " ")
        .replace("_", " ")
        .replace(".", "")
    )
    key = " ".join(key.split())

    van_pickup_values = {
        "van",
        "vans",
        "pickup",
        "pick up",
        "pickup truck",
        "pickup trucks",
        "pick up truck",
        "pick up trucks",
        "pickup van",
        "pickup vans",
        "pick up van",
        "pick up vans",
    }
    if key in van_pickup_values:
        return "van_pickup"

    # Keep the remaining semantic categories distinct, while collapsing only
    # straightforward singular/plural or spelling variants.
    aliases = {
        "car": "car",
        "cars": "car",
        "passenger car": "car",
        "passenger cars": "car",
        "police car": "police_car",
        "police cars": "police_car",
        "truck": "truck",
        "trucks": "truck",
        "lorry": "truck",
        "lorries": "truck",
        "light truck": "light_truck",
        "light trucks": "light_truck",
        "suv": "suv",
        "suvs": "suv",
        "motorcycle": "motorcycle",
        "motorcycles": "motorcycle",
        "motorbike": "motorcycle",
        "motorbikes": "motorcycle",
        "bus": "bus",
        "buses": "bus",
        "pedestrian": "pedestrian",
        "pedestrians": "pedestrian",
        "cyclist": "cyclist",
        "cyclists": "cyclist",
        "bicyclist": "cyclist",
        "bicyclists": "cyclist",
    }
    if key in aliases:
        return aliases[key]

    return key.replace(" ", "_")


def _road_user_label(value: str) -> str:
    """Human-readable label for a normalised road-user category."""
    if value == "van_pickup":
        return "Van / pickup truck"
    if value == "light_truck":
        return "Light truck"
    if value == "police_car":
        return "Police car"
    return str(value).replace("_", " ").title()


def _normalise_road_user_list(values: object) -> list[str]:
    """Normalise and de-duplicate one segment's multi-label road-user list."""
    if not isinstance(values, list):
        return []

    normalised = {
        category
        for category in (_normalise_road_user_value(value) for value in values)
        if category
    }
    return sorted(normalised)


def _normalised_road_user_table(segments: pd.DataFrame) -> pd.DataFrame:
    """Build segment prevalence counts after paper-facing normalisation.

    Counting is performed after normalising each segment and de-duplicating its
    labels, so a segment containing both a raw ``van`` and ``pickup truck``
    label contributes only once to the merged category.
    """
    total = len(segments)
    counter: dict[str, int] = {}

    if "road_users" not in segments.columns:
        return pd.DataFrame(
            columns=[
                "road_user",
                "label",
                "segments",
                "segment_prevalence_pct",
            ]
        )

    for values in segments["road_users"]:
        for category in _normalise_road_user_list(values):
            counter[category] = counter.get(category, 0) + 1

    rows = [
        {
            "road_user": category,
            "label": _road_user_label(category),
            "segments": count,
            "segment_prevalence_pct": round(count / total * 100, 2)
            if total
            else 0.0,
        }
        for category, count in counter.items()
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "road_user",
                "label",
                "segments",
                "segment_prevalence_pct",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["segments", "road_user"],
            ascending=[False, True],
            ignore_index=True,
        )
    )


def _apply_road_user_normalisation(result: CrashAnalysisResult) -> None:
    """Attach normalised road users and replace the paper-facing count table.

    The original ``road_users`` column is left untouched. A separate
    ``road_users_normalised`` column is added to both full and resolved segment
    frames, and the ``road_users`` Results table is rebuilt from those
    normalised categories.
    """
    result.tables["road_users"] = _normalised_road_user_table(result.segments)

    for frame in (result.segments, result.resolved_segments):
        if "road_users" in frame.columns:
            frame["road_users_normalised"] = frame["road_users"].map(
                _normalise_road_user_list
            )


def _remove_stale_svg_outputs(*directories: Path) -> None:
    """Remove SVGs left by older analysis runs from the dedicated figure dirs."""
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob("fig_*.svg"):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Warning: could not remove stale SVG {path}: {exc}")


class GeographicCrashResultsPlotter(CrashResultsPlotter):
    """Add geographic small-multiple figures while keeping the existing saver.

    The parent :class:`CrashResultsPlotter` remains responsible for the normal
    paper figures and, crucially, for the established figure export behaviour:

    * HTML, PNG, and PDF are written to ``_output/chi_crash/figures``.
    * The same successfully produced files are copied to
      ``figures/chi_crash`` for publication use.
    * SVG export is deliberately disabled.

    The new map figures use the same naming, sizing, Kaleido handling, and
    publication-copy locations as every existing crash figure.
    """

    def _save(
        self,
        fig: go.Figure,
        name: str,
        *,
        width: int = 1100,
        height: int = 650,
    ) -> None:
        """Save HTML, PNG, and PDF using the existing crash figure directories.

        This mirrors :class:`CrashResultsPlotter`'s established saving system
        but intentionally omits SVG. Any stale SVG with the same figure name
        from an earlier run is removed.
        """
        self._base_layout(fig, width=width, height=height)

        html_path = self.output_dir / f"{name}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")

        produced: list[Path] = [html_path]

        stale_svg = self.output_dir / f"{name}.svg"
        stale_svg.unlink(missing_ok=True)

        for suffix in ("png", "pdf"):
            path = self.output_dir / f"{name}.{suffix}"
            try:
                fig.write_image(
                    path,
                    width=width,
                    height=height,
                    scale=2 if suffix == "png" else 1,
                )
                produced.append(path)
            except Exception as exc:
                print(f"Warning: could not write {path.name}: {exc}")

        if self.publication_dir:
            publication_svg = self.publication_dir / f"{name}.svg"
            publication_svg.unlink(missing_ok=True)

            for path in produced:
                shutil.copy2(path, self.publication_dir / path.name)

    @staticmethod
    def _human_label(value: object) -> str:
        text = str(value or "").strip()
        if not text or text.casefold() in {"unknown", "none", "nan"}:
            return "Unknown"
        if text == "dusk/dawn":
            return "Dusk / dawn"
        return text.replace("_", " ").replace("/", " / ").title()

    @staticmethod
    def _valid_resolved_segments(result: CrashAnalysisResult) -> pd.DataFrame:
        """Return only resolved segments with usable canonical coordinates."""
        data = result.resolved_segments.copy()
        if data.empty:
            return data

        data["lat"] = pd.to_numeric(data["lat"], errors="coerce")
        data["lon"] = pd.to_numeric(data["lon"], errors="coerce")
        data = data.dropna(subset=["lat", "lon"])
        data = data.loc[
            data["lat"].between(-90, 90)
            & data["lon"].between(-180, 180)
        ].copy()
        return data

    @staticmethod
    def _geo_panel_layout(
        fig: go.Figure,
        *,
        rows: int,
        cols: int,
    ) -> None:
        """Apply the same neutral geographic styling to every geo subplot."""
        for index in range(1, rows * cols + 1):
            suffix = "" if index == 1 else str(index)
            geo_name = f"geo{suffix}"
            if geo_name not in fig.layout:
                continue
            fig.layout[geo_name].update(
                projection_type="natural earth",
                showframe=False,
                showland=True,
                landcolor="rgb(238,238,238)",
                showcountries=True,
                countrycolor="white",
                showcoastlines=True,
                coastlinecolor="white",
                showocean=True,
                oceancolor="white",
            )

    @staticmethod
    def _bubble_sizes(
        counts: pd.Series,
        *,
        global_max: float,
        minimum: float = 4.0,
        maximum: float = 18.0,
    ) -> list[float]:
        """Square-root scale locality counts using one scale across panels."""
        denominator = max(float(global_max), 1.0)
        return [
            minimum
            + (maximum - minimum)
            * math.sqrt(max(float(value), 0.0) / denominator)
            for value in counts
        ]

    @classmethod
    def _aggregate_localities(
        cls,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Count matching resolved segments at each canonical locality."""
        if data.empty:
            return pd.DataFrame()

        grouped = (
            data.groupby(
                [
                    "locality_key",
                    "locality",
                    "state",
                    "country",
                    "iso3",
                    "continent",
                    "lat",
                    "lon",
                ],
                dropna=False,
            )
            .agg(
                segments=("video_id", "size"),
                unique_uploads=("video_id", "nunique"),
                duration_seconds=("duration_seconds", "sum"),
            )
            .reset_index()
        )
        grouped["duration_hours"] = grouped["duration_seconds"] / 3600.0
        return grouped

    @classmethod
    def _add_locality_bubbles(
        cls,
        fig: go.Figure,
        locality_table: pd.DataFrame,
        *,
        row: int,
        col: int,
        global_max: float,
    ) -> None:
        if locality_table.empty:
            return

        customdata = locality_table[
            [
                "locality",
                "state",
                "country",
                "segments",
                "unique_uploads",
                "duration_hours",
            ]
        ].to_numpy()

        fig.add_trace(
            go.Scattergeo(
                lon=locality_table["lon"],
                lat=locality_table["lat"],
                mode="markers",
                marker=dict(
                    size=cls._bubble_sizes(
                        locality_table["segments"],
                        global_max=global_max,
                    ),
                    opacity=0.72,
                    line=dict(width=0.3),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "State/region: %{customdata[1]}<br>"
                    "Country: %{customdata[2]}<br>"
                    "Segments in this panel: %{customdata[3]}<br>"
                    "Unique uploads: %{customdata[4]}<br>"
                    "Duration: %{customdata[5]:.2f} h"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _plot_category_world_maps(
        self,
        category_tables: list[tuple[str, pd.DataFrame]],
        *,
        title: str,
        filename: str,
    ) -> None:
        """Create one 2-column geographic small-multiple figure."""
        category_tables = [
            (label, table)
            for label, table in category_tables
            if table is not None and not table.empty
        ]
        if not category_tables:
            return

        cols = 2
        rows = math.ceil(len(category_tables) / cols)
        subplot_titles = [label for label, _ in category_tables]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[
                [{"type": "geo"} for _ in range(cols)]
                for _ in range(rows)
            ],
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        global_max = max(
            float(table["segments"].max())
            for _, table in category_tables
            if not table.empty
        )

        for panel_index, (_, table) in enumerate(category_tables):
            row = panel_index // cols + 1
            col = panel_index % cols + 1
            self._add_locality_bubbles(
                fig,
                table,
                row=row,
                col=col,
                global_max=global_max,
            )

        self._geo_panel_layout(fig, rows=rows, cols=cols)
        fig.update_layout(
            title=(f"{title}"),
            showlegend=False,
            margin=dict(
                l=0,
                r=0,
                t=80 if str(title).strip() else 0,
                b=0,
            ),
        )

        height = 470 if rows == 1 else 820
        self._save(fig, filename, width=1400, height=height)

    def plot_world_time_of_day(self, result: CrashAnalysisResult) -> None:
        """Map day, night, dusk/dawn, and unknown resolved crash segments."""
        data = self._valid_resolved_segments(result)
        if data.empty:
            return

        values = data["time_of_day"].fillna("unknown").astype(str).str.strip()
        values = values.replace("", "unknown")

        preferred = ["day", "night", "dusk/dawn", "dawn_dusk", "unknown"]
        available = list(dict.fromkeys(values.tolist()))
        categories = [value for value in preferred if value in available]
        categories.extend(value for value in available if value not in categories)

        panels: list[tuple[str, pd.DataFrame]] = []
        used_display_labels: set[str] = set()

        for category in categories:
            display = self._human_label(category)
            if display in used_display_labels:
                continue
            used_display_labels.add(display)
            subset = data.loc[values == category].copy()
            table = self._aggregate_localities(subset)
            if not table.empty:
                panels.append((display, table))

        self._plot_category_world_maps(
            panels,
            title="",
            filename="fig_world_time_of_day",
        )

    def plot_world_road_users(
        self,
        result: CrashAnalysisResult,
        *,
        top_n: int = 4,
    ) -> None:
        """Map the most frequent road-user categories in the resolved subset.

        Road users are multi-label. A segment can therefore contribute to more
        than one panel, which is stated directly in the figure subtitle.
        """
        data = self._valid_resolved_segments(result)
        if data.empty or "road_users" not in data.columns:
            return

        road_user_column = (
            "road_users_normalised"
            if "road_users_normalised" in data.columns
            else "road_users"
        )

        if road_user_column == "road_users":
            normalised_values = data["road_users"].map(
                _normalise_road_user_list
            )
        else:
            normalised_values = data[road_user_column]

        counter: dict[str, int] = {}
        for users in normalised_values:
            if not isinstance(users, list):
                continue
            for value in set(users):
                counter[value] = counter.get(value, 0) + 1

        selected = [
            value
            for value, _ in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[: max(1, top_n)]
        ]

        panels: list[tuple[str, pd.DataFrame]] = []
        for road_user in selected:
            mask = normalised_values.map(
                lambda values: (
                    isinstance(values, list)
                    and road_user in set(values)
                )
            )
            table = self._aggregate_localities(data.loc[mask].copy())
            if not table.empty:
                panels.append((_road_user_label(road_user), table))

        self._plot_category_world_maps(
            panels,
            title="",
            filename="fig_world_road_users",
        )

    @staticmethod
    def _synthetic_unit_interval(segment: pd.Series) -> float:
        """Return a stable pseudo-random number in [0, 1) for one segment."""
        identity_parts = [
            str(segment.get("segment_id") or ""),
            str(segment.get("video_id") or ""),
            str(segment.get("segment_index") or ""),
            str(segment.get("start_time") or ""),
            str(segment.get("end_time") or ""),
        ]
        identity = "|".join(identity_parts)
        digest = hashlib.sha256(
            f"{SYNTHETIC_MANNER_SEED}|{identity}".encode("utf-8")
        ).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return integer / float(2**64)

    @classmethod
    def _synthetic_manner_for_segment(cls, segment: pd.Series) -> str:
        """Assign one draft-only manner using the configured synthetic weights."""
        draw = cls._synthetic_unit_interval(segment)
        cumulative = 0.0
        final_value = "other"

        for value, weight in SYNTHETIC_MANNER_WEIGHTS.items():
            cumulative += float(weight)
            final_value = value
            if draw < cumulative:
                return value

        # Floating-point guard if configured weights sum to just under 1.
        return final_value

    @classmethod
    def _build_synthetic_manner_completion(
        cls,
        result: CrashAnalysisResult,
    ) -> pd.DataFrame:
        """Return a resolved collision frame with measured + synthetic manners.

        Measured/currently classified MMUCC manners are preserved exactly.
        Only otherwise unclassified accepted crash segments are synthetically
        completed. Near-collision segments are excluded because C9 manner of
        collision is a crash/collision element.

        The returned frame contains:
        * ``manner_of_collision_display``: measured or synthetic value
        * ``manner_value_source``: ``measured`` or ``synthetic``
        """
        data = cls._valid_resolved_segments(result)
        if data.empty:
            return data

        data = data.copy()

        measured_mask = (
            (data["crash_taxonomy_version"] == CRASH_TAXONOMY_VERSION)
            & (data["crash_taxonomy_status"] == "classified")
            & (data["event_kind"] == "collision")
            & data["manner_of_collision"].fillna("").astype(str).str.strip().ne("")
            & data["manner_of_collision"].fillna("").astype(str).str.strip().ne("unknown")
        )

        # crash_type comes from the earlier visual crash review and is available
        # before MMUCC backfill. A near collision must not receive a C9 manner.
        if "crash_type" in data.columns:
            near_collision_mask = (
                data["crash_type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                .eq("near_collision")
            )
        else:
            near_collision_mask = (
                data["event_kind"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                .eq("near_collision")
            )

        eligible_mask = ~near_collision_mask
        data = data.loc[eligible_mask].copy()
        measured_mask = measured_mask.loc[data.index]

        data["manner_of_collision_display"] = ""
        data["manner_value_source"] = ""

        # Preserve every actual classified value.
        data.loc[measured_mask, "manner_of_collision_display"] = (
            data.loc[measured_mask, "manner_of_collision"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        data.loc[measured_mask, "manner_value_source"] = "measured"

        pending_index = data.index[~measured_mask]
        for index in pending_index:
            data.at[index, "manner_of_collision_display"] = (
                cls._synthetic_manner_for_segment(data.loc[index])
            )
            data.at[index, "manner_value_source"] = "synthetic"

        return data

    def plot_world_manner_of_collision_synthetic(
        self,
        result: CrashAnalysisResult,
        *,
        top_n: int = 4,
    ) -> None:
        """Create a clearly labelled draft-only synthetic-completion map."""
        completed = self._build_synthetic_manner_completion(result)
        if completed.empty:
            return

        completed["manner_of_collision_display"] = (
            completed["manner_of_collision_display"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        completed = completed.loc[
            ~completed["manner_of_collision_display"].isin(["", "unknown"])
        ].copy()
        if completed.empty:
            return

        selected = (
            completed["manner_of_collision_display"]
            .value_counts()
            .head(max(1, top_n))
            .index.tolist()
        )

        panels: list[tuple[str, pd.DataFrame]] = []
        for manner in selected:
            table = self._aggregate_localities(
                completed.loc[
                    completed["manner_of_collision_display"] == manner
                ].copy()
            )
            if not table.empty:
                panels.append((self._human_label(manner), table))

        # Save an audit table so synthetic values can never be confused with
        # measured MMUCC classifications.
        tables_dir = self.output_dir.parent / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        audit_columns = [
            column
            for column in [
                "segment_id",
                "video_id",
                "segment_index",
                "start_time",
                "end_time",
                "locality",
                "state",
                "country",
                "crash_type",
                "event_kind",
                "crash_taxonomy_status",
                "crash_taxonomy_version",
                "manner_of_collision",
                "manner_of_collision_display",
                "manner_value_source",
            ]
            if column in completed.columns
        ]
        completed[audit_columns].to_csv(
            tables_dir / "manner_of_collision_synthetic_completion.csv",
            index=False,
        )

        measured_count = int(
            (completed["manner_value_source"] == "measured").sum()
        )
        synthetic_count = int(
            (completed["manner_value_source"] == "synthetic").sum()
        )

        self._plot_category_world_maps(
            panels,
            title=(
                "SYNTHETIC COMPLETION — NOT MEASURED DATA"
                f"<br><sup>{measured_count:,} measured + "
                f"{synthetic_count:,} synthetic pending assignments; "
                "draft visualisation only</sup>"
            ),
            filename="fig_world_manner_of_collision_synthetic",
        )

    def plot_world_manner_of_collision(
        self,
        result: CrashAnalysisResult,
        *,
        top_n: int = 4,
    ) -> None:
        """Map top MMUCC collision manners for classified resolved collisions."""
        data = self._valid_resolved_segments(result)
        if data.empty:
            return

        current = data.loc[
            (data["crash_taxonomy_version"] == CRASH_TAXONOMY_VERSION)
            & (data["crash_taxonomy_status"] == "classified")
            & (data["event_kind"] == "collision")
        ].copy()
        if current.empty:
            return

        current["manner_of_collision"] = (
            current["manner_of_collision"].fillna("").astype(str).str.strip()
        )
        current = current.loc[
            ~current["manner_of_collision"].isin(["", "unknown"])
        ].copy()
        if current.empty:
            return

        selected = (
            current["manner_of_collision"]
            .value_counts()
            .head(max(1, top_n))
            .index.tolist()
        )

        panels = []
        for manner in selected:
            table = self._aggregate_localities(
                current.loc[current["manner_of_collision"] == manner].copy()
            )
            if not table.empty:
                panels.append((self._human_label(manner), table))

        self._plot_category_world_maps(
            panels,
            title="",
            filename="fig_world_manner_of_collision",
        )

    def plot_world_first_harmful_event(
        self,
        result: CrashAnalysisResult,
        *,
        top_n: int = 4,
    ) -> None:
        """Map top MMUCC first-harmful-event categories."""
        data = self._valid_resolved_segments(result)
        if data.empty:
            return

        current = data.loc[
            (data["crash_taxonomy_version"] == CRASH_TAXONOMY_VERSION)
            & (data["crash_taxonomy_status"] == "classified")
            & (data["event_kind"] == "collision")
        ].copy()
        if current.empty:
            return

        current["first_harmful_event"] = (
            current["first_harmful_event"].fillna("").astype(str).str.strip()
        )
        current = current.loc[
            ~current["first_harmful_event"].isin(["", "unknown"])
        ].copy()
        if current.empty:
            return

        selected = (
            current["first_harmful_event"]
            .value_counts()
            .head(max(1, top_n))
            .index.tolist()
        )

        panels = []
        for event in selected:
            table = self._aggregate_localities(
                current.loc[current["first_harmful_event"] == event].copy()
            )
            if not table.empty:
                panels.append((self._human_label(event), table))

        self._plot_category_world_maps(
            panels,
            title="",
            filename="fig_world_first_harmful_event",
        )

    def plot_geographic_context_maps(
        self,
        result: CrashAnalysisResult,
        *,
        top_road_users: int = 4,
        top_manners: int = 4,
        top_first_harmful: int = 4,
    ) -> None:
        """Generate the optional geographic context figures."""
        self.plot_world_time_of_day(result)
        self.plot_world_road_users(result, top_n=max(1, top_road_users))

        # Always generate the empirical/currently classified MMUCC result.
        self.plot_world_manner_of_collision(
            result,
            top_n=max(1, top_manners),
        )

        # Optional draft-only projection. This never replaces the actual figure.
        if ENABLE_SYNTHETIC_MANNER_COMPLETION:
            self.plot_world_manner_of_collision_synthetic(
                result,
                top_n=max(1, top_manners),
            )

        self.plot_world_first_harmful_event(
            result,
            top_n=max(1, top_first_harmful),
        )


def print_summary(summary: dict) -> None:
    videos = summary["videos"]
    segments = summary["segments"]
    location = summary["location"]
    geography = summary["geography"]
    taxonomy = summary["taxonomy"]
    mapping = summary["mapping"]

    print("\n=== Crash corpus summary ===")
    print(f"Video records:             {videos['records']:,}")
    print(f"Complete videos:           {videos['complete']:,}")
    print(f"Metadata rejected:         {videos['text_rejected']:,}")
    print(f"Visual rejected:           {videos['visual_rejected']:,}")
    print(f"Visual errors:             {videos['visual_error']:,}")
    print(f"Accepted segments:         {segments['accepted']:,}")
    print(f"Retained duration:         {segments['retained_duration_hours']:.2f} h")

    print("\n=== Geographic coverage ===")
    print(f"Resolved segments:         {location['resolved']:,} ({location['resolved_pct']:.2f}%)")
    print(f"Unresolved segments:       {location['unresolved']:,} ({location['unresolved_pct']:.2f}%)")
    print(f"Canonical localities:      {geography['canonical_locality_entities']:,}")
    print(f"Countries/territories:     {geography['countries_or_territories']:,}")
    print(f"Continents:                {geography['continents']:,}")

    print("\n=== MMUCC taxonomy ===")
    print(f"Current version:           {taxonomy['current_version']}")
    print(
        f"Classified segments:       {taxonomy['classified_segments']:,} "
        f"({taxonomy['classified_pct_of_accepted']:.2f}%)"
    )
    print(
        f"Pending segments:          {taxonomy['pending_segments']:,} "
        f"({taxonomy['pending_pct_of_accepted']:.2f}%)"
    )

    print("\n=== Mapping consistency ===")
    print(f"Mapping rows:              {mapping['rows']:,}")
    print(f"Expanded mapped segments:  {mapping['expanded_segment_records']:,}")
    print(f"State resolved segments:   {mapping['state_resolved_segments']:,}")
    print(f"Counts match:              {mapping['segment_count_matches_state']}")


def main() -> None:
    args = build_parser().parse_args()

    analysis = CrashResultsAnalysis(args.state, args.mapping)
    result = analysis.analyse()

    # Keep raw model labels in ``road_users`` but use a controlled,
    # paper-facing road-user taxonomy for tables and figures.
    _apply_road_user_normalisation(result)

    analysis.write_outputs(result, args.output_dir)
    print_summary(result.summary)

    if args.record_snapshot:
        path = analysis.append_snapshot(
            result,
            args.output_dir,
            label=args.snapshot_label,
        )
        print(f"\nRecorded snapshot: {path}")

    if not args.no_plots:
        figure_dir = args.output_dir / "figures"

        # Older versions exported SVGs. Remove those stale generated files so
        # the figure directories reflect the current HTML/PNG/PDF policy.
        _remove_stale_svg_outputs(figure_dir, args.publication_dir)

        plotter = GeographicCrashResultsPlotter(
            output_dir=figure_dir,
            publication_dir=args.publication_dir,
        )
        plotter.plot_all(result, top_country_n=max(1, args.top_countries))
        plotter.plot_geographic_context_maps(
            result,
            top_road_users=max(1, args.map_top_road_users),
            top_manners=max(1, args.map_top_manners),
            top_first_harmful=max(1, args.map_top_first_harmful),
        )
        plotter.plot_growth_snapshots(args.output_dir / "snapshots.csv")
        print(f"Figures written to:        {figure_dir}")
        print(f"Publication copies:        {args.publication_dir}")

    print(f"Tables and summary:        {args.output_dir}")


if __name__ == "__main__":
    main()
