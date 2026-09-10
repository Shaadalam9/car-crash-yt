"""Publication-oriented Plotly figures for the crash corpus Results section."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.analytics.crash_results import CrashAnalysisResult


class CrashResultsPlotter:
    """Create CHI-facing figures from :class:`CrashAnalysisResult`."""

    def __init__(
        self,
        output_dir: str | Path,
        publication_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.publication_dir = Path(publication_dir) if publication_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.publication_dir:
            self.publication_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _base_layout(fig: go.Figure, *, width: int = 1100, height: int = 650) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            width=width,
            height=height,
            font=dict(family="Arial", size=18),
            margin=dict(l=90, r=35, t=70, b=80),
            legend_title_text="",
        )
        return fig

    def _save(self, fig: go.Figure, name: str, *, width: int = 1100, height: int = 650) -> None:
        self._base_layout(fig, width=width, height=height)

        html_path = self.output_dir / f"{name}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")

        produced: list[Path] = [html_path]
        for suffix in ("png", "svg", "pdf"):
            path = self.output_dir / f"{name}.{suffix}"
            try:
                fig.write_image(path, width=width, height=height, scale=2 if suffix == "png" else 1)
                produced.append(path)
            except Exception as exc:
                print(f"Warning: could not write {path.name}: {exc}")

        if self.publication_dir:
            for path in produced:
                shutil.copy2(path, self.publication_dir / path.name)

    @staticmethod
    def _bar(
        table: pd.DataFrame,
        *,
        label_col: str,
        value_col: str,
        title: str,
        x_title: str,
        horizontal: bool = True,
        top_n: int | None = None,
    ) -> go.Figure | None:
        if table is None or table.empty:
            return None

        data = table.copy()
        if top_n is not None:
            data = data.nlargest(top_n, value_col)
        data = data.sort_values(value_col, ascending=horizontal)

        if horizontal:
            fig = px.bar(data, x=value_col, y=label_col, orientation="h", text=value_col)
            fig.update_xaxes(title=x_title)
            fig.update_yaxes(title="")
            fig.update_traces(textposition="outside", cliponaxis=False)
        else:
            fig = px.bar(data, x=label_col, y=value_col, text=value_col)
            fig.update_xaxes(title="")
            fig.update_yaxes(title=x_title)
            fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(title=title)
        return fig

    def plot_video_status(self, result: CrashAnalysisResult) -> None:
        table = result.tables["video_status"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="Video processing outcomes",
            x_title="Video records",
        )
        if fig:
            self._save(fig, "fig_video_processing_outcomes", height=520)

    def plot_location_status(self, result: CrashAnalysisResult) -> None:
        table = result.tables["location_status"].copy()
        if table.empty:
            return
        preferred = {
            "resolved": 0,
            "not_found": 1,
            "no_evidence": 2,
            "rejected_result": 3,
            "failed": 4,
            "missing_location_record": 5,
            "missing_location_status": 6,
        }
        table["order"] = table["status"].map(lambda value: preferred.get(value, 99))
        table = table.sort_values(["order", "count"], ascending=[True, False])
        table["text"] = table.apply(
            lambda row: f"{int(row['count']):,} ({float(row['percentage']):.2f}%)",
            axis=1,
        )
        fig = px.bar(table, x="count", y="label", orientation="h", text="text")
        fig.update_layout(title="Geographic resolution outcomes")
        fig.update_xaxes(title="Accepted segments")
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_traces(textposition="outside", cliponaxis=False)
        self._save(fig, "fig_location_resolution_outcomes", height=560)

    def plot_world_map(self, result: CrashAnalysisResult) -> None:
        table = result.tables["localities"].copy()
        if table.empty:
            return
        table = table.dropna(subset=["lat", "lon"])
        if table.empty:
            return

        max_count = max(float(table["resolved_segments"].max()), 1.0)
        table["marker_size"] = table["resolved_segments"].map(
            lambda value: 4.0 + 16.0 * math.sqrt(float(value) / max_count)
        )

        customdata = table[
            ["locality", "state", "country", "resolved_segments", "unique_uploads", "resolved_duration_hours"]
        ].to_numpy()
        fig = go.Figure(
            go.Scattergeo(
                lon=table["lon"],
                lat=table["lat"],
                mode="markers",
                marker=dict(
                    size=table["marker_size"],
                    color=table["resolved_segments"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Resolved<br>segments"),
                    opacity=0.75,
                    line=dict(width=0.3),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "State/region: %{customdata[1]}<br>"
                    "Country: %{customdata[2]}<br>"
                    "Resolved segments: %{customdata[3]}<br>"
                    "Unique uploads: %{customdata[4]}<br>"
                    "Duration: %{customdata[5]:.2f} h"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(238,238,238)",
            showcountries=True,
            countrycolor="white",
            showcoastlines=True,
            coastlinecolor="white",
        )
        fig.update_layout(
            title=(
                "Geographic coverage of verified crash localities"
                "<br><sup>Points are canonical locality reference coordinates, not crash-event coordinates.</sup>"
            )
        )
        self._save(fig, "fig_world_geographic_coverage", width=1400, height=760)

    def plot_continents(self, result: CrashAnalysisResult) -> None:
        table = result.tables["continents"].copy()
        if table.empty:
            return
        table["label"] = table["continent"].replace("", "Unknown")
        fig = self._bar(
            table,
            label_col="label",
            value_col="resolved_segments",
            title="Resolved crash segments by continent",
            x_title="Resolved segments",
        )
        if fig:
            self._save(fig, "fig_resolved_segments_by_continent", height=560)

    def plot_top_countries(self, result: CrashAnalysisResult, top_n: int = 15) -> None:
        table = result.tables["countries"].copy()
        if table.empty:
            return
        table["label"] = table.apply(
            lambda row: f"{row['country']} ({row['iso3']})" if row.get("iso3") else str(row["country"]),
            axis=1,
        )
        fig = self._bar(
            table,
            label_col="label",
            value_col="resolved_segments",
            title=f"Top {top_n} countries by resolved crash segments",
            x_title="Resolved segments",
            top_n=top_n,
        )
        if fig:
            self._save(fig, "fig_top_countries_resolved_segments", height=780)

    def plot_taxonomy_coverage(self, result: CrashAnalysisResult) -> None:
        table = result.tables["taxonomy_coverage"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="MMUCC taxonomy processing status",
            x_title="Accepted segments",
        )
        if fig:
            self._save(fig, "fig_taxonomy_coverage", height=560)

    def plot_event_kind(self, result: CrashAnalysisResult) -> None:
        table = result.tables["taxonomy_event_kind"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="Event kind among classified segments",
            x_title="Classified segments",
        )
        if fig:
            self._save(fig, "fig_taxonomy_event_kind", height=500)

    def plot_manner(self, result: CrashAnalysisResult, top_n: int = 12) -> None:
        table = result.tables["taxonomy_manner"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="Manner of collision among classified collisions",
            x_title="Classified collision segments",
            top_n=top_n,
        )
        if fig:
            self._save(fig, "fig_taxonomy_manner_of_collision", height=720)

    def plot_first_harmful_event(self, result: CrashAnalysisResult, top_n: int = 15) -> None:
        table = result.tables["taxonomy_first_harmful_event"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="First harmful event among classified collisions",
            x_title="Classified collision segments",
            top_n=top_n,
        )
        if fig:
            self._save(fig, "fig_taxonomy_first_harmful_event", height=800)

    def plot_sequence_events(self, result: CrashAnalysisResult, top_n: int = 15) -> None:
        table = result.tables["taxonomy_sequence_events"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="segments",
            title="Most frequent sequence-of-events elements",
            x_title="Classified collision segments containing event",
            top_n=top_n,
        )
        if fig:
            self._save(fig, "fig_taxonomy_sequence_events", height=800)

    def plot_road_users(self, result: CrashAnalysisResult, top_n: int = 12) -> None:
        table = result.tables["road_users"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="segments",
            title="Road users represented in accepted crash segments",
            x_title="Segments containing road-user type",
            top_n=top_n,
        )
        if fig:
            self._save(fig, "fig_road_users", height=700)

    def plot_time_of_day(self, result: CrashAnalysisResult) -> None:
        table = result.tables["time_of_day"]
        fig = self._bar(
            table,
            label_col="label",
            value_col="count",
            title="Time of day of accepted crash segments",
            x_title="Accepted segments",
        )
        if fig:
            self._save(fig, "fig_time_of_day", height=500)

    def plot_growth_snapshots(self, snapshots_path: str | Path) -> None:
        path = Path(snapshots_path)
        if not path.is_file():
            return
        snapshots = pd.read_csv(path)
        if len(snapshots) < 2:
            return

        snapshots["recorded_at_utc"] = pd.to_datetime(snapshots["recorded_at_utc"], errors="coerce")
        snapshots = snapshots.dropna(subset=["recorded_at_utc"]).sort_values("recorded_at_utc")
        if len(snapshots) < 2:
            return

        fig = go.Figure()
        for column, label in (
            ("accepted_segments", "Accepted segments"),
            ("resolved_segments", "Resolved segments"),
            ("taxonomy_classified_segments", "Taxonomy classified"),
        ):
            if column in snapshots.columns:
                fig.add_trace(
                    go.Scatter(
                        x=snapshots["recorded_at_utc"],
                        y=snapshots[column],
                        mode="lines+markers",
                        name=label,
                    )
                )
        fig.update_layout(title="Dataset state across explicitly recorded snapshots")
        fig.update_xaxes(title="Snapshot time")
        fig.update_yaxes(title="Segments")
        self._save(fig, "fig_dataset_growth_snapshots", height=620)

    def plot_all(self, result: CrashAnalysisResult, *, top_country_n: int = 15) -> None:
        self.plot_video_status(result)
        self.plot_location_status(result)
        self.plot_world_map(result)
        self.plot_continents(result)
        self.plot_top_countries(result, top_n=top_country_n)
        self.plot_taxonomy_coverage(result)
        self.plot_event_kind(result)
        self.plot_manner(result)
        self.plot_first_harmful_event(result)
        self.plot_sequence_events(result)
        self.plot_road_users(result)
        self.plot_time_of_day(result)
