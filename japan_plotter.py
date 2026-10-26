"""
japan_plotter.py

Japan-specific extension of GenericMapPlotter (v2 Architecture).

Includes:
    * Strict Unit Contract (metres for station corridors, points for strokes/fonts)
    * Targeted admin_level querying (avoids downloading national level-2 borders)
    * Hybrid Administrative + Place Node fallback system:
      Automatically renders 'place=quarter/neighbourhood/suburb/hamlet' point nodes
      when non-major cities lack level 8/9 administrative polygons.
    * Automatic deduplication between polygon centroids and point nodes.
"""

from __future__ import annotations

import logging
import math
import random
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from generic_plotter import (
    GenericMapPlotter,
    decorate_map_layout,
)
from map_styles import JapanMapStyle


logger = logging.getLogger("GenericPlotter")


plt.rcParams["font.sans-serif"] = [
    "Noto Sans JP",
    "Hiragino Sans",
    "Meiryo",
    "Source Han Sans JP",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


# Priority order when rendering fallback place nodes (larger settlements first)
_PLACE_HIERARCHY = {
    "town": 1,
    "suburb": 2,
    "borough": 2,
    "village": 3,
    "quarter": 4,
    "neighbourhood": 5,
    "hamlet": 6,
    "isolated_dwelling": 7,
}

# Equivalent admin_level mapping so place nodes inherit harmonious font sizes
_PLACE_TO_ADMIN_LEVEL = {
    "town": 7,
    "suburb": 7,
    "borough": 7,
    "village": 8,
    "quarter": 8,
    "neighbourhood": 9,
    "hamlet": 9,
    "isolated_dwelling": 9,
}


class JapanMapPlotter(GenericMapPlotter):
    """
    Japan-specific map plotter.
    Inherits the generic geographic and infrastructure engine and adds
    Japanese human-geography layers with rural/suburban point-node fallbacks.
    """

    def __init__(
        self,
        bbox=None,
        center_point=None,
        dist=5000,
        custom_linewidths=None,
        custom_colors=None,
        bg_color=None,
        land_color=None,
        admin_widths=None,
        projection="wgs84",
        style: JapanMapStyle | None = None,
        cache=None,
        cache_enabled=True,
        cache_dir=None,
        cache_ttl_days=30,
        clip_features=True,
        clip_buffer_ratio=0.05,
    ):
        logger.info("Initializing JapanMapPlotter...")

        style = style or JapanMapStyle()

        if admin_widths is not None:
            style.japan_admin.boundary_widths_pt = tuple(admin_widths)

        if len(style.japan_admin.boundary_widths_pt) != 10:
            raise ValueError(
                "boundary_widths_pt must contain exactly 10 admin levels."
            )

        super().__init__(
            bbox=bbox,
            center_point=center_point,
            dist=dist,
            custom_linewidths=custom_linewidths,
            custom_colors=custom_colors,
            bg_color=bg_color,
            land_color=land_color,
            projection=projection,
            style=style,
            cache=cache,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
            clip_features=clip_features,
            clip_buffer_ratio=clip_buffer_ratio,
        )

        self.admin_widths = style.japan_admin.boundary_widths_pt

        self.gdf_admins: dict[int, gpd.GeoDataFrame] = {}
        self.gdf_local_places: gpd.GeoDataFrame | None = None
        self.gdf_railways: gpd.GeoDataFrame | None = None
        self.gdf_exits: gpd.GeoDataFrame | None = None
        self.gdf_stations: gpd.GeoDataFrame | None = None
        self.gdf_piers: gpd.GeoDataFrame | None = None

        self._admins_fetched = False
        self._places_fetched = False

    # ------------------------------------------------------------------
    # Administrative boundaries & Local Place Nodes (Hybrid System)
    # ------------------------------------------------------------------

    def _fetch_and_categorize_admins(self):
        """
        Fetch Japanese administrative boundaries once and categorize by admin_level.
        Explicitly filters out historical entities (e.g. ancient provinces like Shimosa).
        """
        if self._admins_fetched:
            return

        logger.info(
            "Fetching and categorizing Japanese administrative boundaries..."
        )

        tags = {
            "boundary": "administrative",
            "admin_level": ["4", "5", "6", "7", "8", "9", "10"],
        }

        raw_gdf = self._fetch_features(
            tags,
            layer_name="jp_admin_boundaries",
        )

        if raw_gdf is not None and not raw_gdf.empty:
            df = raw_gdf[
                raw_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
            ].copy()

            # --------------------------------------------------------------
            # 🌟 FILTER 1: Drop historical boundaries (historic=* or historic:place=*)
            # --------------------------------------------------------------
            historic_cols = [col for col in df.columns if "historic" in col.lower()]
            is_historic = pd.Series(False, index=df.index)
            
            for col in historic_cols:
                is_historic = is_historic | df[col].notna() & (df[col].astype(str).str.lower() != "nan")

            if "historic" in df.columns:
                # If 'historic' is explicitly set to something like 'province' or 'yes'
                is_historic = is_historic | df["historic"].notna()

            # --------------------------------------------------------------
            # 🌟 FILTER 2: Drop entities with a past 'end_date'
            # --------------------------------------------------------------
            if "end_date" in df.columns:
                has_ended = df["end_date"].notna() & (df["end_date"].astype(str).str.strip() != "")
                is_historic = is_historic | has_ended

            # Keep only modern, active administrative boundaries
            df_modern = df[~is_historic].copy()

            logger.info(
                "Filtered out %d historical boundaries (e.g., ancient provinces). Remaining modern: %d",
                len(df) - len(df_modern),
                len(df_modern),
            )

            if "admin_level" in df_modern.columns:
                df_modern["admin_level_clean"] = df_modern["admin_level"].apply(
                    lambda value: (
                        str(value[0]) if isinstance(value, list) else str(value)
                    )
                )

                for level_string, group in df_modern.groupby(
                    "admin_level_clean"
                ):
                    if (
                        level_string.replace(".", "", 1).isdigit()
                        and level_string != "nan"
                    ):
                        level = int(float(level_string))
                        self.gdf_admins[level] = group
                        logger.debug(
                            "Admin level %d: %d polygon boundaries.",
                            level,
                            len(group),
                        )

        self._admins_fetched = True

    def _fetch_local_places(self):
        """
        Fetch point/polygon settlement features (neighbourhood, quarter, Oaza,
        suburb, hamlet, village). Crucial for non-major Japanese cities where
        Chome/Machicho are mapped as nodes rather than administrative polygons.
        """
        if self._places_fetched:
            return

        logger.info(
            "Fetching local place nodes (neighbourhood/quarter/suburb fallback)..."
        )

        tags = {
            "place": list(_PLACE_HIERARCHY.keys()),
        }

        raw_places = self._fetch_features(
            tags,
            layer_name="jp_local_places",
        )

        if raw_places is not None and not raw_places.empty:
            df = raw_places.copy()
            if "place" in df.columns:
                df["place_clean"] = df["place"].apply(
                    lambda v: str(v[0]) if isinstance(v, list) else str(v)
                )
                df = df[df["place_clean"].isin(_PLACE_HIERARCHY.keys())].copy()
                df["place_rank"] = df["place_clean"].map(_PLACE_HIERARCHY).fillna(99)
                df.sort_values("place_rank", inplace=True)
                self.gdf_local_places = df
            else:
                self.gdf_local_places = gpd.GeoDataFrame(
                    geometry=[], crs=self.crs
                )
        else:
            self.gdf_local_places = gpd.GeoDataFrame(
                geometry=[], crs=self.crs
            )

        self._places_fetched = True

    def draw_admins(self):
        """
        Draw Japanese administrative boundaries.
        Fills are placed at zorder 0.15 (underneath water at 0.20) so coastal
        zones never bleed into the ocean.
        """
        self._fetch_and_categorize_admins()

        style = self.style.japan_admin

        for index, width_pt in enumerate(style.boundary_widths_pt):
            level = index + 1
            if width_pt <= 0:
                continue

            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty:
                continue

            if level in {7, 8, 9}:
                color_ids = list(range(len(gdf)))
                random.seed(42 + level)
                random.shuffle(color_ids)

                gdf_fill = gdf.copy()
                gdf_fill["color_id"] = color_ids

                alpha_val = style.fill_alpha.get(level, 0.1)

                gdf_fill.plot(
                    ax=self.ax,
                    column="color_id",
                    cmap=style.fill_colormap,
                    alpha=alpha_val,
                    edgecolor="none",
                    zorder=0.15 + level * 0.001,
                )

            gdf.plot(
                ax=self.ax,
                facecolor="none",
                edgecolor=style.boundary_color,
                linewidth=self._get_stroke_width(width_pt),
                linestyle=style.boundary_linestyle,
                alpha=style.boundary_alpha,
                zorder=0.201 + level * 0.001,
            )

        return self

    # ------------------------------------------------------------------
    # Bilingual Label Formatting Helper
    # ------------------------------------------------------------------

    def _extract_bilingual_names(
        self, row: pd.Series, uppercase_secondary: bool = False
    ) -> tuple[str, str]:
        """Extract and format primary (Kanji) and secondary (Romaji/English) names."""
        primary_name = str(
            row.get("name") or row.get("name:ja") or ""
        ).strip()

        if (
            not primary_name
            or primary_name.lower() in {"none", "nan"}
            or "所属未定地" in primary_name
        ):
            return "", ""

        secondary_name = ""
        for tag in ["name:ja_rm", "name:ja-Latn", "name:ja:Latn", "name:en"]:
            value = row.get(tag)
            if pd.notna(value) and str(value).strip().lower() not in {
                "",
                "nan",
                "none",
            }:
                secondary_name = str(value).strip()
                break

        if secondary_name and uppercase_secondary:
            secondary_name = secondary_name.upper()

        if (
            " " in secondary_name or "-" in secondary_name
        ) and len(secondary_name) >= 18:
            parts = re.split(r"[-\s]", secondary_name)
            if parts and parts[0]:
                secondary_name = (
                    parts[0][0].upper() + ".-" + "-".join(parts[1:])
                )

        return primary_name, secondary_name

    def _render_bilingual_label(
        self,
        x: float,
        y: float,
        primary_name: str,
        secondary_name: str,
        primary_fontsize: float,
        secondary_fontsize: float,
        text_color: str,
        y_offset: float,
        alpha_primary: float = 0.85,
        alpha_secondary: float = 0.70,
        zorder: float = 6.0,
    ):
        """Render a clean, shadow-less two-line Kanji + Romaji label."""
        display_pri = (
            " ".join(primary_name)
            if len(primary_name) < 4
            else primary_name
        )

        if secondary_name:
            self.ax.text(
                x,
                y + y_offset,
                display_pri,
                color=text_color,
                fontsize=primary_fontsize,
                ha="center",
                va="bottom",
                zorder=zorder,
                clip_on=True,
                fontweight="bold",
                alpha=alpha_primary,
            )
            self.ax.text(
                x,
                y - y_offset,
                secondary_name,
                color=text_color,
                fontsize=secondary_fontsize,
                ha="center",
                va="top",
                zorder=zorder,
                clip_on=True,
                fontweight="normal",
                alpha=alpha_secondary,
            )
        else:
            self.ax.text(
                x,
                y,
                display_pri,
                color=text_color,
                fontsize=primary_fontsize,
                ha="center",
                va="center",
                zorder=zorder,
                clip_on=True,
                fontweight="bold",
                alpha=alpha_primary,
            )

    def draw_admin_labels(
        self,
        levels=(4, 6, 7, 8, 9),
        min_distance_ratio=0.015,
        include_place_nodes: bool = True,
    ):
        """
        Draw bilingual Japanese administrative labels.

        Hybrid Strategy
        ---------------
        1. Renders polygon-based administrative labels for requested ``levels``.
        2. If ``include_place_nodes=True``, automatically supplements the map
           with point/polygon ``place=quarter/neighbourhood/suburb/hamlet``
           features that were not already represented by administrative polygons.
           This prevents rural or suburban maps from looking empty.
        """
        if self.ax is None:
            raise ValueError("Canvas not found.")

        self._fetch_and_categorize_admins()

        style = self.style.japan_admin
        target_levels = (
            [levels]
            if isinstance(levels, int)
            else (list(levels) if levels else [4, 6, 7, 8, 9])
        )

        placed_labels: list[tuple[float, float]] = []
        placed_names: set[str] = set()

        min_dist = (self.maxx - self.minx) * min_distance_ratio
        y_offset = (self.maxy - self.miny) * style.label_y_offset_ratio

        # --------------------------------------------------------------
        # Pass 1: Polygon-based Administrative Boundaries
        # --------------------------------------------------------------
        for level in sorted(target_levels):
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty:
                continue

            primary_fontsize = style.label_font_sizes_pt.get(level, 10.0)
            secondary_fontsize = max(
                6.0, primary_fontsize * style.label_secondary_ratio
            )

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                raw_pri, sec_name = self._extract_bilingual_names(
                    row, uppercase_secondary=(level == 9)
                )
                if not raw_pri:
                    continue

                if isinstance(geom, MultiPolygon):
                    geom = max(geom.geoms, key=lambda polygon: polygon.area)

                center_pt = (
                    geom.centroid
                    if geom.contains(geom.centroid)
                    else geom.representative_point()
                )
                x, y = center_pt.x, center_pt.y

                if not (
                    self.minx <= x <= self.maxx
                    and self.miny <= y <= self.maxy
                ):
                    continue

                if any(
                    math.hypot(x - px, y - py) < min_dist
                    for px, py in placed_labels
                ):
                    continue

                placed_labels.append((x, y))
                placed_names.add(raw_pri)

                self._render_bilingual_label(
                    x=x,
                    y=y,
                    primary_name=raw_pri,
                    secondary_name=sec_name,
                    primary_fontsize=primary_fontsize,
                    secondary_fontsize=secondary_fontsize,
                    text_color=style.label_color,
                    y_offset=y_offset,
                )
                logger.info("Rendering %s...", raw_pri)

        # --------------------------------------------------------------
        # Pass 2: Point/Polygon 'place=*' Fallback for Non-Major Cities
        # --------------------------------------------------------------
        if include_place_nodes:
            self._fetch_local_places()

            if (
                self.gdf_local_places is not None
                and not self.gdf_local_places.empty
            ):
                custom_place_sizes = getattr(
                    style, "place_font_sizes_pt", {}
                )
                fallback_count = 0

                for _, row in self.gdf_local_places.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue

                    place_type = row.get("place_clean", "neighbourhood")
                    equiv_level = _PLACE_TO_ADMIN_LEVEL.get(place_type, 9)

                    # Only draw place types whose equivalent level is requested
                    if equiv_level not in target_levels and not (
                        equiv_level >= 8
                        and any(lvl >= 8 for lvl in target_levels)
                    ):
                        continue

                    raw_pri, sec_name = self._extract_bilingual_names(
                        row, uppercase_secondary=(equiv_level == 9)
                    )
                    if not raw_pri or raw_pri in placed_names:
                        continue

                    if geom.geom_type in ("Point", "MultiPoint"):
                        pt = (
                            geom
                            if geom.geom_type == "Point"
                            else geom.geoms[0]
                        )
                    else:
                        pt = (
                            geom.centroid
                            if geom.contains(geom.centroid)
                            else geom.representative_point()
                        )

                    x, y = pt.x, pt.y
                    if not (
                        self.minx <= x <= self.maxx
                        and self.miny <= y <= self.maxy
                    ):
                        continue

                    if any(
                        math.hypot(x - px, y - py) < min_dist
                        for px, py in placed_labels
                    ):
                        continue

                    pri_fs = custom_place_sizes.get(
                        place_type,
                        style.label_font_sizes_pt.get(equiv_level, 9.5),
                    )
                    sec_fs = max(6.0, pri_fs * style.label_secondary_ratio)

                    placed_labels.append((x, y))
                    placed_names.add(raw_pri)
                    fallback_count += 1

                    self._render_bilingual_label(
                        x=x,
                        y=y,
                        primary_name=raw_pri,
                        secondary_name=sec_name,
                        primary_fontsize=pri_fs,
                        secondary_fontsize=sec_fs,
                        text_color=style.label_color,
                        y_offset=y_offset,
                        alpha_primary=0.80,
                        alpha_secondary=0.65,
                        zorder=5.9,
                    )
                    #logger.info("Rendering %s...", raw_pri)

                logger.info(
                    "Supplemented map with %d local place-node labels.",
                    fallback_count,
                )

        return self

    # ------------------------------------------------------------------
    # Railways
    # ------------------------------------------------------------------

    def _fetch_railways(self):
        """Fetch and normalize Japanese railway geometries."""
        if self.gdf_railways is not None:
            return

        tags = {
            "railway": [
                "rail",
                "subway",
                "light_rail",
                "tram",
                "narrow_gauge",
                "monorail",
                "funicular",
            ]
        }

        raw_rail = self._fetch_features(tags, layer_name="jp_railways")

        if raw_rail is None or raw_rail.empty:
            self.gdf_railways = gpd.GeoDataFrame(geometry=[], crs=self.crs)
            return

        self.gdf_railways = raw_rail[
            raw_rail.geometry.type.isin(["LineString", "MultiLineString"])
        ].copy()

    def draw_railways(self):
        """Draw Japan's complex railway network (stroke widths in POINTS)."""
        self._fetch_railways()

        if self.gdf_railways.empty:
            return self

        style = self.style.japan_railways

        color_col = (
            "colour"
            if "colour" in self.gdf_railways.columns
            else ("color" if "color" in self.gdf_railways.columns else None)
        )

        def get_valid_color(value):
            if pd.isna(value) or str(value).strip().lower() in ["", "nan"]:
                return None
            return str(value).strip().lower().split(";")[0]

        self.gdf_railways["parsed_color"] = (
            self.gdf_railways[color_col].apply(get_valid_color)
            if color_col
            else None
        )

        high_capacity = self.gdf_railways[
            self.gdf_railways["railway"].isin(["rail", "narrow_gauge"])
        ]
        other_rail = self.gdf_railways[
            ~self.gdf_railways["railway"].isin(["rail", "narrow_gauge"])
        ]

        if not other_rail.empty:
            other_rail = other_rail.copy()
            other_rail["render_color"] = other_rail["parsed_color"].fillna(
                style.default_color
            )

            for color, group in other_rail.groupby("render_color"):
                group.plot(
                    ax=self.ax,
                    color=color,
                    linewidth=self._get_stroke_width(style.urban_width_pt),
                    zorder=style.urban_zorder,
                    alpha=style.alpha,
                )

        if not high_capacity.empty:
            high_capacity.plot(
                ax=self.ax,
                color=style.heavy_base_color,
                linewidth=self._get_stroke_width(style.heavy_width_pt),
                zorder=style.heavy_zorder,
            )

            dash_df = high_capacity.copy()
            dash_df["dash_color"] = dash_df["parsed_color"].fillna(
                style.heavy_dash_color
            )

            for color, group in dash_df.groupby("dash_color"):
                group.plot(
                    ax=self.ax,
                    color=color,
                    linewidth=self._get_stroke_width(style.heavy_dash_width_pt),
                    linestyle=(0, style.dash_pattern),
                    zorder=style.heavy_dash_zorder,
                )

        return self

    # ------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------

    def _fetch_stations(self):
        """Fetch and filter Japanese station infrastructure (excluding bus stops)."""
        if self.gdf_stations is not None:
            return

        tags = {
            "public_transport": "stop_area",
            "building": "train_station",
            "railway": ["station", "platform"],
        }

        gdf = self._fetch_features(tags, layer_name="jp_stations")

        if gdf is None or gdf.empty:
            self.gdf_stations = gpd.GeoDataFrame(geometry=[], crs=self.crs)
            return

        is_bus = pd.Series(False, index=gdf.index)
        for col in ["highway", "amenity"]:
            if col in gdf.columns:
                is_bus = is_bus | gdf[col].isin(["bus_station", "bus_stop"])

        self.gdf_stations = gdf[~is_bus].copy()

    def _prepare_station_footprint(self, station_gdf):
        """
        Build physical station footprints in metric_crs.
        Station corridors are buffered by physical metres (corridor_width_m / 2).
        """
        style = self.style.japan_stations
        render_geoms = []

        metric = self._to_metric(station_gdf)

        for geom in metric.geometry:
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type in ("Point", "MultiPoint"):
                continue

            if geom.geom_type in ("LineString", "MultiLineString"):
                geom = geom.buffer(style.corridor_width_m / 2.0)

            render_geoms.append(geom)

        if not render_geoms:
            return None

        merged = unary_union(render_geoms)

        return gpd.GeoDataFrame(
            geometry=[merged],
            crs=self.metric_crs,
        ).to_crs(self.crs)

    def draw_stations(self, min_distance_ratio=None):
        """Draw Japanese station complexes and bilingual signboards."""
        self._fetch_stations()

        if self.gdf_stations.empty:
            return self

        style = self.style.japan_stations

        min_distance_ratio = (
            style.label_collision_ratio
            if min_distance_ratio is None
            else min_distance_ratio
        )

        footprint = self._prepare_station_footprint(self.gdf_stations)

        if footprint is not None:
            footprint.plot(
                ax=self.ax,
                facecolor=style.footprint_facecolor,
                edgecolor=style.footprint_edgecolor,
                linewidth=style.footprint_linewidth_pt,
                zorder=style.footprint_zorder,
            )

        station_groups = {}

        for _, row in self.gdf_stations.iterrows():
            primary = str(
                row.get("name") or row.get("name:ja") or ""
            ).strip()

            if not primary or primary.lower() in ["nan", "none"]:
                continue

            secondary = next(
                (
                    str(value).strip()
                    for value in (
                        row.get(tag)
                        for tag in [
                            "name:en",
                            "name:ja_rm",
                            "name:ja-Latn",
                            "name:ja:Latn",
                        ]
                    )
                    if pd.notna(value)
                    and str(value).strip().lower() not in ["", "nan", "none"]
                ),
                "",
            )

            if not secondary:
                continue

            station_groups.setdefault((primary, secondary), []).append(
                row.geometry
            )

        placed_stations = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        bbox_style = dict(
            facecolor=style.label_facecolor,
            alpha=style.label_alpha,
            edgecolor=style.label_edgecolor,
            linewidth=style.label_linewidth_pt,
            boxstyle=f"square,pad={style.label_pad}",
        )

        for (primary, secondary), geometries in station_groups.items():
            merged = unary_union(geometries)
            pt = (
                merged.centroid
                if merged.contains(merged.centroid)
                else merged.representative_point()
            )

            if not (
                self.minx <= pt.x <= self.maxx
                and self.miny <= pt.y <= self.maxy
            ):
                continue

            if any(
                math.hypot(pt.x - px, pt.y - py) < min_dist
                for px, py in placed_stations
            ):
                continue

            placed_stations.append((pt.x, pt.y))

            self.ax.text(
                pt.x,
                pt.y,
                f"{primary}\n{secondary}",
                color=style.label_color,
                fontsize=style.label_font_size_pt,
                ha="center",
                va="center",
                zorder=style.label_zorder,
                clip_on=True,
                fontfamily="sans-serif",
                fontweight="bold",
                bbox=bbox_style,
                linespacing=1.2,
            )

        return self

    # ------------------------------------------------------------------
    # Highway references, Exits & Piers
    # ------------------------------------------------------------------

    def draw_highway_refs(self, min_distance_ratio=None):
        """Draw Japanese highway reference boxes."""
        if self.gdf_edges is None or self.gdf_edges.empty:
            return self

        style = self.style.japan_labels

        min_distance_ratio = (
            style.highway_collision_ratio
            if min_distance_ratio is None
            else min_distance_ratio
        )

        placed_refs = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        bbox_style = dict(
            facecolor=style.highway_facecolor,
            alpha=0.85,
            edgecolor=style.highway_edgecolor,
            linewidth=style.highway_linewidth_pt,
            boxstyle="square,pad=0.25",
        )

        for _, row in self.gdf_edges.iterrows():
            ref = str(row.get("ref")).strip()
            highway = row.get("highway")
            if isinstance(highway, list):
                highway = highway[0]

            if (
                ref.lower() == "nan"
                or not ref
                or highway
                not in ["motorway", "motorway_link", "trunk", "trunk_link"]
            ):
                continue

            geom = row.geometry
            if (
                geom is None
                or geom.is_empty
                or geom.length < (self.maxx - self.minx) * 0.08
            ):
                continue

            pt = geom.interpolate(0.5, normalized=True)

            if not (
                self.minx <= pt.x <= self.maxx
                and self.miny <= pt.y <= self.maxy
            ):
                continue

            if any(
                math.hypot(pt.x - px, pt.y - py) < min_dist
                for px, py in placed_refs
            ):
                continue

            placed_refs.append((pt.x, pt.y))

            self.ax.text(
                pt.x,
                pt.y,
                ref.split(";")[0],
                color="white",
                fontsize=style.highway_font_size_pt,
                ha="center",
                va="center",
                zorder=style.highway_zorder,
                clip_on=True,
                fontfamily="sans-serif",
                fontweight="bold",
                bbox=bbox_style,
            )

        return self

    def draw_exits(self, min_distance_ratio=None):
        """Draw Japanese motorway junction and exit labels."""
        style = self.style.japan_labels

        min_distance_ratio = (
            style.exit_collision_ratio
            if min_distance_ratio is None
            else min_distance_ratio
        )

        if self.gdf_exits is None:
            self.gdf_exits = self._fetch_features(
                {"highway": "motorway_junction"},
                layer_name="jp_exits",
            )

        if self.gdf_exits is None or self.gdf_exits.empty:
            return self

        placed_exits = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        bbox_style = dict(
            facecolor=style.exit_facecolor,
            alpha=0.9,
            edgecolor=style.exit_edgecolor,
            linewidth=style.exit_linewidth_pt,
            boxstyle="square,pad=0.2",
        )

        for _, row in self.gdf_exits.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            pt = (
                geom.centroid
                if geom.geom_type in ["Polygon", "MultiPolygon"]
                else geom
            )

            if not (
                self.minx <= pt.x <= self.maxx
                and self.miny <= pt.y <= self.maxy
            ):
                continue

            name = str(row.get("name", "")).strip()
            ref = str(row.get("ref", "")).strip()

            valid_name = bool(name and name.lower() != "nan")
            valid_ref = bool(ref and ref.lower() != "nan")

            if not (valid_name or valid_ref):
                continue

            if any(
                math.hypot(pt.x - px, pt.y - py) < min_dist
                for px, py in placed_exits
            ):
                continue

            placed_exits.append((pt.x, pt.y))

            label_text = (
                f"{ref if valid_ref else ''} {name if valid_name else ''}"
            ).strip()

            self.ax.text(
                pt.x,
                pt.y,
                label_text,
                color="#fef08a",
                fontsize=style.exit_font_size_pt,
                ha="left",
                va="center",
                zorder=style.exit_zorder,
                clip_on=True,
                fontfamily="sans-serif",
                fontweight="bold",
                bbox=bbox_style,
            )

        return self

    def draw_piers(self, min_distance_ratio=None):
        """Draw minimalist anchor icons for Japanese ferry facilities."""
        style = self.style.japan_labels

        min_distance_ratio = (
            style.pier_collision_ratio
            if min_distance_ratio is None
            else min_distance_ratio
        )

        if self.gdf_piers is None:
            self.gdf_piers = self._fetch_features(
                {"amenity": "ferry_terminal", "man_made": "pier"},
                layer_name="jp_piers",
            )

        if self.gdf_piers is None or self.gdf_piers.empty:
            return self

        placed_piers = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        for _, row in self.gdf_piers.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type in [
                "Polygon",
                "MultiPolygon",
            ] and not geom.contains(geom.centroid):
                pt = geom.representative_point()
            else:
                pt = geom.centroid

            if not (
                self.minx <= pt.x <= self.maxx
                and self.miny <= pt.y <= self.maxy
            ):
                continue

            if any(
                math.hypot(pt.x - px, pt.y - py) < min_dist
                for px, py in placed_piers
            ):
                continue

            placed_piers.append((pt.x, pt.y))

            self.ax.text(
                pt.x,
                pt.y,
                "⚓",
                color=style.pier_color,
                fontsize=style.pier_font_size_pt,
                ha="center",
                va="center",
                zorder=style.pier_zorder,
                clip_on=True,
                fontfamily="DejaVu Sans",
            )

        return self
