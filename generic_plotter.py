"""
generic_plotter.py

Generic geographic/map rendering engine (v2 Architecture).

Includes:
    * Strict Unit Contract (CRS units, physical metres, visual points, scale)
    * RenderContext with WGS84 geodesic ground metrics
    * Local metric CRS estimation for safe physical buffering
    * Two-tier application caching (OSMDataCache)
    * Download auditing (rows, vertices, memory footprint, span ratio)
    * Automatic spatial clipping (prevents oversized OSM geometries)
    * Viewport locking and margin protection for titles and grid labels
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import CRS, Geod, Transformer
from shapely.geometry import box

from map_styles import MapStyle
from osm_cache import OSMDataCache


logger = logging.getLogger("GenericPlotter")

# Keep OSMnx's HTTP cache enabled alongside OSMDataCache.
ox.settings.use_cache = True


@dataclass(frozen=True)
class RenderContext:
    """
    Physical information about the current rendering.
    All values are explicit.
    """

    ground_width_m: float
    ground_height_m: float

    axes_width_in: float
    axes_height_in: float

    scale_x: float
    scale_y: float

    map_units_per_ground_meter_x: float
    map_units_per_ground_meter_y: float

    dpi: float

    @property
    def representative_scale(self) -> float:
        """Return the geometric mean of horizontal and vertical scale."""
        return math.sqrt(max(self.scale_x, 1e-12) * max(self.scale_y, 1e-12))

    @property
    def axes_width_pt(self) -> float:
        """Actual physical axes width in Matplotlib points."""
        return self.axes_width_in * 72.0

    @property
    def axes_height_pt(self) -> float:
        """Actual physical axes height in Matplotlib points."""
        return self.axes_height_in * 72.0


# ----------------------------------------------------------------------
# Unit / measurement utilities
# ----------------------------------------------------------------------

_GEOD = Geod(ellps="WGS84")


def _get_actual_ax_size_in(fig, ax) -> tuple[float, float]:
    """
    Return the actual physical axes dimensions in inches.
    Measures the AXES rather than the whole figure.
    """
    if fig.canvas is not None:
        fig.canvas.draw()

    bbox = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )

    return bbox.width, bbox.height


def _compute_ground_metrics(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    crs,
) -> tuple[float, float]:
    """
    Calculate horizontal/vertical ground dimensions in metres.
    Uses WGS84 ellipsoid geodesic distances for geographic CRS.
    """
    crs_obj = CRS.from_user_input(crs)

    if crs_obj.is_geographic:
        mid_lat = (miny + maxy) / 2.0
        mid_lon = (minx + maxx) / 2.0

        _, _, ground_w_m = _GEOD.inv(minx, mid_lat, maxx, mid_lat)
        _, _, ground_h_m = _GEOD.inv(mid_lon, miny, mid_lon, maxy)

        return abs(ground_w_m), abs(ground_h_m)

    return abs(maxx - minx), abs(maxy - miny)


def _is_metric_crs(crs) -> bool:
    """Return True when the first CRS axis uses metres."""
    crs_obj = CRS.from_user_input(crs)

    if not crs_obj.is_projected or not crs_obj.axis_info:
        return False

    unit_name = crs_obj.axis_info[0].unit_name.lower()

    return unit_name in {"metre", "meter", "metres", "meters"}


def _adaptive_width(
    base_width_pt: float,
    render_context: RenderContext | None,
    adaptive: bool,
    reference_scale: float,
    exponent: float,
    minimum_pt: float,
    maximum_pt: float,
) -> float:
    """Convert a base point width into its final scale-adjusted rendering width."""
    if not adaptive or render_context is None:
        return base_width_pt

    actual_scale = max(render_context.representative_scale, 1.0)
    factor = (reference_scale / actual_scale) ** exponent

    return max(minimum_pt, min(maximum_pt, base_width_pt * factor))


def _count_geometry_vertices(geom) -> int:
    """Safely count total coordinate vertices across any Shapely geometry type."""
    if geom is None or geom.is_empty:
        return 0

    gtype = geom.geom_type
    if gtype == "Point":
        return 1
    if gtype in ("LineString", "LinearRing"):
        return len(geom.coords)
    if gtype == "Polygon":
        interior_pts = sum(len(ring.coords) for ring in geom.interiors)
        return len(geom.exterior.coords) + interior_pts
    if hasattr(geom, "geoms"):
        return sum(_count_geometry_vertices(sub) for sub in geom.geoms)

    return 0


# ----------------------------------------------------------------------
# Layout decoration
# ----------------------------------------------------------------------

def decorate_map_layout(
    fig,
    ax,
    plotter: GenericMapPlotter,
    title_text: str = "Urban Map",
):
    """
    Decorate the map with title and calculated physical dimensions
    derived directly from the plotter's RenderContext.
    """
    logger.info("Decorating map layout with title and dimension details...")

    context = plotter.render_context
    if context is None:
        raise ValueError(
            "Render context is unavailable. Call setup_canvas() first."
        )

    ax_width_cm = context.axes_width_in * 2.54
    ax_height_cm = context.axes_height_in * 2.54

    full_title = (
        f"{title_text}\n"
        f"Dimensions: {ax_width_cm:.1f} cm × {ax_height_cm:.1f} cm  |  "
        f"Scale H: 1:{int(context.scale_x):,}  |  "
        f"Scale V: 1:{int(context.scale_y):,}"
    )

    ax.set_title(
        full_title,
        color="white",
        fontsize=14,
        fontfamily="monospace",
        fontweight="bold",
        pad=18,
        ha="center",
    )

    logger.info(
        "Layout decorated successfully. Size: %.1fx%.1f cm",
        ax_width_cm,
        ax_height_cm,
    )


# ======================================================================
# GenericMapPlotter
# ======================================================================

class GenericMapPlotter:
    """
    Generic physical geography and map foundation plotting class.
    Country-neutral core engine.
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
        projection="wgs84",
        style: MapStyle | None = None,
        cache: OSMDataCache | None = None,
        cache_enabled=True,
        cache_dir=None,
        cache_ttl_days=30,
        clip_features=True,
        clip_buffer_ratio=0.05,
        query_padding=0.08
    ):
        logger.info("Initializing GenericMapPlotter...")

        if (bbox is None) == (center_point is None):
            raise ValueError(
                "You must provide either 'bbox' OR 'center_point'."
            )

        self.bbox = bbox
        self.center_point = center_point
        self.dist = dist

        if self.center_point:
            self._check_center_point()

        if self.bbox:
            self._check_bbox()

        self.style = style or MapStyle()

        # Backward-compatible explicit overrides
        if bg_color is not None:
            self.style.base.background = bg_color
            self.style.base.water = bg_color

        if land_color is not None:
            self.style.base.land = land_color

        if custom_linewidths:
            self.style.roads.widths_pt.update(pd.Series(custom_linewidths))

        if custom_colors:
            self.style.roads.colors.update(pd.Series(custom_colors))

        self.bg_color = self.style.base.background
        self.land_color = self.style.base.land

        self.projection = projection.lower()
        self.crs = "EPSG:4326"
        self.metric_crs = None

        # Spatial clipping configuration
        self.clip_features = bool(clip_features)
        self.clip_buffer_ratio = float(clip_buffer_ratio)

        self.G = None

        self.gdf_admin_base = None
        self.gdf_water = None
        self.gdf_nodes = None
        self.gdf_edges = None
        self.gdf_greenery = None
        self.gdf_buildings = None
        self.gdf_aeroway = None

        self.fig = None
        self.ax = None
        self._is_user_ax = False

        self.minx = self.miny = self.maxx = self.maxy = 0.0
        self.render_context: RenderContext | None = None

        self.cache = cache or OSMDataCache(
            cache_dir=cache_dir,
            ttl_days=cache_ttl_days,
            enabled=cache_enabled,
        )
        
        self.grid_cells = None

        mode_str = (
            f"BBox mode ({self.bbox})"
            if self.bbox
            else f"Center Point mode ({self.center_point}, dist: {self.dist}m)"
        )

        logger.info(
            "GenericMapPlotter initialized with %s | Projection: %s",
            mode_str,
            self.projection.upper(),
        )
        self.target_bbox = bbox 
        self.query_bbox = self._calculate_query_bbox(bbox, query_padding) if bbox else None
        
        logger.info("Query Bbox is %s.", self.query_bbox)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_center_point(self):
        lat, lon = self.center_point
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid center point.")

    def _check_bbox(self):
        west, south, east, north = self.bbox
        if not (
            -180 <= west <= 180
            and -180 <= east <= 180
            and -90 <= south <= 90
            and -90 <= north <= 90
        ):
            raise ValueError("Invalid BBox.")

        if west >= east or south >= north:
            raise ValueError("Invalid BBox range.")

    # ------------------------------------------------------------------
    # CRS helpers
    # ------------------------------------------------------------------

    def _project(self, gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
        """Project a GeoDataFrame into the active rendering CRS."""
        if gdf is not None and not gdf.empty and self.crs != "EPSG:4326":
            return gdf.to_crs(self.crs)
        return gdf

    def _determine_metric_crs(self):
        """Determine a CRS suitable for physical measurements in metres."""
        if _is_metric_crs(self.crs):
            self.metric_crs = self.crs
            return self.metric_crs

        if self.gdf_nodes is None or self.gdf_nodes.empty:
            raise ValueError(
                "Cannot determine metric CRS without network geometry."
            )

        nodes_wgs84 = self.gdf_nodes.to_crs("EPSG:4326")
        estimated = nodes_wgs84.estimate_utm_crs()

        if estimated is None:
            raise ValueError("Unable to estimate a metric CRS for this map.")

        self.metric_crs = estimated
        logger.info(
            "Metric CRS for physical geometry operations: %s",
            self.metric_crs,
        )
        return self.metric_crs

    def _to_metric(self, gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
        """Return a GeoDataFrame projected into the physical metric CRS."""
        if gdf is None or gdf.empty:
            return gdf

        if self.metric_crs is None:
            self._determine_metric_crs()

        if CRS.from_user_input(gdf.crs) == CRS.from_user_input(self.metric_crs):
            return gdf.copy()

        return gdf.to_crs(self.metric_crs)

    def _from_metric(self, gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
        """Return metric geometry in the active rendering CRS."""
        if gdf is None or gdf.empty:
            return gdf

        return gdf.to_crs(self.crs)

    # ------------------------------------------------------------------
    # Auditing & Spatial Clipping
    # ------------------------------------------------------------------

    def _audit_geodataframe(
        self,
        gdf: gpd.GeoDataFrame | None,
        layer_name: str,
        elapsed_sec: float,
        source: str = "OSM",
    ):
        """
        Audit a GeoDataFrame: log row count, total vertices, memory usage,
        and spatial span ratio relative to the active map canvas.
        """
        if gdf is None or gdf.empty:
            logger.info(
                "[Audit:%s] %-24s | Empty | %.2fs",
                source,
                layer_name,
                elapsed_sec,
            )
            return

        mem_mb = gdf.memory_usage(deep=True).sum() / (1024.0 * 1024.0)
        total_vertices = int(
            np.sum([_count_geometry_vertices(g) for g in gdf.geometry])
        )

        canvas_w = max(self.maxx - self.minx, 1e-9)
        canvas_h = max(self.maxy - self.miny, 1e-9)

        if self.maxx > self.minx and self.maxy > self.miny:
            g_minx, g_miny, g_maxx, g_maxy = gdf.total_bounds
            span_x = (g_maxx - g_minx) / canvas_w
            span_y = (g_maxy - g_miny) / canvas_h
        else:
            span_x = span_y = 1.0

        logger.info(
            "[Audit:%s] %-24s | Rows: %5d | Vertices: %7d | "
            "Mem: %5.2f MB | Span: %4.1fx W, %4.1fx H | %.2fs",
            source,
            layer_name,
            len(gdf),
            total_vertices,
            mem_mb,
            span_x,
            span_y,
            elapsed_sec,
        )

        if span_x > 2.5 or span_y > 2.5:
            logger.warning(
                "Layer '%s' spans %.1fx the canvas extent! "
                "Spatial clipping will trim out-of-bounds vertices.",
                layer_name,
                max(span_x, span_y),
            )

    def _clip_to_canvas(
        self,
        gdf: gpd.GeoDataFrame | None,
        layer_name: str = "layer",
    ) -> gpd.GeoDataFrame | None:
        """
        Spatially clip oversized geometries to the active canvas bounding box
        (with a small margin buffer so boundary strokes do not terminate early).
        """
        if (
            not self.clip_features
            or gdf is None
            or gdf.empty
            or self.maxx <= self.minx
            or self.maxy <= self.miny
        ):
            return gdf

        dx = (self.maxx - self.minx) * self.clip_buffer_ratio
        dy = (self.maxy - self.miny) * self.clip_buffer_ratio

        clip_box = box(
            self.minx - dx,
            self.miny - dy,
            self.maxx + dx,
            self.maxy + dy,
        )

        try:
            cleaned = gdf.copy()
            # Repair self-intersecting OSM polygons before clipping
            cleaned["geometry"] = cleaned.geometry.make_valid()
            clipped = gpd.clip(cleaned, clip_box, keep_geom_type=True)
            clipped = clipped[
                clipped.geometry.notna() & ~clipped.geometry.is_empty
            ].copy()
            return clipped
        except Exception as exc:
            logger.warning(
                "keep_geom_type clip failed on '%s' (%s); retrying standard clip...",
                layer_name,
                exc,
            )
            try:
                clipped = gpd.clip(cleaned, clip_box, keep_geom_type=False)
                return clipped[
                    clipped.geometry.notna() & ~clipped.geometry.is_empty
                ].copy()
            except Exception as fallback_exc:
                logger.warning(
                    "Spatial clipping failed for '%s': %s. Returning unclipped GeoDataFrame.",
                    layer_name,
                    fallback_exc,
                )
                return gdf

    # ------------------------------------------------------------------
    # OSM cache keys
    # ------------------------------------------------------------------

    def _location_cache_payload(self) -> dict:
        """Describe the geographic request independently of styling."""
        if self.bbox is not None:
            return {
                "mode": "bbox",
                "bbox": tuple(float(v) for v in self.bbox),
            }

        return {
            "mode": "point",
            "center_point": tuple(float(v) for v in self.center_point),
            "dist_m": float(self.dist),
        }

    # ------------------------------------------------------------------
    # Graph acquisition & preparation
    # ------------------------------------------------------------------

    def _fetch_drive_graph(self, custom_filter=None):
        """Fetch the road graph, checking persistent application cache first."""
        payload = {
            **self._location_cache_payload(),
            "network_type": "drive",
            "custom_filter": custom_filter,
            "osmnx_version": getattr(ox, "__version__", "unknown"),
        }

        graph = self.cache.get("graph", payload)
        if graph is not None:
            return graph

        logger.info(
            "Road graph not present in application cache; requesting OSM data..."
        )

        if self.query_bbox:
            graph = ox.graph_from_bbox(
                bbox=self.query_bbox,
                network_type="drive",
                custom_filter=custom_filter,
            )
        else:
            graph = ox.graph_from_point(
                self.center_point,
                dist=self.dist,
                network_type="drive",
                custom_filter=custom_filter,
            )

        self.cache.set("graph", payload, graph)
        return graph

    def _project_graph(self, graph):
        """Project the graph into the requested rendering CRS."""
        if self.projection == "wgs84":
            self.crs = "EPSG:4326"
            return graph

        logger.debug("Projecting graph to target projection: %s", self.projection)

        if self.projection in {"utm", "transverse_mercator"}:
            graph = ox.project_graph(graph)
            self.crs = graph.graph["crs"]
        else:
            projection_map = {
                "conformal": "EPSG:3857",
                "mercator": "EPSG:3857",
                "equal_area": "+proj=cea",
                "equidistant": "+proj=eqc",
            }
            self.crs = projection_map.get(self.projection, "EPSG:3857")
            graph = ox.project_graph(graph, to_crs=self.crs)

        logger.info("Graph projected successfully. CRS=%s", self.crs)
        return graph

    def _prepare_network_geodata(self):
        """Convert the NetworkX graph into GeoPandas objects and audit edges."""
        t0 = time.perf_counter()
        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)

        if self.gdf_nodes.empty:
            raise ValueError("Invalid graph bounds: No road nodes found.")

        self._process_edges()
        elapsed = time.perf_counter() - t0
        self._audit_geodataframe(
            self.gdf_edges,
            layer_name="road_network_edges",
            elapsed_sec=elapsed,
            source="Graph",
        )

    def _calculate_bounds(self):
        """強制以使用者原始指定的 target_bbox 作為最終視角與邊界，不受外擴查詢影響。"""
        if self.target_bbox:
            w, s, e, n = self.target_bbox
            # 如果有做投影轉換，需要把 WGS84 的 target_bbox 轉成目標投影坐標系
            if self.projection != "wgs84" and self.crs != "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
                self.minx, self.miny = transformer.transform(w, s)
                self.maxx, self.maxy = transformer.transform(e, n)
            else:
                self.minx, self.miny, self.maxx, self.maxy = w, s, e, n
        else:
            # Center point 模式維持原本邏輯
            self.minx, self.miny, self.maxx, self.maxy = self.gdf_nodes.total_bounds

        logger.debug(
            "Strict viewport bounds established: (%.6f, %.6f, %.6f, %.6f)",
            self.minx, self.miny, self.maxx, self.maxy,
        )

    # ------------------------------------------------------------------
    # Road processing
    # ------------------------------------------------------------------

    def _process_edges(self):
        """Process OSM road attributes using vectorized Pandas operations."""
        logger.info("Processing road network edges using Pandas...")

        hw_series = self.gdf_edges["highway"].apply(
            lambda value: value[0] if isinstance(value, list) else value
        )

        road_style = self.style.roads

        self.gdf_edges["custom_width_pt"] = (
            hw_series.map(road_style.widths_pt).fillna(road_style.fallback_width_pt)
        )
        self.gdf_edges["custom_color"] = (
            hw_series.map(road_style.colors).fillna(road_style.fallback_color)
        )
        self.gdf_edges["custom_zorder"] = road_style.ground_zorder

        def is_yes(column):
            if column not in self.gdf_edges.columns:
                return pd.Series(False, index=self.gdf_edges.index)
            return self.gdf_edges[column].apply(
                lambda value: "yes" in value if isinstance(value, list) else value == "yes"
            )

        is_tunnel = is_yes("tunnel")
        is_bridge = is_yes("bridge")

        self.gdf_edges.loc[is_tunnel, "custom_zorder"] = road_style.tunnel_zorder
        self.gdf_edges.loc[is_tunnel, "custom_color"] = road_style.tunnel_color

        self.gdf_edges.loc[is_bridge, "custom_zorder"] = road_style.bridge_zorder
        self.gdf_edges.loc[is_bridge, "custom_color"] = road_style.bridge_color

        logger.info("Successfully processed %d road edges.", len(self.gdf_edges))

    # ------------------------------------------------------------------
    # Generic feature retrieval (with Cache + Audit + Spatial Clipping)
    # ------------------------------------------------------------------

    def _fetch_features(self, tags, layer_name: str) -> gpd.GeoDataFrame | None:
        """
        Fetch an OSM feature layer with application-level caching,
        download auditing, automatic spatial clipping, and empty-set protection.
        """
        t0 = time.perf_counter()
        
        # 將原本的 self.bbox 改為 self.query_bbox
        

        payload = {
            **self._location_cache_payload(),
            "layer": layer_name,
            "tags": tags,
            "osmnx_version": getattr(ox, "__version__", "unknown"),
        }

        raw = self.cache.get("features", payload)

        if raw is not None:
            # 🌟 防禦：如果快取裡記錄的是空集合 (無 elements)，直接返回 None，絕不觸發 osmnx 解析
            if isinstance(raw, gpd.GeoDataFrame) and raw.empty:
                logger.info("[Cache] Layer '%s' is a known empty set. Skipping.", layer_name)
                return None

            projected = self._project(raw.copy())
            elapsed = time.perf_counter() - t0
            self._audit_geodataframe(projected, layer_name, elapsed, source="Cache")
            return self._clip_to_canvas(projected, layer_name=layer_name)

        logger.info("Fetching '%s' features from OSM...", layer_name)

        try:
            if self.query_bbox:
                raw = ox.features_from_bbox(bbox=self.query_bbox, tags=tags)
            else:
                raw = ox.features_from_point(
                    self.center_point, tags=tags, dist=self.dist
                )
                
            logger.info("Crawled '%s' features from OSM...", layer_name)

            # 🌟 防禦：確保拿到的是有效 GeoDataFrame，若空則主動標記
            if raw is None or raw.empty:
                logger.info("OSM returned an empty set for '%s'. Caching empty result.", layer_name)
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                self.cache.set("features", payload, empty_gdf)
                return None

            self.cache.set("features", payload, raw)

            projected = self._project(raw.copy())
            elapsed = time.perf_counter() - t0
            self._audit_geodataframe(projected, layer_name, elapsed, source="OSM")
            return self._clip_to_canvas(projected, layer_name=layer_name)
        
        except ox._errors.InsufficientResponseError:
            return None

        except Exception as exc:
            err_msg = str(exc).lower()
            no_data_keywords = [
                "no data",
                "empty",
                "insufficient",
                "not find",
                "no features",
                "insufficientresponseerror"
            ]

            if any(keyword in err_msg for keyword in no_data_keywords):
                logger.info("No features found for '%s' (Caught exception). Safely ignoring.", layer_name)
                # 將空集合寫入快取，下次直接避開
                try:
                    self.cache.set("features", payload, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
                except Exception:
                    pass
                return None

            logger.error("Fatal error while fetching '%s': %s", layer_name, exc)
            raise
        
    def _calculate_query_bbox(self, bbox, padding: float):
        """將使用者 BBox 向外擴展指定的比例（例如 8%），用於向 API 抓取更寬鬆的周邊資料。"""
        west, south, east, north = bbox
        w_span = east - west
        h_span = north - south
        
        dx = w_span * padding
        dy = h_span * padding
        
        return (west - dx, south - dy, east + dx, north + dy)

    # ------------------------------------------------------------------
    # Canvas construction
    # ------------------------------------------------------------------

    def _create_figure(self, base_width):
        """
        Create the Matplotlib figure (base_width in INCHES).
        Reserves breathing room around the axes so titles and grid
        labels are never clipped off the canvas.
        """
        ground_w_m, ground_h_m = _compute_ground_metrics(
            self.minx,
            self.miny,
            self.maxx,
            self.maxy,
            self.crs,
        )

        if ground_w_m <= 0 or ground_h_m <= 0:
            raise ValueError("Map bounds have zero ground extent.")

        aspect_ratio = ground_h_m / ground_w_m
        fig_height = base_width * aspect_ratio

        self.fig, self.ax = plt.subplots(figsize=(base_width, fig_height))

        # Leave margins so top titles and right/top grid labels stay inside the figure
        self.fig.subplots_adjust(
            left=0.02,
            right=0.95,
            bottom=0.02,
            top=0.90,
        )

        self._is_user_ax = False

        logger.info(
            "New canvas initialized: %.2f x %.2f inches.",
            base_width,
            fig_height,
        )

    def _attach_user_axes(self, ax):
        """Use an externally supplied Matplotlib Axes."""
        self.ax = ax
        self.fig = ax.figure
        self._is_user_ax = True

    def _configure_axes(self):
        """Apply map bounds, zero margins, background, and aspect."""
        self.ax.set_xlim([self.minx, self.maxx])
        self.ax.set_ylim([self.miny, self.maxy])
        self.ax.margins(0)
        self.ax.set_facecolor(self.bg_color)

        if self.crs == "EPSG:4326":
            center_lat = (self.miny + self.maxy) / 2.0
            self.ax.set_aspect(1.0 / math.cos(math.radians(center_lat)))
        else:
            self.ax.set_aspect("equal")

    def _build_render_context(self):
        """Measure the actual rendered axes and create RenderContext."""
        axes_width_in, axes_height_in = _get_actual_ax_size_in(
            self.fig, self.ax
        )

        ground_width_m, ground_height_m = _compute_ground_metrics(
            self.minx,
            self.miny,
            self.maxx,
            self.maxy,
            self.crs,
        )

        scale_x = ground_width_m / (axes_width_in * 0.0254)
        scale_y = ground_height_m / (axes_height_in * 0.0254)

        map_span_x = self.maxx - self.minx
        map_span_y = self.maxy - self.miny

        units_per_meter_x = (
            map_span_x / ground_width_m if ground_width_m else 0.0
        )
        units_per_meter_y = (
            map_span_y / ground_height_m if ground_height_m else 0.0
        )

        self.render_context = RenderContext(
            ground_width_m=ground_width_m,
            ground_height_m=ground_height_m,
            axes_width_in=axes_width_in,
            axes_height_in=axes_height_in,
            scale_x=scale_x,
            scale_y=scale_y,
            map_units_per_ground_meter_x=units_per_meter_x,
            map_units_per_ground_meter_y=units_per_meter_y,
            dpi=self.fig.dpi,
        )

        logger.info(
            "Render context: axes %.2f x %.2f in | scale H=1:%d V=1:%d",
            axes_width_in,
            axes_height_in,
            int(scale_x),
            int(scale_y),
        )

    def setup_canvas(
        self,
        ax=None,
        base_width=16,
        highway_filter=None,
    ):
        """Prepare the complete map canvas through modular setup stages."""
        logger.info("Setting up map canvas...")

        custom_filter = None
        if highway_filter:
            custom_filter = f'["highway"~"{"|".join(highway_filter)}"]'

        # 1. Network acquisition
        self.G = self._fetch_drive_graph(custom_filter=custom_filter)

        # 2. Projection
        self.G = self._project_graph(self.G)

        # 3. Bounds calculation first (so network audit knows canvas span)
        self.gdf_nodes = ox.graph_to_gdfs(self.G, edges=False)
        if self.gdf_nodes.empty:
            raise ValueError("Invalid graph bounds: No road nodes found.")
        self._calculate_bounds()

        # 4. Graph -> GeoDataFrames & Edge Styling
        self._prepare_network_geodata()

        # 5/6. Canvas creation
        if ax is None:
            self._create_figure(base_width)
        else:
            self._attach_user_axes(ax)

        # 7. Axes configuration
        self._configure_axes()

        # 8. Metric CRS for physical geometry
        self._determine_metric_crs()

        # 9. Actual physical rendering context
        self._build_render_context()

        logger.info("Canvas setup complete.")
        return self

    # ------------------------------------------------------------------
    # Style / linewidth helpers
    # ------------------------------------------------------------------

    def _get_stroke_width(self, base_width_pt: float) -> float:
        """Return the final stroke width in POINTS."""
        roads = self.style.roads
        return _adaptive_width(
            base_width_pt=base_width_pt,
            render_context=self.render_context,
            adaptive=roads.adaptive_widths,
            reference_scale=roads.reference_scale,
            exponent=roads.width_scale_exponent,
            minimum_pt=roads.minimum_width_pt,
            maximum_pt=roads.maximum_width_pt,
        )

    # ------------------------------------------------------------------
    # Generic layers
    # ------------------------------------------------------------------

    def draw_land(self):
        """Draw the map's base land rectangle."""
        logger.info("Drawing base land layer...")

        if self.gdf_admin_base is None:
            poly = box(self.minx, self.miny, self.maxx, self.maxy)
            self.gdf_admin_base = gpd.GeoDataFrame(
                geometry=[poly], crs=self.crs
            )

        if not self.gdf_admin_base.empty:
            self.gdf_admin_base.plot(
                ax=self.ax,
                color=self.style.base.land,
                zorder=self.style.base.land_zorder,
            )

        return self

    def draw_water(self):
        """Retrieve and draw OSM water features."""
        logger.info("Drawing water features...")

        if self.gdf_water is None:
            tags = {
                "natural": ["water", "bay", "coastline"],
                "waterway": ["river", "riverbank", "canal", "dock"],
                "water": True,
            }
            self.gdf_water = self._fetch_features(tags, layer_name="water")

        if self.gdf_water is not None and not self.gdf_water.empty:
            self.gdf_water.plot(
                ax=self.ax,
                color=self.style.base.water,
                edgecolor="none",
                zorder=self.style.base.water_zorder,
            )

        return self

    def draw_greenery(self):
        """Retrieve and draw generic greenery polygons."""
        logger.info("Drawing greenery features...")

        if self.gdf_greenery is None:
            tags = {
                "leisure": ["park", "pitch", "garden", "golf_course"],
                "landuse": ["forest", "grass", "cemetery", "recreation_ground"],
                "natural": ["wood", "scrub"],
            }
            self.gdf_greenery = self._fetch_features(
                tags, layer_name="greenery"
            )

        if self.gdf_greenery is not None and not self.gdf_greenery.empty:
            poly_gdf = self.gdf_greenery[
                self.gdf_greenery.geometry.type.isin(["Polygon", "MultiPolygon"])
            ]

            if not poly_gdf.empty:
                poly_gdf.plot(
                    ax=self.ax,
                    color=self.style.base.greenery,
                    alpha=self.style.base.greenery_alpha,
                    edgecolor="none",
                    zorder=self.style.base.greenery_zorder,
                )

        return self

    def draw_buildings(self):
        """Retrieve and draw generic building footprints."""
        logger.info("Drawing building footprints...")

        if self.gdf_buildings is None:
            self.gdf_buildings = self._fetch_features(
                {"building": True}, layer_name="buildings"
            )

        if self.gdf_buildings is None or self.gdf_buildings.empty:
            return self

        poly_gdf = self.gdf_buildings[
            self.gdf_buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
        ]

        if poly_gdf.empty:
            return self

        is_commercial = poly_gdf["building"].isin(
            ["commercial", "retail", "office", "hotel", "supermarket"]
        )

        bldg_comm = poly_gdf[is_commercial]
        bldg_resi = poly_gdf[~is_commercial]

        if not bldg_resi.empty:
            bldg_resi.plot(
                ax=self.ax,
                color=self.style.base.building_residential,
                alpha=self.style.base.building_residential_alpha,
                edgecolor="none",
                zorder=self.style.base.residential_building_zorder,
            )

        if not bldg_comm.empty:
            bldg_comm.plot(
                ax=self.ax,
                color=self.style.base.building_commercial,
                alpha=self.style.base.building_commercial_alpha,
                edgecolor="none",
                zorder=self.style.base.commercial_building_zorder,
            )

        return self

    def draw_roads(self):
        """Draw the processed road network (linewidth in POINTS)."""
        logger.info("Drawing road network...")

        if self.gdf_edges is None:
            raise ValueError(
                "Road network is not prepared. Call setup_canvas() first."
            )

        tunnels = self.gdf_edges[
            self.gdf_edges["custom_zorder"] == self.style.roads.tunnel_zorder
        ]
        ground = self.gdf_edges[
            self.gdf_edges["custom_zorder"] == self.style.roads.ground_zorder
        ]
        bridges = self.gdf_edges[
            self.gdf_edges["custom_zorder"] == self.style.roads.bridge_zorder
        ]

        def render_group(gdf, zorder, linestyle="-"):
            if gdf.empty:
                return

            linewidths = gdf["custom_width_pt"].map(self._get_stroke_width)
            gdf.plot(
                ax=self.ax,
                color=gdf["custom_color"],
                linewidth=linewidths,
                zorder=zorder,
                linestyle=linestyle,
            )

        render_group(tunnels, self.style.roads.tunnel_zorder, "--")
        render_group(ground, self.style.roads.ground_zorder)
        render_group(bridges, self.style.roads.bridge_zorder)

        logger.info("Road network rendered.")
        return self

    # ------------------------------------------------------------------
    # Airport rendering
    # ------------------------------------------------------------------

    def _parse_width_m(self, value, default_width_m: float) -> float:
        """Parse an OSM width value into physical metres."""
        if pd.isna(value):
            return default_width_m

        text = str(value).strip()
        if not text:
            return default_width_m

        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if not match:
            return default_width_m

        try:
            parsed = float(match.group(0))
        except ValueError:
            return default_width_m

        return parsed if parsed > 0 else default_width_m

    def _prepare_airport_surface_geometries(self, poly_gdf, line_gdf):
        """Convert airport linear features into physical surface polygons in metric_crs."""
        airport_style = self.style.airports
        final_surfaces = {"runway": [], "taxiway": []}

        if not poly_gdf.empty:
            metric_polys = self._to_metric(poly_gdf)
            for _, row in metric_polys.iterrows():
                aeroway = row.get("aeroway")
                if isinstance(aeroway, list):
                    aeroway = aeroway[0] if aeroway else ""
                if aeroway in final_surfaces:
                    final_surfaces[aeroway].append(row.geometry)

        if not line_gdf.empty:
            metric_lines = self._to_metric(line_gdf)
            for _, metric_row in metric_lines.iterrows():
                # 🌟 直接從當前迭代的 metric_row 取得 aeroway，避免 index 重複導致 .loc 回傳 Series
                aeroway = metric_row.get("aeroway")
                if isinstance(aeroway, list):
                    aeroway = aeroway[0] if aeroway else ""

                if aeroway not in final_surfaces:
                    continue

                geom = metric_row.geometry
                if geom is None or geom.is_empty:
                    continue

                default_width = (
                    airport_style.default_runway_width_m
                    if aeroway == "runway"
                    else airport_style.default_taxiway_width_m
                )

                # 🌟 同理，寬度也直接從 metric_row 取得
                width_raw = metric_row.get("width")
                width_m = self._parse_width_m(width_raw, default_width)

                final_surfaces[aeroway].append(geom.buffer(width_m / 2.0))

        return final_surfaces

    def draw_airports(self):
        """Draw airport physical surfaces (metres) and centerlines (points)."""
        logger.info("Drawing airport features...")

        if self.gdf_aeroway is None:
            self.gdf_aeroway = self._fetch_features(
                {"aeroway": ["runway", "taxiway", "apron", "helipad"]},
                layer_name="airports",
            )

        if self.gdf_aeroway is None or self.gdf_aeroway.empty:
            return self

        airport_style = self.style.airports

        poly_gdf = self.gdf_aeroway[
            self.gdf_aeroway.geometry.type.isin(["Polygon", "MultiPolygon"])
        ]
        line_gdf = self.gdf_aeroway[
            ~self.gdf_aeroway.geometry.type.isin(["Polygon", "MultiPolygon"])
        ]

        if not poly_gdf.empty:
            aprons = poly_gdf[poly_gdf["aeroway"] == "apron"]
            if not aprons.empty:
                aprons.plot(
                    ax=self.ax,
                    color=airport_style.apron_color,
                    zorder=airport_style.apron_zorder,
                    edgecolor="none",
                )

        surfaces = self._prepare_airport_surface_geometries(poly_gdf, line_gdf)

        if surfaces["taxiway"]:
            metric_gdf = gpd.GeoDataFrame(
                geometry=surfaces["taxiway"], crs=self.metric_crs
            )
            metric_gdf.to_crs(self.crs).plot(
                ax=self.ax,
                color=airport_style.taxiway_surface_color,
                zorder=airport_style.taxiway_zorder,
                edgecolor="none",
            )

        if surfaces["runway"]:
            metric_gdf = gpd.GeoDataFrame(
                geometry=surfaces["runway"], crs=self.metric_crs
            )
            metric_gdf.to_crs(self.crs).plot(
                ax=self.ax,
                color=airport_style.runway_surface_color,
                zorder=airport_style.runway_zorder,
                edgecolor="none",
            )

        if not line_gdf.empty:
            taxiways = line_gdf[line_gdf["aeroway"] == "taxiway"]
            runways = line_gdf[line_gdf["aeroway"] == "runway"]

            if not taxiways.empty:
                taxiways.plot(
                    ax=self.ax,
                    color=airport_style.taxiway_centerline_color,
                    linewidth=self._get_stroke_width(
                        airport_style.taxiway_centerline_width_pt
                    ),
                    zorder=airport_style.taxiway_centerline_zorder,
                )

            if not runways.empty:
                runways.plot(
                    ax=self.ax,
                    color=airport_style.runway_centerline_color,
                    linewidth=self._get_stroke_width(
                        airport_style.runway_centerline_width_pt
                    ),
                    linestyle=(0, (10, 10)),
                    zorder=airport_style.runway_centerline_zorder,
                )

            placed_markings = []

            for _, row in line_gdf.iterrows():
                ref = row.get("ref")
                if pd.isna(ref) or not str(ref).strip():
                    continue

                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                center_pt = geom.representative_point()
                x, y = center_pt.x, center_pt.y

                if not (
                    self.minx <= x <= self.maxx and self.miny <= y <= self.maxy
                ):
                    continue

                coords = (
                    list(geom.coords)
                    if geom.geom_type == "LineString"
                    else (
                        list(geom.geoms[0].coords)
                        if geom.geom_type == "MultiLineString"
                        else []
                    )
                )

                if len(coords) < 2:
                    continue

                angle = math.degrees(
                    math.atan2(
                        coords[-1][1] - coords[0][1],
                        coords[-1][0] - coords[0][0],
                    )
                )
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                is_runway = row.get("aeroway") == "runway"
                fontsize = (
                    airport_style.runway_label_font_size_pt
                    if is_runway
                    else airport_style.taxiway_label_font_size_pt
                )
                fontcolor = (
                    airport_style.runway_centerline_color
                    if is_runway
                    else airport_style.taxiway_centerline_color
                )

                if any(
                    math.hypot(x - px, y - py)
                    < (self.maxx - self.minx)
                    * airport_style.label_collision_ratio
                    for px, py in placed_markings
                ):
                    continue

                placed_markings.append((x, y))

                self.ax.text(
                    x,
                    y,
                    str(ref).strip(),
                    color=fontcolor,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    rotation=angle,
                    zorder=airport_style.label_zorder,
                    clip_on=True,
                    fontfamily="sans-serif",
                    fontweight="bold",
                )

        logger.info("Airport rendering complete.")
        return self
    
    def draw_landuse(self):
        """
        Retrieve and draw OSM landuse polygons (industrial, commercial, 
        residential, construction, etc.) with proper z-ordering.
        """
        logger.info("Drawing landuse features...")

        if getattr(self, "gdf_landuse", None) is None:
            tags = {
                "landuse": [
                    "industrial",
                    "commercial",
                    "retail",
                    "residential",
                    "construction",
                    "military",
                    "quarry",
                    "cemetery",
                    "brownfield",
                    "landfill",
                ]
            }
            self.gdf_landuse = self._fetch_features(tags, layer_name="landuse")

        if self.gdf_landuse is not None and not self.gdf_landuse.empty:
            poly_gdf = self.gdf_landuse[
                self.gdf_landuse.geometry.type.isin(["Polygon", "MultiPolygon"])
            ].copy()

            if not poly_gdf.empty:
                style = self.style.landuse
                
                # Vectorized mapping of landuse types to style colors
                lu_series = poly_gdf["landuse"].apply(
                    lambda v: v[0] if isinstance(v, list) else v
                )
                poly_gdf["render_color"] = lu_series.map(style.colors).fillna(style.default_color)

                poly_gdf.plot(
                    ax=self.ax,
                    color=poly_gdf["render_color"],
                    alpha=style.alpha,
                    edgecolor="none",
                    zorder=style.zorder,
                )

        return self

    # ------------------------------------------------------------------
    # Grid / coordinates / scale
    # ------------------------------------------------------------------

    def draw_grid(
        self,
        rows=5,
        cols=5,
        color=None,
        alpha=None,
        linewidth=None,
    ):
        """
        Draw the alphanumeric map grid (linewidth in POINTS).
        Supports Excel-style column labeling (A-Z, then AA, AB, AC...) for large maps.
        """
        if self.ax is None:
            raise ValueError("Canvas not found.")

        style = self.style.grid

        color = color if color is not None else style.color
        alpha = alpha if alpha is not None else style.alpha
        linewidth = linewidth if linewidth is not None else style.linewidth_pt

        self.grid_cells = {}

        x_steps = [
            self.minx + i * (self.maxx - self.minx) / cols
            for i in range(cols + 1)
        ]
        y_steps = [
            self.miny + j * (self.maxy - self.miny) / rows
            for j in range(rows + 1)
        ]

        # 🌟 Excel-style column label generator (A-Z, AA, AB, AC...)
        def _get_col_label(n: int) -> str:
            res = []
            while n >= 0:
                res.append(chr(65 + n % 26))
                n = n // 26 - 1
            return "".join(reversed(res))

        cols_labels = [_get_col_label(i) for i in range(cols)]

        for i, x in enumerate(x_steps):
            self.ax.plot(
                [x, x],
                [self.miny, self.maxy],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle="--",
                zorder=style.zorder,
            )

            if i < cols:
                self.ax.text(
                    x + (self.maxx - self.minx) / (cols * 2.0),
                    self.maxy,
                    cols_labels[i],
                    color=color,
                    alpha=0.85,
                    fontsize=style.font_size_pt,
                    ha="center",
                    va="bottom",
                    zorder=style.zorder,
                    fontfamily="monospace",
                    fontweight="bold",
                )

        for j, y in enumerate(y_steps):
            self.ax.plot(
                [self.minx, self.maxx],
                [y, y],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle="--",
                zorder=style.zorder,
            )

            if j < rows:
                self.ax.text(
                    self.maxx,
                    y + (self.maxy - self.miny) / (rows * 2.0),
                    str(rows - j),
                    color=color,
                    alpha=0.85,
                    fontsize=style.font_size_pt,
                    ha="left",
                    va="center",
                    zorder=style.zorder,
                    fontfamily="monospace",
                    fontweight="bold",
                )

        for i in range(cols):
            for j in range(rows):
                self.grid_cells[f"{cols_labels[i]}{rows - j}"] = (
                    x_steps[i],
                    y_steps[j],
                    x_steps[i + 1],
                    y_steps[j + 1],
                )

        return self
    
    def get_grid_cell_for_point(self, x: float, y: float) -> str | None:
        """
        Determine which grid cell (e.g., 'A3', 'C2') a given map coordinate (x, y) falls into.

        Parameters
        ----------
        x: float
            Map X coordinate (e.g., projected easting, UTM coordinate, or longitude).
        y: float
            Map Y coordinate (e.g., projected northing, UTM coordinate, or latitude).

        Returns
        -------
        str or None
            The grid cell label (e.g., 'B4') if the point lies within the map extent;
            otherwise, returns None.

        Example
        -------
        >>> cell = plotter.get_grid_cell_for_point(139.735, 35.635)
        >>> print(f"The landmark is located in grid cell: {cell}")
        """
        if not hasattr(self, "grid_cells") or not self.grid_cells:
            logger.warning(
                "Grid cells have not been indexed yet. Call draw_grid() first."
            )
            return None

        # Quick boundary check
        if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy):
            return None

        # Iterate through indexed grid cells to find containment
        for cell_label, (minx, miny, maxx, maxy) in self.grid_cells.items():
            if minx <= x <= maxx and miny <= y <= maxy:
                return cell_label

        return None
    
    def calculate_optimal_grid(self, min_cell_cm: float = 4.0) -> tuple[int, int]:
        """
        Automatically calculate the optimal number of grid rows and columns 
        based on the actual rendered canvas dimensions, keeping cells as close 
        to square as possible while ensuring the side length is not smaller 
        than the specified minimum (default 4 cm).

        Parameters
        ----------
        min_cell_cm : float
            The minimum desired side length for each grid cell in centimeters.

        Returns
        -------
        tuple[int, int]
            A tuple of (rows, cols) optimized for square-like cells.

        Example
        -------
        >>> rows, cols = plotter.calculate_optimal_grid(min_cell_cm=4.0)
        >>> plotter.draw_grid(rows=rows, cols=cols)
        """
        if self.render_context is None:
            raise ValueError(
                "Render context is unavailable. Call setup_canvas() first."
            )

        # Convert physical axes dimensions from inches to centimeters
        axes_w_cm = self.render_context.axes_width_in * 2.54
        axes_h_cm = self.render_context.axes_height_in * 2.54

        # Calculate maximum possible divisions while enforcing the minimum cell size
        max_cols = max(1, int(axes_w_cm // min_cell_cm))
        max_rows = max(1, int(axes_h_cm // min_cell_cm))

        if max_cols == 1 and max_rows == 1:
            return 1, 1

        # Match the canvas aspect ratio to keep grid cells as square as possible
        aspect_ratio = axes_w_cm / axes_h_cm

        if aspect_ratio >= 1.0:
            rows = max_rows
            cols = max(1, round(rows * aspect_ratio))
            # Ensure cell width does not drop below min_cell_cm
            while (axes_w_cm / cols) < min_cell_cm and cols > 1:
                cols -= 1
        else:
            cols = max_cols
            rows = max(1, round(cols / aspect_ratio))
            # Ensure cell height does not drop below min_cell_cm
            while (axes_h_cm / rows) < min_cell_cm and rows > 1:
                rows -= 1

        cell_w = axes_w_cm / cols
        cell_h = axes_h_cm / rows

        logger.info(
            "Optimal grid calculated: %d rows × %d cols | Cell dimensions approx: %.1f cm × %.1f cm",
            rows,
            cols,
            cell_w,
            cell_h,
        )

        return rows, cols

    def draw_coordinates(self):
        """Draw geographic DMS coordinate ticks along map borders."""
        if self.ax is None:
            raise ValueError("Canvas not found.")

        if self.crs != "EPSG:4326":
            to_wgs84 = Transformer.from_crs(
                self.crs, "EPSG:4326", always_xy=True
            )
            to_map = Transformer.from_crs(
                "EPSG:4326", self.crs, always_xy=True
            )
            w, s = to_wgs84.transform(self.minx, self.miny)
            e, n = to_wgs84.transform(self.maxx, self.maxy)
        else:
            w, s, e, n = self.minx, self.miny, self.maxx, self.maxy
            to_map = None

        span = min(e - w, n - s)
        interval = (
            1 / 120.0
            if span < 0.05
            else (
                1 / 60.0
                if span < 0.1
                else (5 / 60.0 if span < 0.5 else 0.1)
            )
        )

        def format_dms(value, is_lat):
            value_abs = abs(value)
            deg = int(value_abs)
            mins_float = (value_abs - deg) * 60
            mins = int(mins_float)
            secs = int(round((mins_float - mins) * 60))

            if secs >= 60:
                secs = 0
                mins += 1
            if mins >= 60:
                mins = 0
                deg += 1

            direction = (
                ("N" if value >= 0 else "S")
                if is_lat
                else ("E" if value >= 0 else "W")
            )

            if secs > 0:
                return f'{deg}°{mins}\'{secs}"{direction}'
            if mins > 0:
                return f"{deg}°{mins}'{direction}"
            return f"{deg}°{direction}"

        lon_ticks, lat_ticks = [], []
        curr_lon = math.ceil(w / interval) * interval
        while curr_lon <= e:
            lon_ticks.append(curr_lon)
            curr_lon += interval

        curr_lat = math.ceil(s / interval) * interval
        while curr_lat <= n:
            lat_ticks.append(curr_lat)
            curr_lat += interval

        style = self.style.coordinates
        halo = [
            pe.withStroke(
                linewidth=style.halo_width_pt,
                foreground="#091c15",
            )
        ]

        tick_len_x = (self.maxx - self.minx) * 0.005
        tick_len_y = (self.maxy - self.miny) * 0.005

        for lon in lon_ticks:
            px = max(
                self.minx,
                min(
                    to_map.transform(lon, s)[0] if to_map else lon,
                    self.maxx,
                ),
            )

            self.ax.plot(
                [px, px],
                [self.miny, self.miny + tick_len_y],
                color=style.color,
                linewidth=style.line_width_pt,
                zorder=style.zorder,
            )
            self.ax.plot(
                [px, px],
                [self.maxy, self.maxy - tick_len_y],
                color=style.color,
                linewidth=style.line_width_pt,
                zorder=style.zorder,
            )

            label = format_dms(lon, False)
            self.ax.text(
                px,
                self.miny + tick_len_y * 1.5,
                label,
                color=style.color,
                fontsize=style.font_size_pt,
                ha="center",
                va="bottom",
                fontfamily=style.font_family,
                zorder=style.text_zorder,
                fontweight="bold",
                path_effects=halo,
            )
            self.ax.text(
                px,
                self.maxy - tick_len_y * 1.5,
                label,
                color=style.color,
                fontsize=style.font_size_pt,
                ha="center",
                va="top",
                fontfamily=style.font_family,
                zorder=style.text_zorder,
                fontweight="bold",
                path_effects=halo,
            )

        for lat in lat_ticks:
            py = max(
                self.miny,
                min(
                    to_map.transform(w, lat)[1] if to_map else lat,
                    self.maxy,
                ),
            )

            self.ax.plot(
                [self.minx, self.minx + tick_len_x],
                [py, py],
                color=style.color,
                linewidth=style.line_width_pt,
                zorder=style.zorder,
            )
            self.ax.plot(
                [self.maxx, self.maxx - tick_len_x],
                [py, py],
                color=style.color,
                linewidth=style.line_width_pt,
                zorder=style.zorder,
            )

            label = format_dms(lat, True)
            self.ax.text(
                self.minx + tick_len_x * 1.5,
                py,
                label,
                color=style.color,
                fontsize=style.font_size_pt,
                ha="left",
                va="center",
                fontfamily=style.font_family,
                zorder=style.text_zorder,
                fontweight="bold",
                path_effects=halo,
            )
            self.ax.text(
                self.maxx - tick_len_x * 1.5,
                py,
                label,
                color=style.color,
                fontsize=style.font_size_pt,
                ha="right",
                va="center",
                fontfamily=style.font_family,
                zorder=style.text_zorder,
                fontweight="bold",
                path_effects=halo,
            )

        return self

    def draw_scale(
        self,
        x_offset=None,
        y_offset=None,
        color=None,
        font_size=None,
    ):
        """Draw horizontal/vertical scale bars and north arrow."""
        if self.ax is None or self.fig is None:
            raise ValueError("Canvas not found.")

        style = self.style.scale

        x_offset = style.x_offset_ratio if x_offset is None else x_offset
        y_offset = style.y_offset_ratio if y_offset is None else y_offset
        color = style.color if color is None else color
        font_size = style.font_size_pt if font_size is None else font_size

        context = self.render_context
        if context is None:
            raise ValueError("Render context is unavailable.")

        bar_length_x = (
            style.bar_ground_length_m * context.map_units_per_ground_meter_x
        )
        bar_length_y = (
            style.bar_ground_length_m * context.map_units_per_ground_meter_y
        )

        ax_x = self.maxx - (self.maxx - self.minx) * x_offset
        ax_y = self.miny + (self.maxy - self.miny) * y_offset

        sx = ax_x - bar_length_x
        sy = ax_y + bar_length_y

        tick_h = (self.maxy - self.miny) * 0.004
        tick_w = (self.maxx - self.minx) * 0.004

        thick_lw = style.thick_line_width_pt
        thin_lw = style.thin_line_width_pt
        z = style.zorder

        halo = [pe.withStroke(linewidth=2.5, foreground="#f8fafc")]

        self.ax.plot(
            [sx, ax_x],
            [ax_y, ax_y],
            color=color,
            linewidth=thick_lw,
            zorder=z,
        )

        for cx in [sx, sx + bar_length_x / 2.0, ax_x]:
            self.ax.plot(
                [cx, cx],
                [ax_y - tick_h, ax_y + tick_h],
                color=color,
                linewidth=thin_lw,
                zorder=z,
            )

        half_ground = style.bar_ground_length_m / 2.0

        for tx, value in zip(
            [sx, sx + bar_length_x / 2.0, ax_x],
            ["0", f"{half_ground:.0f}m", f"{style.bar_ground_length_m:.0f}m"],
        ):
            self.ax.text(
                tx,
                ax_y - tick_h * 1.8,
                value,
                color=color,
                fontsize=font_size,
                ha="center",
                va="top",
                zorder=z,
                fontfamily="monospace",
                fontweight="bold",
                path_effects=halo,
            )

        self.ax.text(
            ax_x - bar_length_x / 2.0,
            ax_y + tick_h * 2.5,
            f"H Scale: 1:{int(context.scale_x):,}",
            color=color,
            fontsize=font_size,
            ha="center",
            va="bottom",
            zorder=z,
            fontfamily="monospace",
            fontweight="bold",
            path_effects=halo,
        )

        self.ax.plot(
            [ax_x, ax_x],
            [ax_y, sy],
            color=color,
            linewidth=thick_lw,
            zorder=z,
        )

        for cy in [ax_y, ax_y + bar_length_y / 2.0, sy]:
            self.ax.plot(
                [ax_x - tick_w, ax_x + tick_w],
                [cy, cy],
                color=color,
                linewidth=thin_lw,
                zorder=z,
            )

        for ty, value in zip(
            [ax_y, ax_y + bar_length_y / 2.0, sy],
            ["0", f"{half_ground:.0f}m", f"{style.bar_ground_length_m:.0f}m"],
        ):
            self.ax.text(
                ax_x + tick_w * 1.8,
                ty,
                value,
                color=color,
                fontsize=font_size,
                ha="left",
                va="center",
                zorder=z,
                fontfamily="monospace",
                fontweight="bold",
                path_effects=halo,
            )

        self.ax.text(
            ax_x + tick_w * 4.5,
            ax_y + bar_length_y / 2.0,
            f"V Scale: 1:{int(context.scale_y):,}",
            color=color,
            fontsize=font_size,
            ha="center",
            va="center",
            rotation=90,
            zorder=z,
            fontfamily="monospace",
            fontweight="bold",
            path_effects=halo,
        )

        nx = ax_x - bar_length_x / 2.0
        ny = sy + (bar_length_y / style.bar_ground_length_m * 500)
        arr_sz = bar_length_y * 0.5

        self.ax.annotate(
            "",
            xy=(nx, ny + arr_sz),
            xytext=(nx, ny),
            arrowprops=dict(
                facecolor=color,
                edgecolor=color,
                width=1.5,
                headwidth=6,
                headlength=8,
            ),
            zorder=z,
        )

        self.ax.text(
            nx,
            ny + arr_sz * 1.3,
            "N",
            color=color,
            fontsize=font_size + 2,
            ha="center",
            va="bottom",
            zorder=z,
            fontfamily="sans-serif",
            fontweight="bold",
            path_effects=halo,
        )

        return self

    def render(self):
        """
        Finalize the map by strictly re-locking the viewport bounds
        (preventing GeoPandas layers from auto-expanding the view)
        and hiding native Matplotlib axes.
        """
        if self.ax is None:
            raise ValueError("Canvas not found.")

        self.ax.set_xlim([self.minx, self.maxx])
        self.ax.set_ylim([self.miny, self.maxy])
        self.ax.set_axis_off()

        # Recalculate context once more in case axes bounds shifted
        self._build_render_context()

        return self