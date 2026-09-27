import warnings
import logging
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
import random
import pandas as pd
import re
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from pyproj import Transformer

plt.rcParams['font.sans-serif'] = ['Source Han Sans JP Normal', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MapPlotter")
ox.settings.use_cache = True

# ==========================================
# Shared Utility Functions
# ==========================================
def _get_actual_ax_size_cm(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width * 2.54, bbox.height * 2.54

def _compute_ground_metrics(minx, miny, maxx, maxy, crs):
    if crs == "EPSG:4326":
        a, b = 6378137.0, 6356752.314245
        e_sq = 1.0 - (b**2 / a**2)
        center_lat = math.radians((miny + maxy) / 2.0)
        sin_lat = math.sin(center_lat)
        N = a / math.sqrt(1.0 - e_sq * sin_lat**2)
        M = (a * (1.0 - e_sq)) / math.pow(1.0 - e_sq * sin_lat**2, 1.5)

        ground_w_m = math.radians(maxx - minx) * N * math.cos(center_lat)
        ground_h_m = math.radians(maxy - miny) * M
    else:
        ground_w_m = maxx - minx
        ground_h_m = maxy - miny

    return ground_w_m, ground_h_m

def decorate_map_layout(fig, ax, plotter, title_text="Urban Map"):
    fig.canvas.draw() if fig.canvas else None
    ax_width_cm, ax_height_cm = _get_actual_ax_size_cm(fig, ax)

    ground_w_m, ground_h_m = _compute_ground_metrics(
        plotter.minx, plotter.miny, plotter.maxx, plotter.maxy, plotter.crs
    )

    scale_x = ground_w_m / (ax_width_cm * 0.01)
    scale_y = ground_h_m / (ax_height_cm * 0.01)

    full_title = (
        f"{title_text}\n"
        f"Dimensions: {ax_width_cm:.1f} cm × {ax_height_cm:.1f} cm  |  "
        f"Scale H: 1:{int(scale_x):,}  |  Scale V: 1:{int(scale_y):,}"
    )

    ax.set_title(
        full_title, 
        color="white", 
        fontsize=14, 
        fontfamily='monospace', 
        fontweight='bold', 
        pad=15,
        ha='center'
    )
    logger.info(f"Layout decorated successfully. Size: {ax_width_cm:.1f}x{ax_height_cm:.1f} cm")


# ==========================================
# GenericMapPlotter: Core Physical Geography & Canvas
# ==========================================
class GenericMapPlotter:
    """
    通用自然地理與地圖基礎架構類別。
    負責投影、基礎網格、陸地、水域、植被、建築物與一般道路系統。
    """
    def __init__(self, bbox=None, center_point=None, dist=5000, 
                 custom_linewidths=None, custom_colors=None, 
                 bg_color="#cfe4fe", land_color="#54ead1", projection="wgs84"):

        if (bbox is None) == (center_point is None):
            raise ValueError("You must provide either 'bbox' OR 'center_point'.")

        self.bbox = bbox
        self.center_point = center_point
        self.dist = dist

        if self.center_point: self._check_center_point()
        if self.bbox: self._check_bbox()

        self.bg_color = bg_color
        self.land_color = land_color

        self.linewidths_map = {
            'motorway': 4.0, 'motorway_link': 3.0, 'trunk': 3.5, 'trunk_link': 2.5,
            'primary': 2.5,  'primary_link': 2.0, 'secondary': 1.5,'secondary_link': 1.0,
            'tertiary': 1.0, 'tertiary_link': 0.8,
            'residential': 0.5, 'unclassified': 0.5, 'living_street': 0.5
        }
        if custom_linewidths: self.linewidths_map.update(custom_linewidths)

        self.color_map = {
            'motorway': '#e63946',  'motorway_link': "#c934ea",
            'trunk': '#f4a261',     'trunk_link': "#ff6f00",
            'primary': '#e9c46a',   'primary_link': '#e9c46a',
            'secondary': '#2a9d8f', 'secondary_link': '#2a9d8f',
            'tertiary': "#19567c",  'tertiary_link': "#19567c",
            'residential': '#555555', 'unclassified': '#555555', 'living_street': '#555555'
        }
        if custom_colors: self.color_map.update(custom_colors)

        self.projection = projection.lower()
        self.crs = "EPSG:4326" 

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
        self.minx = self.miny = self.maxx = self.maxy = 0

        mode_str = f"BBox mode ({self.bbox})" if self.bbox else f"Center Point mode ({self.center_point}, {self.dist}m)"
        logger.info(f"GenericMapPlotter initialized with {mode_str} | Projection: {self.projection.upper()}")

    def _check_center_point(self):
        lat, lon = self.center_point
        if not (-90 <= lat <= 90 and -180 <= lon <= 180): raise ValueError("Invalid center point.")

    def _check_bbox(self):
        west, south, east, north = self.bbox
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90): raise ValueError("Invalid BBox.")
        if west >= east or south >= north: raise ValueError("Invalid BBox range.")

    def _check_is_yes(self, attr):
        if attr is None: return False
        if isinstance(attr, str): return attr == 'yes'
        if isinstance(attr, list): return 'yes' in attr
        return (attr == 'yes').any() if hasattr(attr, 'any') else False

    def _project(self, gdf):
        return gdf.to_crs(self.crs) if gdf is not None and not gdf.empty and self.crs != "EPSG:4326" else gdf

    def _process_edges(self):
        edge_colors, edge_widths, edge_zorders = [], [], []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            hw = data.get('highway', 'unclassified')
            hw = hw[0] if isinstance(hw, list) else (str(hw.iloc[0]) if hasattr(hw, 'iloc') else hw)

            is_bridge, is_tunnel = self._check_is_yes(data.get('bridge')), self._check_is_yes(data.get('tunnel'))
            width = self.linewidths_map.get(hw, 0.5)

            if is_tunnel: color, zorder = self.color_map.get(hw, '#ff9999'), 1      
            elif is_bridge: color, zorder = self.color_map.get(hw, '#ff9999'), 3        
            else: color, zorder = self.color_map.get(hw, '#666666'), 2        

            edge_widths.append(width); edge_colors.append(color); edge_zorders.append(zorder)

        self.gdf_edges['custom_width'] = edge_widths
        self.gdf_edges['custom_color'] = edge_colors
        self.gdf_edges['custom_zorder'] = edge_zorders

    def _fetch_features(self, tags, layer_name):
        """
        共用的圖層抓取邏輯。
        若區域內真的沒有該標籤資料，安全回傳 None。
        但若是遭遇網路斷線、API 超時等致命錯誤，將強制拋出例外中止程式。
        """
        try:
            raw = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
            return self._project(raw)
        except Exception as e:
            err_msg = str(e).lower()
            # 判斷是否為「查無資料」的良性例外 (OSMnx 在抓不到特徵時會拋出此類錯誤)
            if any(keyword in err_msg for keyword in ['no data', 'empty', 'insufficient', 'not find', 'no features']):
                logger.info(f"No features found for {layer_name} in this area.")
                return None
            else:
                # 遭遇網路問題或其他底層致命錯誤，不再吞掉，直接報錯
                logger.error(f"Fatal error encountered while fetching {layer_name}: {e}")
                raise e

    def setup_canvas(self, ax=None, base_width=96, highway_filter=None):
        custom_filter = f'["highway"~"{"|".join(highway_filter)}"]' if highway_filter else None

        # graph_from_bbox 若發生網路錯誤，本身就會直接噴出 requests 或 urllib 錯誤
        if self.bbox:
            self.G = ox.graph_from_bbox(bbox=self.bbox, network_type="drive", custom_filter=custom_filter)
        else:
            self.G = ox.graph_from_point(self.center_point, dist=self.dist, network_type="drive", custom_filter=custom_filter)

        if self.projection != "wgs84":
            if self.projection in ["utm", "transverse_mercator"]:
                self.G = ox.project_graph(self.G) 
                self.crs = self.G.graph['crs']
            else:
                self.crs = {"conformal": "EPSG:3857", "mercator": "EPSG:3857", "equal_area": "+proj=cea", "equidistant": "+proj=eqc"}.get(self.projection, "EPSG:3857")
                self.G = ox.project_graph(self.G, to_crs=self.crs)

        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)
        if self.gdf_nodes.empty: raise ValueError("Invalid graph bounds: No road nodes found.")

        self._process_edges()
        self.minx, self.miny, self.maxx, self.maxy = self.gdf_nodes.total_bounds 

        if ax is None:
            ground_w, ground_h = _compute_ground_metrics(self.minx, self.miny, self.maxx, self.maxy, self.crs)
            aspect_ratio = ground_h / ground_w 
            fig_height = base_width * aspect_ratio

            self.fig, self.ax = plt.subplots(figsize=(base_width, fig_height))
            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.99)
            self._is_user_ax = False
        else:
            self.ax = ax
            self.fig = ax.figure
            self._is_user_ax = True

        self.ax.set_xlim([self.minx, self.maxx])
        self.ax.set_ylim([self.miny, self.maxy])
        self.ax.set_facecolor(self.bg_color)

        if self.crs == "EPSG:4326":
            self.ax.set_aspect(1.0 / math.cos(math.radians((self.miny + self.maxy) / 2.0))) 
        else:
            self.ax.set_aspect('equal') 

        return self

    def draw_land(self):
        if getattr(self, 'gdf_admin_base', None) is None:
            poly = box(self.minx, self.miny, self.maxx, self.maxy)
            self.gdf_admin_base = gpd.GeoDataFrame(geometry=[poly], crs=self.crs)
            
        if not self.gdf_admin_base.empty:
            self.gdf_admin_base.plot(ax=self.ax, color=self.land_color, zorder=0.1)
        return self

    def draw_water(self):
        if self.gdf_water is None:
            tags = {'natural': ['water', 'bay', 'coastline'], 'waterway': ['river', 'riverbank', 'canal', 'dock'], 'water': True}
            self.gdf_water = self._fetch_features(tags, layer_name="water")
            
        if self.gdf_water is not None and not self.gdf_water.empty:
            self.gdf_water.plot(ax=self.ax, color=self.bg_color, edgecolor='none', zorder=0.2)
        return self

    def draw_greenery(self):
        if getattr(self, 'gdf_greenery', None) is None:
            tags = {'leisure': ['park', 'pitch', 'garden', 'golf_course'], 'landuse': ['forest', 'grass', 'cemetery', 'recreation_ground'], 'natural': ['wood', 'scrub']}
            self.gdf_greenery = self._fetch_features(tags, layer_name="greenery")

        if self.gdf_greenery is not None and not self.gdf_greenery.empty:
            poly_gdf = self.gdf_greenery[self.gdf_greenery.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not poly_gdf.empty:
                poly_gdf.plot(ax=self.ax, color='#86efac', alpha=0.35, edgecolor='none', zorder=0.15)
        return self

    def draw_buildings(self):
        if getattr(self, 'gdf_buildings', None) is None:
            tags = {'building': True}
            self.gdf_buildings = self._fetch_features(tags, layer_name="buildings")

        if self.gdf_buildings is not None and not self.gdf_buildings.empty:
            poly_gdf = self.gdf_buildings[self.gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not poly_gdf.empty:
                is_commercial = poly_gdf['building'].isin(['commercial', 'retail', 'office', 'hotel', 'supermarket'])
                bldg_comm, bldg_resi = poly_gdf[is_commercial], poly_gdf[~is_commercial]
                if not bldg_resi.empty: bldg_resi.plot(ax=self.ax, color='#cbd5e1', alpha=0.45, edgecolor='none', zorder=0.17)
                if not bldg_comm.empty: bldg_comm.plot(ax=self.ax, color='#fca5a5', alpha=0.65, edgecolor='none', zorder=0.18)
        return self

    def draw_roads(self):
        gdf_tunnels = self.gdf_edges[self.gdf_edges['custom_zorder'] == 1]
        gdf_ground = self.gdf_edges[self.gdf_edges['custom_zorder'] == 2]
        gdf_bridges = self.gdf_edges[self.gdf_edges['custom_zorder'] == 3]

        if not gdf_tunnels.empty: gdf_tunnels.plot(ax=self.ax, color=gdf_tunnels['custom_color'], linewidth=gdf_tunnels['custom_width'], zorder=1.0, linestyle='--')
        if not gdf_ground.empty: gdf_ground.plot(ax=self.ax, color=gdf_ground['custom_color'], linewidth=gdf_ground['custom_width'], zorder=2.0)
        if not gdf_bridges.empty: gdf_bridges.plot(ax=self.ax, color=gdf_bridges['custom_color'], linewidth=gdf_bridges['custom_width'], zorder=3.0)
        return self

    def draw_airports(self):
        if getattr(self, 'gdf_aeroway', None) is None:
            tags = {'aeroway': ['runway', 'taxiway', 'apron', 'helipad']}
            self.gdf_aeroway = self._fetch_features(tags, layer_name="airports")

        if self.gdf_aeroway is None or self.gdf_aeroway.empty: return self

        scale_factor = self.fig.get_figwidth() / 72.0
        poly_gdf = self.gdf_aeroway[self.gdf_aeroway.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        line_gdf = self.gdf_aeroway[~self.gdf_aeroway.geometry.type.isin(['Polygon', 'MultiPolygon'])]

        if not poly_gdf.empty:
            aprons = poly_gdf[poly_gdf['aeroway'] == 'apron']
            if not aprons.empty: aprons.plot(ax=self.ax, color='#d4d4d8', zorder=0.4, edgecolor='none')
            aero_areas = poly_gdf[poly_gdf['aeroway'].isin(['runway', 'taxiway'])]
            if not aero_areas.empty: aero_areas.plot(ax=self.ax, color='#52525b', zorder=0.45, edgecolor='none')

        if not line_gdf.empty:
            taxiways, runways = line_gdf[line_gdf['aeroway'] == 'taxiway'], line_gdf[line_gdf['aeroway'] == 'runway']
            if not taxiways.empty:
                taxiways.plot(ax=self.ax, color='#71717a', linewidth=4.0 * scale_factor, zorder=0.46)
                taxiways.plot(ax=self.ax, color='#fde047', linewidth=0.8 * scale_factor, zorder=0.47)
            if not runways.empty:
                runways.plot(ax=self.ax, color='#3f3f46', linewidth=14.0 * scale_factor, zorder=0.48)
                runways.plot(ax=self.ax, color='#ffffff', linewidth=1.5 * scale_factor, linestyle=(0, (10, 10)), zorder=0.49)

            placed_markings = []
            for idx, row in line_gdf.iterrows():
                if 'ref' not in row or pd.isna(row['ref']) or not str(row['ref']).strip(): continue
                geom = row.geometry
                if geom.is_empty: continue

                center_pt = geom.centroid
                x, y = center_pt.x, center_pt.y
                if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): continue

                coords = list(geom.coords) if geom.geom_type == 'LineString' else (list(geom.geoms[0].coords) if geom.geom_type == 'MultiLineString' else [])
                if len(coords) < 2: continue
                angle = math.degrees(math.atan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0]))
                if angle > 90: angle -= 180
                elif angle < -90: angle += 180

                is_rw = row.get('aeroway') == 'runway'
                fontsize, fontcolor, fontweight, z = (int(12 * scale_factor), '#ffffff', 'bold', 4.5) if is_rw else (int(7 * scale_factor), '#fde047', 'bold', 4.5)
                bbox_style = dict(facecolor='#3f3f46' if is_rw else '#71717a', alpha=0.8, edgecolor='none', boxstyle='square,pad=0.1' if is_rw else 'round,pad=0.1')

                if any(math.hypot(x - px, y - py) < (self.maxx - self.minx) * 0.005 for px, py in placed_markings): continue
                placed_markings.append((x, y))

                self.ax.text(x, y, str(row['ref']).strip(), color=fontcolor, fontsize=fontsize, ha='center', va='center', rotation=angle, zorder=z, clip_on=True, fontfamily='sans-serif', fontweight=fontweight, bbox=bbox_style)
        return self

    def draw_grid(self, rows=5, cols=5, color="#8072ff", alpha=0.15, linewidth=0.8):
        if self.ax is None: raise ValueError("Canvas not found.")

        self.grid_cells = {}
        x_steps = [self.minx + i * (self.maxx - self.minx) / cols for i in range(cols + 1)]
        y_steps = [self.miny + j * (self.maxy - self.miny) / rows for j in range(rows + 1)]
        cols_labels = [chr(65 + i) for i in range(cols)]

        for i in range(cols + 1):
            self.ax.plot([x_steps[i], x_steps[i]], [self.miny, self.maxy], color=color, alpha=alpha, linewidth=linewidth, linestyle='--', zorder=5.0)
            if i < cols: self.ax.text(x_steps[i] + (self.maxx - self.minx) / (cols * 2.0), self.maxy, cols_labels[i], color=color, alpha=0.7, fontsize=9, ha='center', va='bottom', zorder=5.0, fontfamily='monospace')

        for j in range(rows + 1):
            self.ax.plot([self.minx, self.maxx], [y_steps[j], y_steps[j]], color=color, alpha=alpha, linewidth=linewidth, linestyle='--', zorder=5.0)
            if j < rows: self.ax.text(self.maxx, y_steps[j] + (self.maxy - self.miny) / (rows * 2.0), str(rows - j), color=color, alpha=0.7, fontsize=9, ha='left', va='center', zorder=5.0, fontfamily='monospace')

        for i in range(cols):
            for j in range(rows):
                self.grid_cells[f"{cols_labels[i]}{rows - j}"] = (x_steps[i], y_steps[j], x_steps[i+1], y_steps[j+1])
        return self

    def draw_coordinates(self):
        if self.ax is None: raise ValueError("Canvas not found.")

        if self.crs != "EPSG:4326":
            to_wgs84, to_map = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True), Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            w, s = to_wgs84.transform(self.minx, self.miny)
            e, n = to_wgs84.transform(self.maxx, self.maxy)
        else:
            w, s, e, n = self.minx, self.miny, self.maxx, self.maxy
            to_map = None

        span = min(e - w, n - s)
        interval = 1 / 120.0 if span < 0.05 else (1 / 60.0 if span < 0.1 else (5 / 60.0 if span < 0.5 else 0.1))

        def format_dms(val_orig, is_lat):
            val = abs(val_orig)
            deg, mins_float = int(val), (val - int(val)) * 60
            mins, secs = int(mins_float), int(round((mins_float - int(mins_float)) * 60))
            if secs >= 60: secs, mins = 0, mins + 1
            if mins >= 60: mins, deg = 0, deg + 1
            dir_str = ('N' if val_orig >= 0 else 'S') if is_lat else ('E' if val_orig >= 0 else 'W')
            return f"{deg}°{mins}'{secs}\"{dir_str}" if secs > 0 else (f"{deg}°{mins}'{dir_str}" if mins > 0 else f"{deg}°{dir_str}")

        lon_ticks, lat_ticks = [], []
        curr_lon = math.ceil(w / interval) * interval
        while curr_lon <= e: lon_ticks.append(curr_lon); curr_lon += interval
        curr_lat = math.ceil(s / interval) * interval
        while curr_lat <= n: lat_ticks.append(curr_lat); curr_lat += interval

        scale_factor = self.fig.get_figwidth() / 72.0
        fontsize, color, halo = max(6, int(7 * scale_factor)), "#e2e8f0", [pe.withStroke(linewidth=2.5 * scale_factor, foreground="#091c15")]
        tick_len_x, tick_len_y = (self.maxx - self.minx) * 0.005, (self.maxy - self.miny) * 0.005

        for lon in lon_ticks:
            px = max(self.minx, min(to_map.transform(lon, s)[0] if to_map else lon, self.maxx))
            self.ax.plot([px, px], [self.miny, self.miny + tick_len_y], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.plot([px, px], [self.maxy, self.maxy - tick_len_y], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.text(px, self.miny + tick_len_y * 1.5, format_dms(lon, False), color=color, fontsize=fontsize, ha='center', va='bottom', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)
            self.ax.text(px, self.maxy - tick_len_y * 1.5, format_dms(lon, False), color=color, fontsize=fontsize, ha='center', va='top', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)

        for lat in lat_ticks:
            py = max(self.miny, min(to_map.transform(w, lat)[1] if to_map else lat, self.maxy))
            self.ax.plot([self.minx, self.minx + tick_len_x], [py, py], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.plot([self.maxx, self.maxx - tick_len_x], [py, py], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.text(self.minx + tick_len_x * 1.5, py, format_dms(lat, True), color=color, fontsize=fontsize, ha='left', va='center', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)
            self.ax.text(self.maxx - tick_len_x * 1.5, py, format_dms(lat, True), color=color, fontsize=fontsize, ha='right', va='center', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)

        return self

    def draw_scale(self, x_offset=0.08, y_offset=0.08, color="white", font_size=9):
        if self.ax is None or self.fig is None: raise ValueError("Canvas not found.")
        self.fig.canvas.draw() if self.fig.canvas else None

        map_w, map_h = _get_actual_ax_size_cm(self.fig, self.ax)
        grd_w, grd_h = _compute_ground_metrics(self.minx, self.miny, self.maxx, self.maxy, self.crs)

        sc_x, sc_y = grd_w / (map_w * 0.01), grd_h / (map_h * 0.01)
        u_x, u_y = (self.maxx - self.minx) / map_w, (self.maxy - self.miny) / map_h 
        gm_x, gm_y = sc_x * 0.01, sc_y * 0.01

        bar_len_x, bar_len_y = u_x * 2.0, u_y * 2.0
        ax_x, ax_y = self.maxx - (self.maxx - self.minx) * x_offset, self.miny + (self.maxy - self.miny) * y_offset
        sx, sy = ax_x - bar_len_x, ax_y + bar_len_y
        tick_h, tick_w = (self.maxy - self.miny) * 0.004, (self.maxx - self.minx) * 0.004

        self.ax.plot([sx, ax_x], [ax_y, ax_y], color=color, linewidth=2.0, zorder=10)
        for cx in [sx, sx + bar_len_x/2.0, ax_x]: self.ax.plot([cx, cx], [ax_y - tick_h, ax_y + tick_h], color=color, linewidth=1.5, zorder=10)
        for tx, val in zip([sx, sx + bar_len_x/2.0, ax_x], ["0", f"{gm_x:.0f}m", f"{gm_x*2:.0f}m"]): self.ax.text(tx, ax_y - tick_h*1.8, val, color=color, fontsize=font_size, ha='center', va='top', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(ax_x - bar_len_x/2.0, ax_y + tick_h*2.5, f"H Scale: 1:{int(sc_x):,}", color=color, fontsize=font_size, ha='center', va='bottom', zorder=10, fontfamily='monospace', fontweight='bold')

        self.ax.plot([ax_x, ax_x], [ax_y, sy], color=color, linewidth=2.0, zorder=10)
        for cy in [ax_y, ax_y + bar_len_y/2.0, sy]: self.ax.plot([ax_x - tick_w, ax_x + tick_w], [cy, cy], color=color, linewidth=1.5, zorder=10)
        for ty, val in zip([ax_y, ax_y + bar_len_y/2.0, sy], ["0", f"{gm_y:.0f}m", f"{gm_y*2:.0f}m"]): self.ax.text(ax_x + tick_w*1.8, ty, val, color=color, fontsize=font_size, ha='left', va='center', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(ax_x + tick_w*4.5, ax_y + bar_len_y/2.0, f"V Scale: 1:{int(sc_y):,}", color=color, fontsize=font_size, ha='center', va='center', rotation=90, zorder=10, fontfamily='monospace', fontweight='bold')

        nx, ny, arr_sz = ax_x - bar_len_x/2.0, sy + u_y*0.8, u_y*0.5
        self.ax.annotate('', xy=(nx, ny + arr_sz), xytext=(nx, ny), arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=8), zorder=10)
        self.ax.text(nx, ny + arr_sz*1.3, "N", color=color, fontsize=font_size+2, ha='center', va='bottom', zorder=10, fontfamily='sans-serif', fontweight='bold')

        return self

    def render(self):
        if self.ax is None: raise ValueError("Canvas not found.")
        self.ax.set_axis_off()
        return self