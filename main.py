import warnings
import logging
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
import random
import pandas as pd
import re
from shapely.geometry import MultiPolygon, box
import geopandas as gpd

plt.rcParams['font.sans-serif'] = ['Source Han Sans JP Normal', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MapPlotter")
ox.settings.use_cache = True

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
    logger.info(f"Layout decorated successfully. Size: {ax_width_cm:.1f}x{ax_height_cm:.1f} cm | H-Scale: 1:{int(scale_x):,} | V-Scale: 1:{int(scale_y):,}")

class MapPlotter:
    def __init__(self, bbox=None, center_point=None, dist=5000, 
                 custom_linewidths=None, custom_colors=None, 
                 bg_color="#8FA8DA", land_color="#a6dcca", 
                 admin_widths=(0, 0, 0, 0, 0, 0, 2.5, 0, 0.5, 0), 
                 projection="wgs84"):
        
        if (bbox is None) == (center_point is None):
            logger.error("Initialization error: You must provide either 'bbox' OR 'center_point', not both or neither.")
            raise ValueError("You must provide either 'bbox' OR 'center_point'.")

        self.bbox = bbox
        self.center_point = center_point
        self.dist = dist
        
        if self.center_point: self._check_center_point()
        if self.bbox: self._check_bbox()
            
        self.bg_color = bg_color
        self.land_color = land_color
        
        if len(admin_widths) != 10:
            raise ValueError("admin_widths must be a tuple of length 10")
        self.admin_widths = admin_widths
        
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
        self.gdf_admins = {} 
        self.gdf_railways = None
        self.gdf_aeroway = None
        self.gdf_exits = None

        self.fig = None
        self.ax = None
        self._is_user_ax = False
        self.minx = self.miny = self.maxx = self.maxy = 0
        self._admins_fetched = False
        
        mode_str = f"BBox mode ({self.bbox})" if self.bbox else f"Center Point mode ({self.center_point}, {self.dist}m)"
        logger.info(f"MapPlotter initialized with {mode_str} | Projection: {self.projection.upper()}")
        
    def _fetch_and_categorize_admins(self):
        if self._admins_fetched: return
            
        logger.info("Batch fetching ALL administrative boundaries...")
        try:
            tags = {'boundary': 'administrative'}
            raw_gdf = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                
            gdf_proj = self._project(raw_gdf)
            gdf_poly = gdf_proj[gdf_proj.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            
            if 'admin_level' in gdf_poly.columns:
                gdf_poly['admin_level_clean'] = gdf_poly['admin_level'].apply(
                    lambda x: str(x[0]) if isinstance(x, list) else str(x)
                )
                
                for lvl_str, group in gdf_poly.groupby('admin_level_clean'):
                    if lvl_str.replace('.', '', 1).isdigit() and lvl_str != 'nan':
                        self.gdf_admins[int(float(lvl_str))] = group
                        
            self._admins_fetched = True
        except Exception as e:
            logger.warning(f"Failed to fetch administrative boundaries: {e}")
            self._admins_fetched = True

    def draw_admins(self):
        self._fetch_and_categorize_admins()
        for i, width in enumerate(self.admin_widths):
            level = i + 1 
            if width <= 0: continue
            
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
            
            if level == 9:
                color_ids = list(range(len(gdf)))
                random.seed(42); random.shuffle(color_ids)
                gdf = gdf.copy()
                gdf['color_id'] = color_ids
                gdf.plot(ax=self.ax, column='color_id', cmap='hsv', alpha=0.25, edgecolor='#ffffff', linewidth=width, zorder=0.3)
            else:
                gdf.plot(ax=self.ax, facecolor='none', edgecolor="#776de5", linewidth=width, linestyle='-.', zorder=0.35 + (level * 0.01))
        return self

    def draw_admin_labels(self, levels=(4, 6, 7, 8, 9), min_distance_ratio=0.015):
        if self.ax is None: raise ValueError("Canvas not found.")
        self._fetch_and_categorize_admins()
        
        target_levels = [levels] if isinstance(levels, int) else (levels if levels else [4, 6, 7, 8, 9])
        fig_w = self.fig.get_figwidth()
        scale_factor = fig_w / 48.0
        
        font_size_map = {4: int(72 * scale_factor), 6: int(55 * scale_factor), 7: int(35 * scale_factor), 8: int(22 * scale_factor), 9: int(15 * scale_factor)}
        placed_labels = []
        map_span_x = self.maxx - self.minx
        min_dist = map_span_x * min_distance_ratio 

        for level in sorted(target_levels):
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
                
            pri_fontsize = font_size_map.get(level, int(8 * scale_factor))
            sec_fontsize = max(int(4 * scale_factor), int(pri_fontsize * 0.6))

            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty: continue
                
                pri_name = str(row.get('name') or row.get('name:ja')).strip()
                if not pri_name or pri_name == 'None' or "所属未定地" in pri_name: continue

                sec_name = ""
                priority_tags = ['name:ja_rm', 'name:en']
                
                for tag in priority_tags:
                    val = row.get(tag)
                    # 確保值存在、不是 NaN、也不是空字串
                    if pd.notna(val) and str(val).strip().lower() not in ['', 'nan', 'none']:
                        sec_name = str(val).strip()
                        break  # 找到最高優先級的有效名稱，立刻中斷跳出！

                # 針對 Admin 9 強制大寫處理
                if sec_name:
                    if level == 9:
                        sec_name = sec_name.upper()
                        
                if isinstance(geom, MultiPolygon): geom = max(geom.geoms, key=lambda p: p.area)
                center_pt = geom.centroid if geom.contains(geom.centroid) else geom.representative_point()
                x, y = center_pt.x, center_pt.y

                if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): continue

                if any(math.hypot(x - px, y - py) < min_dist for px, py in placed_labels): continue 
                placed_labels.append((x, y))

                y_offset = -map_span_x * 0.0001 * scale_factor
                bg_style = dict(facecolor='#0a0d0c', alpha=0.6, edgecolor='none', boxstyle=f'round,pad={0.08 * scale_factor}')
                
                if (" " in sec_name or "-" in sec_name) and len(sec_name) >= 14:
                    _splited = re.split(r"[-\s]", sec_name)
                    sec_name = (_splited[0][0].upper() + ".-") + "-".join(_splited[1:])
                
                if len(pri_name) < 4: pri_name = " ".join(pri_name)

                self.ax.text(x, y + y_offset, pri_name, color="white", fontsize=pri_fontsize, ha='center', va='bottom', zorder=6.0, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=bg_style)
                if sec_name: self.ax.text(x, y - y_offset, sec_name, color="#cbd5e1", fontsize=sec_fontsize, ha='center', va='top', zorder=6.0, clip_on=True, fontfamily='sans-serif', bbox=bg_style)
        return self
    
    def draw_greenery(self):
        """
        繪製城市綠地 (公園、森林、草地、墓地等)，作為城市肌理的底色。
        """
        if getattr(self, 'gdf_greenery', None) is None:
            logger.info("Fetching greenery features (parks, forests, grass)...")
            try:
                tags = {
                    'leisure': ['park', 'pitch', 'garden', 'golf_course'],
                    'landuse': ['forest', 'grass', 'cemetery', 'recreation_ground'],
                    'natural': ['wood', 'scrub']
                }
                raw_green = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_greenery = self._project(raw_green)
            except Exception as e:
                logger.warning(f"Failed to fetch greenery features: {e}")
                self.gdf_greenery = None

        if self.gdf_greenery is not None and not self.gdf_greenery.empty:
            poly_gdf = self.gdf_greenery[self.gdf_greenery.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not poly_gdf.empty:
                # 採用極淡的若草色 (Soft Green) 融入地圖底層
                poly_gdf.plot(ax=self.ax, color='#86efac', alpha=0.35, edgecolor='none', zorder=0.15)
                logger.info("Greenery layers rendered.")
        return self

    def draw_buildings(self):
        """
        繪製建築物輪廓，並區分商業/辦公建築與一般住宅，豐富地圖的城市緻密度。
        """
        if getattr(self, 'gdf_buildings', None) is None:
            logger.info("Fetching building footprints (this may take a moment)...")
            try:
                tags = {'building': True}
                raw_bldgs = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_buildings = self._project(raw_bldgs)
            except Exception as e:
                logger.warning(f"Failed to fetch buildings: {e}")
                self.gdf_buildings = None

        if self.gdf_buildings is not None and not self.gdf_buildings.empty:
            poly_gdf = self.gdf_buildings[self.gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not poly_gdf.empty:
                # 區分商業區與住宅區
                commercial_types = ['commercial', 'retail', 'office', 'hotel', 'supermarket']
                is_commercial = poly_gdf['building'].isin(commercial_types)
                
                bldg_comm = poly_gdf[is_commercial]
                bldg_resi = poly_gdf[~is_commercial]

                # 一般住宅：淺灰色、半透明
                if not bldg_resi.empty:
                    bldg_resi.plot(ax=self.ax, color='#cbd5e1', alpha=0.45, edgecolor='none', zorder=0.17)
                
                # 商業/辦公建築：淡玫瑰紅/橘色，突顯城市商業核心
                if not bldg_comm.empty:
                    bldg_comm.plot(ax=self.ax, color='#fca5a5', alpha=0.65, edgecolor='none', zorder=0.18)
                    
                logger.info("Building footprints rendered (separated by commercial/residential).")
        return self

    def draw_coordinates(self):
        """
        在地圖的四個邊框繪製經緯度 (WGS84) 的精確刻度與度分秒標籤。
        自動適應比例尺，並將 WGS84 座標投影貼合目前的畫布投影系統。
        """
        from pyproj import Transformer
        import matplotlib.patheffects as pe

        if self.ax is None: 
            raise ValueError("Canvas not found.")
        
        # 取得地圖邊界在 WGS84 (真實經緯度) 下的範圍
        if self.crs != "EPSG:4326":
            to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            to_map = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            w, s = to_wgs84.transform(self.minx, self.miny)
            e, n = to_wgs84.transform(self.maxx, self.maxy)
        else:
            w, s, e, n = self.minx, self.miny, self.maxx, self.maxy
            to_map = None

        # 動態決定刻度間距 (1:25,000 的地圖，通常 1分鐘 或 30秒 是一個好間距)
        span = min(e - w, n - s)
        if span < 0.05:
            interval = 1 / 120.0  # 30 秒
        elif span < 0.1:
            interval = 1 / 60.0   # 1 分鐘
        elif span < 0.5:
            interval = 5 / 60.0   # 5 分鐘
        else:
            interval = 0.1        # 0.1 度
        
        def format_dms(val_orig, is_lat):
            """將小數度數轉換為標準的地圖學 度、分、秒 (DMS) 格式"""
            val = abs(val_orig)
            deg = int(val)
            mins_float = (val - deg) * 60
            mins = int(mins_float)
            secs = int(round((mins_float - mins) * 60))
            
            # 處理進位問題
            if secs >= 60:
                secs = 0
                mins += 1
            if mins >= 60:
                mins = 0
                deg += 1
                
            dir_str = ('N' if val_orig >= 0 else 'S') if is_lat else ('E' if val_orig >= 0 else 'W')
            
            if secs > 0:
                return f"{deg}°{mins}'{secs}\"{dir_str}"
            elif mins > 0:
                return f"{deg}°{mins}'{dir_str}"
            else:
                return f"{deg}°{dir_str}"

        # 產生經度 (Longitude) 與緯度 (Latitude) 的陣列
        lon_ticks, lat_ticks = [], []
        
        curr_lon = math.ceil(w / interval) * interval
        while curr_lon <= e:
            lon_ticks.append(curr_lon)
            curr_lon += interval
            
        curr_lat = math.ceil(s / interval) * interval
        while curr_lat <= n:
            lat_ticks.append(curr_lat)
            curr_lat += interval

        scale_factor = self.fig.get_figwidth() / 72.0
        fontsize = max(6, int(7 * scale_factor))
        color = "#e2e8f0" # 質感的淺金屬灰
        halo = [pe.withStroke(linewidth=2.5 * scale_factor, foreground="#091c15")] # 配合背景顏色的光暈防遮擋

        tick_len_x = (self.maxx - self.minx) * 0.005
        tick_len_y = (self.maxy - self.miny) * 0.005

        # 繪製經度 (Longitude) 標籤於上下邊界
        for lon in lon_ticks:
            label = format_dms(lon, is_lat=False)
            
            if to_map:
                px, py_bottom = to_map.transform(lon, s)
                _, py_top = to_map.transform(lon, n)
            else:
                px, py_bottom, py_top = lon, self.miny, self.maxy
            
            px = max(self.minx, min(px, self.maxx)) # 限制在邊界內
            
            # 上下邊緣的小刻度 (向內延伸)
            self.ax.plot([px, px], [self.miny, self.miny + tick_len_y], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.plot([px, px], [self.maxy, self.maxy - tick_len_y], color=color, linewidth=1.2 * scale_factor, zorder=8)
            
            # 文字標籤
            self.ax.text(px, self.miny + tick_len_y * 1.5, label, color=color, fontsize=fontsize, ha='center', va='bottom', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)
            self.ax.text(px, self.maxy - tick_len_y * 1.5, label, color=color, fontsize=fontsize, ha='center', va='top', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)

        # 繪製緯度 (Latitude) 標籤於左右邊界
        for lat in lat_ticks:
            label = format_dms(lat, is_lat=True)
            
            if to_map:
                px_left, py = to_map.transform(w, lat)
                px_right, _ = to_map.transform(e, lat)
            else:
                px_left, px_right, py = self.minx, self.maxx, lat
            
            py = max(self.miny, min(py, self.maxy)) # 限制在邊界內
            
            # 左右邊緣的小刻度 (向內延伸)
            self.ax.plot([self.minx, self.minx + tick_len_x], [py, py], color=color, linewidth=1.2 * scale_factor, zorder=8)
            self.ax.plot([self.maxx, self.maxx - tick_len_x], [py, py], color=color, linewidth=1.2 * scale_factor, zorder=8)
            
            # 文字標籤
            self.ax.text(self.minx + tick_len_x * 1.5, py, label, color=color, fontsize=fontsize, ha='left', va='center', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)
            self.ax.text(self.maxx - tick_len_x * 1.5, py, label, color=color, fontsize=fontsize, ha='right', va='center', fontfamily='monospace', zorder=8.5, fontweight='bold', path_effects=halo)

        logger.info("Geographic coordinates (Lat/Lon) successfully rendered on borders.")
        return self

    def draw_stations(self, min_distance_ratio=0.010):
        """
        繪製鐵路車站的完整複合體 (Station Complex)。
        1. 抓取 stop_area(關聯), train_station(站體), platform(月台)。
        2. 【繪圖脫鉤】：將所有 LineString (通道/線狀月台) 展開 3 公尺變為走廊，
           並與所有 Polygon 聯集 (Union)，形成單一乾淨的車站輪廓。自動過濾 Point(出口)。
        3. 【標籤脫鉤】：獨立計算具備英日雙語站名的幾何中心，將標籤貼於輪廓之上。
        """
        import pandas as pd
        import geopandas as gpd
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, MultiPolygon

        if getattr(self, 'gdf_stations', None) is None:
            try:
                # 🌟 抓取車站複合體的三大核心要素
                tags = {
                    'public_transport': 'stop_area', # 包含站體、通道與出口的關聯集合
                    'building': 'train_station',     # 實體建築
                    'railway': ['station', 'platform'] # 車站節點與月台
                }
                raw_stations = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                
                gdf_proj = self._project(raw_stations)
                
                # 剔除公車相關設施 (避免 stop_area 混入公車總站)
                is_bus = pd.Series(False, index=gdf_proj.index)
                for col in ['highway', 'amenity']:
                    if col in gdf_proj.columns:
                        is_bus = is_bus | gdf_proj[col].isin(['bus_station', 'bus_stop'])
                
                self.gdf_stations = gdf_proj[~is_bus].copy()
                logger.info(f"Loaded {len(self.gdf_stations)} station features (building, platform, stop_area).")
                
            except Exception as e:
                logger.warning(f"Failed to fetch stations: {e}")
                self.gdf_stations = None

        if self.gdf_stations is not None and not self.gdf_stations.empty:
            scale_factor = self.fig.get_figwidth() / 72.0
            
            # ==========================================
            # 第一階段：視覺渲染 (融合站體、月台與通道)
            # ==========================================
            render_geoms = []
            
            for idx, row in self.gdf_stations.iterrows():
                geom = row.geometry
                if geom.is_empty:
                    continue
                    
                # 點 (Point)：通常是孤立出口或標籤錨點，絕對不畫 (避免滿地出口)
                if geom.geom_type in ['Point', 'MultiPoint']:
                    continue
                # 線 (LineString)：通道或線狀月台。給予 3 公尺的緩衝 (Buffer)，變成 6 公尺寬的實體走廊！
                elif geom.geom_type in ['LineString', 'MultiLineString']:
                    render_geoms.append(geom.buffer(3.0)) # 因為已是投影座標，單位為公尺
                # 面 (Polygon)：標準站體與大月台，直接加入
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    render_geoms.append(geom)

            if render_geoms:
                # 🌟 透過聯集 (unary_union) 消除內部重疊的邊界，把所有碎片融合成一個乾淨的「車站複合體」
                merged_footprint = unary_union(render_geoms)
                
                # 將融合後的巨大多邊形包裝回 GeoDataFrame 並繪製
                footprint_gdf = gpd.GeoDataFrame(geometry=[merged_footprint], crs=self.gdf_stations.crs)
                footprint_gdf.plot(ax=self.ax, facecolor='#f4f4f5', edgecolor='#52525b', linewidth=1.5 * scale_factor, zorder=3.8)

            # ==========================================
            # 第二階段：站名標籤處理 (精準命名與防衝突)
            # ==========================================
            station_groups = {}
            for idx, row in self.gdf_stations.iterrows():
                pri_name = str(row.get('name') or row.get('name:ja')).strip()
                if not pri_name or pri_name.lower() in ['nan', 'none']:
                    continue
                
                sec_name = ""
                for tag in ['name:en', 'name:ja_rm', 'name:ja-Latn', 'name:ja:Latn']:
                    val = row.get(tag)
                    if pd.notna(val) and str(val).strip().lower() not in ['', 'nan', 'none']:
                        sec_name = str(val).strip()
                        break
                
                # 🌟 嚴格要求：只有具備英文/羅馬字的主站，才有資格被標上名字
                if not sec_name:
                    continue
                    
                key = (pri_name, sec_name)
                if key not in station_groups:
                    station_groups[key] = []
                # 把具備名字的節點、多邊形收集起來，用來算標籤的中心點
                station_groups[key].append(row.geometry)

            fontsize = max(4, int(4.5 * scale_factor))
            bbox_style = dict(
                facecolor='#18181b', 
                alpha=0.85, 
                edgecolor='#facc15', 
                linewidth=0.6 * scale_factor, 
                boxstyle=f'square,pad={0.15 * scale_factor}'
            )

            placed_stations = []
            min_dist = (self.maxx - self.minx) * min_distance_ratio

            for (pri_name, sec_name), geoms in station_groups.items():
                merged_geom = unary_union(geoms)
                center_pt = merged_geom.centroid if merged_geom.contains(merged_geom.centroid) else merged_geom.representative_point()
                
                x, y = center_pt.x, center_pt.y
                if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): 
                    continue

                if any(math.hypot(x - px, y - py) < min_dist for px, py in placed_stations): 
                    continue
                placed_stations.append((x, y))

                label_text = pri_name + f"\n{sec_name}"

                self.ax.text(
                    x, y, label_text,
                    color="#fef08a",
                    fontsize=fontsize,
                    ha='center',
                    va='center',
                    zorder=6.5,
                    clip_on=True,
                    fontfamily='sans-serif',
                    fontweight='bold',
                    bbox=bbox_style,
                    linespacing=1.2
                )

        return self
    
    def draw_piers(self):
        """
        獨立抓取碼頭與渡輪站。
        解決名稱過長的問題，僅畫一個優雅的錨點符號作為示意。
        支援萬用幾何形狀 (Point, LineString, Polygon)。
        """
        import pandas as pd
        if getattr(self, 'gdf_piers', None) is None:
            try:
                # 針對碼頭/渡輪的專屬 OSM 標籤
                tags = {'amenity': 'ferry_terminal', 'man_made': 'pier'}
                raw_piers = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_piers = self._project(raw_piers)
            except Exception:
                self.gdf_piers = None

        if self.gdf_piers is None or self.gdf_piers.empty:
            return self
            
        scale_factor = self.fig.get_figwidth() / 72.0
        fontsize = max(8, int(15 * scale_factor))

        placed_piers = []
        min_dist = (self.maxx - self.minx) * 0.015

        for idx, row in self.gdf_piers.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            
            center_pt = geom.centroid
            if geom.geom_type in ['Polygon', 'MultiPolygon'] and not geom.contains(center_pt):
                center_pt = geom.representative_point()

            x, y = center_pt.x, center_pt.y
            
            if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): continue

            if any(math.hypot(x - px, y - py) < min_dist for px, py in placed_piers): continue
            placed_piers.append((x, y))

            # 🌟 修正：強制指定 fontfamily 為 'DejaVu Sans'
            # Matplotlib 內建的 DejaVu Sans 字型保證包含了 U+2693 (⚓) 錨點符號！
            self.ax.text(
                x, y, "⚓", 
                color="#0ea5e9", # 航海藍
                fontsize=fontsize, 
                ha='center', 
                va='center',
                zorder=6.4, # 確保在水域上方
                clip_on=True, 
                fontfamily='DejaVu Sans' 
            )

        logger.info(f"Rendered {len(placed_piers)} ferry terminals and piers with anchor symbols.")
        return self
    
    def draw_highway_refs(self, min_distance_ratio=0.03):
        """
        繪製高速公路與幹道的編號 (ref 屬性)。
        依據使用者要求：不使用盾牌，僅使用簡約的方框包覆數字。
        """
        import pandas as pd
        if self.gdf_edges is None or self.gdf_edges.empty: 
            return self

        scale_factor = self.fig.get_figwidth() / 72.0
        fontsize = max(6, int(12 * scale_factor))
        
        placed_refs = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio
        
        for idx, row in self.gdf_edges.iterrows():
            ref = row.get('ref')
            if str(ref).strip().lower() == 'nan' or not str(ref).strip(): 
                continue
                
            hw = row.get('highway')
            # 僅標示主要封閉式道路或重要幹道
            if hw not in ['motorway', 'motorway_link', 'trunk', 'trunk_link']:
                continue
                
            # 若有多個編號 (以分號隔開)，僅取第一個
            ref_str = str(ref).strip().split(';')[0]
            geom = row.geometry
            if geom is None or geom.is_empty: 
                continue
            
            # 過濾掉太短的線段，避免標籤過度密集
            if geom.length < (self.maxx - self.minx) * 0.01: 
                continue
            
            # 取線段正中央作為標籤錨點
            mid_pt = geom.interpolate(0.5, normalized=True)
            x, y = mid_pt.x, mid_pt.y
            if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): 
                continue
            
            # 防碰撞偵測
            if any(math.hypot(x - px, y - py) < min_dist for px, py in placed_refs): 
                continue
            placed_refs.append((x, y))
            
            # 繪製方框數字 (深灰底色 + 白框 + 白字，極簡風格)
            bbox_style = dict(
                facecolor='#27272a', 
                alpha=0.85, 
                edgecolor='#d4d4d8', 
                linewidth=1.0 * scale_factor, 
                boxstyle=f'square,pad={0.25 * scale_factor}' # 🌟 指定為方塊形狀 (square)
            )
            
            self.ax.text(
                x, y, ref_str, 
                color="white", fontsize=fontsize, ha='center', va='center', 
                zorder=4.8, clip_on=True, fontfamily='sans-serif', fontweight='bold', 
                bbox=bbox_style
            )

        logger.info(f"Rendered {len(placed_refs)} highway reference boxes.")
        return self

    def draw_exits(self, min_distance_ratio=0.015):
        """
        獨立抓取並繪製交流道與出口 (motorway_junction)。
        通常為點狀 (Node)，會萃取 name 與 ref 屬性組合標示。
        """
        import pandas as pd
        if getattr(self, 'gdf_exits', None) is None:
            try:
                # OSM 中標準的交流道/出口標籤
                tags = {'highway': 'motorway_junction'}
                raw_exits = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_exits = self._project(raw_exits)
            except Exception:
                self.gdf_exits = None

        if self.gdf_exits is None or self.gdf_exits.empty: 
            return self
            
        scale_factor = self.fig.get_figwidth() / 72.0
        fontsize = max(6, int(10 * scale_factor))
        
        placed_exits = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio
        
        for idx, row in self.gdf_exits.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: 
                continue
            
            # 確保取得中心點
            center_pt = geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom
            x, y = center_pt.x, center_pt.y
            if not (self.minx <= x <= self.maxx and self.miny <= y <= self.maxy): 
                continue
            
            name = str(row.get('name', '')).strip()
            ref = str(row.get('ref', '')).strip()
            if name.lower() == 'nan': name = ""
            if ref.lower() == 'nan': ref = ""
            
            if not name and not ref: 
                continue
            
            # 組合出口編號與名稱 (例如: "3 渋谷" 或單純 "渋谷")
            label = f"{ref} {name}".strip()
            
            if any(math.hypot(x - px, y - py) < min_dist for px, py in placed_exits): 
                continue
            placed_exits.append((x, y))
            
            # 繪製交流道標示 (黑底 + 亮黃字/黃框，模擬高速公路標誌盤)
            bbox_style = dict(
                facecolor='#18181b', 
                alpha=0.9, 
                edgecolor='#facc15', 
                linewidth=0.8 * scale_factor, 
                boxstyle=f'square,pad={0.2 * scale_factor}'
            )
            
            self.ax.text(
                x, y, label, 
                color="#fef08a", fontsize=fontsize, ha='left', va='center', 
                zorder=4.9, clip_on=True, fontfamily='sans-serif', fontweight='bold', 
                bbox=bbox_style
            )

        logger.info(f"Rendered {len(placed_exits)} exits and interchanges.")
        return self

    def _check_center_point(self):
        lat, lon = self.center_point
        if not (-90 <= lat <= 90 and -180 <= lon <= 180): raise ValueError(f"Invalid center point.")

    def _check_bbox(self):
        west, south, east, north = self.bbox
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90): raise ValueError(f"Invalid BBox.")
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

    def setup_canvas(self, ax=None, base_width=96, highway_filter=None):
        """
        修正優化：根據指定的 base_width (橫向寬度) 以及 BBox 的實際幾何長寬比，
        自動計算並建立對應的縱向高度畫布，完美貼合邊界，絕不拉扯失真！
        """
        custom_filter = f'["highway"~"{"|".join(highway_filter)}"]' if highway_filter else None

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
            # 🌟 透過真實物理長寬計算完美貼合的高度
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
        
        # 🌟 鎖死地理邊界極限，不准 Matplotlib 自動擴展
        self.ax.set_xlim([self.minx, self.maxx])
        self.ax.set_ylim([self.miny, self.maxy])
        
        self.ax.set_facecolor(self.bg_color)
        if self.crs == "EPSG:4326":
            self.ax.set_aspect(1.0 / math.cos(math.radians((self.miny + self.maxy) / 2.0))) 
        else:
            self.ax.set_aspect('equal') 

        return self

    def draw_land(self):
        if self.gdf_admin_base is None:
            self.gdf_admin_base = self._project(gpd.GeoDataFrame(geometry=[box(self.minx, self.miny, self.maxx, self.maxy)], crs="EPSG:4326"))
        if not self.gdf_admin_base.empty:
            self.gdf_admin_base.plot(ax=self.ax, color=self.land_color, zorder=0.1)
        return self

    def draw_water(self):
        if self.gdf_water is None:
            try:
                tags = {'natural': ['water', 'bay', 'coastline'], 'waterway': ['river', 'riverbank', 'canal', 'dock'], 'water': True}
                raw_water = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_water = self._project(raw_water)
            except Exception:
                self.gdf_water = None
        if self.gdf_water is not None and not self.gdf_water.empty:
            self.gdf_water.plot(ax=self.ax, color=self.bg_color, edgecolor='none', zorder=0.2)
        return self

    def draw_roads(self):
        gdf_tunnels = self.gdf_edges[self.gdf_edges['custom_zorder'] == 1]
        gdf_ground = self.gdf_edges[self.gdf_edges['custom_zorder'] == 2]
        gdf_bridges = self.gdf_edges[self.gdf_edges['custom_zorder'] == 3]

        if not gdf_tunnels.empty: gdf_tunnels.plot(ax=self.ax, color=gdf_tunnels['custom_color'], linewidth=gdf_tunnels['custom_width'], zorder=1.0, linestyle='--')
        if not gdf_ground.empty: gdf_ground.plot(ax=self.ax, color=gdf_ground['custom_color'], linewidth=gdf_ground['custom_width'], zorder=2.0)
        if not gdf_bridges.empty: gdf_bridges.plot(ax=self.ax, color=gdf_bridges['custom_color'], linewidth=gdf_bridges['custom_width'], zorder=3.0)
        return self
    
    def draw_railways(self):
        """
        繪製鐵軌網格，並依照運量與系統進行視覺分級：
        1. 傳統重軌 (rail, narrow_gauge)：底層黑色，上層虛線為專屬顏色 (預設白色)。形成「黑白/黑綠相間」。
        2. 都市捷運 (subway, light_rail, monorail, tram)：單一實線，有顏色用顏色，無顏色預設深灰。
        """
        import pandas as pd
        
        if getattr(self, 'gdf_railways', None) is None:
            try:
                rail_tags = {'railway': ['rail', 'subway', 'light_rail', 'tram', 'narrow_gauge', 'monorail', 'funicular']}
                raw_rail = ox.features_from_bbox(bbox=self.bbox, tags=rail_tags) if self.bbox else ox.features_from_point(self.center_point, tags=rail_tags, dist=self.dist)
                gdf_r = self._project(raw_rail)
                self.gdf_railways = gdf_r[gdf_r.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
            except Exception:
                self.gdf_railways = None

        if self.gdf_railways is None or self.gdf_railways.empty: 
            return self

        color_col = 'colour' if 'colour' in self.gdf_railways.columns else ('color' if 'color' in self.gdf_railways.columns else None)
        
        def get_valid_color(c):
            if pd.isna(c) or str(c).strip().lower() == 'nan' or not str(c).strip(): return None
            return str(c).strip().lower().split(';')[0]

        self.gdf_railways['parsed_color'] = self.gdf_railways[color_col].apply(get_valid_color) if color_col else None

        # 🌟 修正：只保留傳統鐵路為高運量 (相間線)，把 subway, monorail 踢除到一般實線分類
        high_cap_types = ['rail', 'narrow_gauge']
        is_high_cap = self.gdf_railways['railway'].isin(high_cap_types)
        
        high_cap_gdf = self.gdf_railways[is_high_cap].copy()
        other_rail_gdf = self.gdf_railways[~is_high_cap].copy()

        scale_factor = self.fig.get_figwidth() / 72.0
        base_lw = 2.5 * scale_factor
        dash_lw = 1.2 * scale_factor

        # 類型 A：都市捷運 / 中低運量軌道 (單一實線)
        if not other_rail_gdf.empty:
            other_rail_gdf['render_color'] = other_rail_gdf['parsed_color'].fillna('#555555')
            for color, group in other_rail_gdf.groupby('render_color'):
                group.plot(ax=self.ax, color=color, linewidth=base_lw * 0.7, zorder=3.5, alpha=0.9)

        # 類型 B：傳統重軌 (黑底 + 專屬色虛線)
        if not high_cap_gdf.empty:
            high_cap_gdf['dash_color'] = high_cap_gdf['parsed_color'].fillna('#ffffff')
            high_cap_gdf.plot(ax=self.ax, color='#18181b', linewidth=base_lw, zorder=3.6)
            for color, group in high_cap_gdf.groupby('dash_color'):
                group.plot(ax=self.ax, color=color, linewidth=dash_lw, linestyle=(0, (4, 4)), zorder=3.7)

        return self

    def draw_grid(self, rows=5, cols=5, color="#ffffff", alpha=0.15, linewidth=0.8):
        if self.ax is None: raise ValueError("Canvas not found.")
        x_steps = [self.minx + i * (self.maxx - self.minx) / cols for i in range(cols + 1)]
        y_steps = [self.miny + j * (self.maxy - self.miny) / rows for j in range(rows + 1)]

        for i in range(cols + 1):
            self.ax.plot([x_steps[i], x_steps[i]], [self.miny, self.maxy], color=color, alpha=alpha, linewidth=linewidth, linestyle='--', zorder=5.0)
            if i < cols: self.ax.text(x_steps[i] + (self.maxx - self.minx) / (cols * 2.0), self.maxy, chr(65 + i), color=color, alpha=0.7, fontsize=9, ha='center', va='bottom', zorder=5.0, fontfamily='monospace')

        for j in range(rows + 1):
            self.ax.plot([self.minx, self.maxx], [y_steps[j], y_steps[j]], color=color, alpha=alpha, linewidth=linewidth, linestyle='--', zorder=5.0)
            if j < rows: self.ax.text(self.maxx, y_steps[j] + (self.maxy - self.miny) / (rows * 2.0), str(rows - j), color=color, alpha=0.7, fontsize=9, ha='left', va='center', zorder=5.0, fontfamily='monospace')

        return self

    def draw_scale(self, x_offset=0.08, y_offset=0.08, color="white", font_size=9):
        if self.ax is None or self.fig is None: raise ValueError("Canvas not found.")

        minx, miny, maxx, maxy = self.minx, self.miny, self.maxx, self.maxy
        self.fig.canvas.draw() if self.fig.canvas else None
        
        map_width_cm, map_height_cm = _get_actual_ax_size_cm(self.fig, self.ax)
        ground_width_m, ground_height_m = _compute_ground_metrics(minx, miny, maxx, maxy, self.crs)

        scale_ratio_x = ground_width_m / (map_width_cm * 0.01)
        scale_ratio_y = ground_height_m / (map_height_cm * 0.01)

        unit_per_cm_x = (maxx - minx) / map_width_cm 
        unit_per_cm_y = (maxy - miny) / map_height_cm 
        ground_m_per_cm_x = scale_ratio_x * 0.01
        ground_m_per_cm_y = scale_ratio_y * 0.01

        bar_cm = 2.0
        bar_len_x = unit_per_cm_x * bar_cm
        bar_len_y = unit_per_cm_y * bar_cm

        anchor_x = maxx - (maxx - minx) * x_offset
        anchor_y = miny + (maxy - miny) * y_offset
        start_x, start_y = anchor_x - bar_len_x, anchor_y + bar_len_y

        half_len_x, half_m_x, total_m_x = bar_len_x / 2.0, ground_m_per_cm_x * 1.0, ground_m_per_cm_x * 2.0
        self.ax.plot([start_x, anchor_x], [anchor_y, anchor_y], color=color, linewidth=2.0, zorder=10)
        tick_h = (maxy - miny) * 0.004
        for cx in [start_x, start_x + half_len_x, anchor_x]:
            self.ax.plot([cx, cx], [anchor_y - tick_h, anchor_y + tick_h], color=color, linewidth=1.5, zorder=10)

        self.ax.text(start_x, anchor_y - (tick_h * 1.8), "0", color=color, fontsize=font_size, ha='center', va='top', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(start_x + half_len_x, anchor_y - (tick_h * 1.8), f"{half_m_x:.0f}m", color=color, fontsize=font_size, ha='center', va='top', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(anchor_x, anchor_y - (tick_h * 1.8), f"{total_m_x:.0f}m", color=color, fontsize=font_size, ha='center', va='top', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(anchor_x - half_len_x, anchor_y + (tick_h * 2.5), f"H Scale: 1:{int(scale_ratio_x):,}", color=color, fontsize=font_size, ha='center', va='bottom', zorder=10, fontfamily='monospace', fontweight='bold')

        half_len_y, half_m_y, total_m_y = bar_len_y / 2.0, ground_m_per_cm_y * 1.0, ground_m_per_cm_y * 2.0
        self.ax.plot([anchor_x, anchor_x], [anchor_y, start_y], color=color, linewidth=2.0, zorder=10)
        tick_w = (maxx - minx) * 0.004
        for cy in [anchor_y, anchor_y + half_len_y, start_y]:
            self.ax.plot([anchor_x - tick_w, anchor_x + tick_w], [cy, cy], color=color, linewidth=1.5, zorder=10)

        self.ax.text(anchor_x + (tick_w * 1.8), anchor_y, "0", color=color, fontsize=font_size, ha='left', va='center', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(anchor_x + (tick_w * 1.8), anchor_y + half_len_y, f"{half_m_y:.0f}m", color=color, fontsize=font_size, ha='left', va='center', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(anchor_x + (tick_w * 1.8), start_y, f"{total_m_y:.0f}m", color=color, fontsize=font_size, ha='left', va='center', zorder=10, fontfamily='monospace', fontweight='bold')
        self.ax.text(anchor_x + (tick_w * 4.5), anchor_y + half_len_y, f"V Scale: 1:{int(scale_ratio_y):,}", color=color, fontsize=font_size, ha='center', va='center', rotation=90, zorder=10, fontfamily='monospace', fontweight='bold')

        north_x, north_y, arrow_size = anchor_x - half_len_x, start_y + (unit_per_cm_y * 0.8), unit_per_cm_y * 0.5
        self.ax.annotate('', xy=(north_x, north_y + arrow_size), xytext=(north_x, north_y), arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=8), zorder=10)
        self.ax.text(north_x, north_y + arrow_size + (arrow_size * 0.3), "N", color=color, fontsize=font_size + 2, ha='center', va='bottom', zorder=10, fontfamily='sans-serif', fontweight='bold')

        return self

    def render(self):
        if self.ax is None: raise ValueError("Canvas not found.")
        self.ax.set_axis_off()
        return self
    
    def draw_airports(self):
        if getattr(self, 'gdf_aeroway', None) is None:
            try:
                tags = {'aeroway': ['runway', 'taxiway', 'apron', 'helipad']}
                raw_aero = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_aeroway = self._project(raw_aero)
            except Exception:
                self.gdf_aeroway = None

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
            taxiways = line_gdf[line_gdf['aeroway'] == 'taxiway']
            runways = line_gdf[line_gdf['aeroway'] == 'runway']

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

                # 🌟 修正 3：機場標籤 zorder 提高，確保不會被道路覆蓋
                if row.get('aeroway') == 'runway':
                    fontsize, fontcolor, fontweight, z = int(12 * scale_factor), '#ffffff', 'bold', 4.5
                    bbox_style = dict(facecolor='#3f3f46', alpha=0.8, edgecolor='none', boxstyle='square,pad=0.1')
                else:
                    fontsize, fontcolor, fontweight, z = int(7 * scale_factor), '#fde047', 'bold', 4.5
                    bbox_style = dict(facecolor='#71717a', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1')

                if any(math.hypot(x - px, y - py) < (self.maxx - self.minx) * 0.005 for px, py in placed_markings): continue
                placed_markings.append((x, y))

                self.ax.text(x, y, str(row['ref']).strip(), color=fontcolor, fontsize=fontsize, ha='center', va='center', rotation=angle, zorder=z, clip_on=True, fontfamily='sans-serif', fontweight=fontweight, bbox=bbox_style)

        return self
    
if __name__ == "__main__":
    bbox = (139.617014, 35.543913, 139.967247, 35.773251)
    
    # 🌟 修正 5：不再由外部提供 ax，而是交由 setup_canvas 內部透過比例完美計算並動態生成！
    plotter_utm = MapPlotter(
        bbox=bbox,
        admin_widths=(0, 0, 0, 3.5, 0, 0, 2.5, 0, 0.5, 0),
        projection="utm"
    )
    
    # 指定橫向寬度 base_width=96，縱向將根據真實地理比例完美計算
    (plotter_utm.setup_canvas(base_width=96)
                .draw_land()
                .draw_water()
                .draw_airports()
                .draw_roads()
                .draw_scale()
                .draw_admins()
                .draw_railways()
                .draw_highway_refs()
                .draw_exits()
                .draw_buildings()
                .draw_greenery()
                .draw_stations()
                .draw_piers()
                .draw_coordinates()
                .draw_grid(10, 10, linewidth=3) 
                .draw_admin_labels(levels=(4,6,7,8,9))
                .render())
                
    decorate_map_layout(plotter_utm.fig, plotter_utm.ax, plotter_utm, "Tokyo Metropoliton Sub-area Map")
    plotter_utm.fig.patch.set_facecolor("#091c15")
    plt.savefig("tokyo_map_3.png", dpi=200, facecolor=plotter_utm.fig.get_facecolor(), edgecolor='none')