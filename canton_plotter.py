from generic_plotter import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random, math, re

# ==========================================
# CantonMapPlotter: Hong Kong & Macau (SARs)
# ==========================================
# ==========================================
# CantonMapPlotter: Hong Kong & Macau SAR Geography
# ==========================================
class CantonMapPlotter(GenericMapPlotter):
    """
    承繼 GenericMapPlotter，專門處理港澳特別行政區 (SAR) 的地理特徵。
    特徵：
    1. 雙語系支援：繁體中文 + 英文 (香港) / 葡萄牙文 (澳門)。
    2. 專注於 Admin Level 5, 6 (區/堂區)。
    3. 特化圖層：海運航線 (Ferry Routes) 與 歷史/傳統地名 (Town/Suburb)。
    """
    def __init__(self, bbox=None, center_point=None, dist=5000, 
                 custom_linewidths=None, custom_colors=None, 
                 bg_color="#8FA8DA", land_color="#a6dcca", 
                 # 針對 SAR，只專注渲染 level 5 與 6 的邊界
                 admin_widths=(0, 0, 0, 0, 2.5, 1.5, 0, 0, 0, 0), 
                 projection="wgs84"):
        
        super().__init__(bbox, center_point, dist, custom_linewidths, custom_colors, bg_color, land_color, projection)
        
        if len(admin_widths) != 10:
            raise ValueError("admin_widths must be a tuple of length 10")
        self.admin_widths = admin_widths
        
        self.gdf_admins = {} 
        self.gdf_historic_places = None
        self.gdf_ferry_routes = None
        self.gdf_railways = None
        self.gdf_stations = None
        self.gdf_exits = None
        self.gdf_piers = None
        self._admins_fetched = False

    def _fetch_and_categorize_admins(self):
        if self._admins_fetched: return
            
        logger.info("Batch fetching HK/Macau administrative boundaries...")
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
        """僅針對 Level 5 (香港的區) 與 Level 6 (澳門的堂區) 進行著色"""
        self._fetch_and_categorize_admins()
        for i, width in enumerate(self.admin_widths):
            level = i + 1 
            if width <= 0: continue
            
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
            
            if level in [5, 6]:
                color_ids = list(range(len(gdf)))
                random.seed(42 + level); random.shuffle(color_ids)
                gdf = gdf.copy()
                gdf['color_id'] = color_ids
                alpha_val = 0.15 if level == 5 else 0.2
                gdf.plot(ax=self.ax, column='color_id', cmap='Pastel1', alpha=alpha_val, edgecolor='none', zorder=0.3 + (level * 0.001))
            
            gdf.plot(ax=self.ax, facecolor='none', edgecolor="#776de5", linewidth=width, linestyle='-.', zorder=0.35 + (level * 0.01))
        return self

    def draw_admin_labels(self, min_distance_ratio=0.015):
        """港澳專屬雙語標籤：繁體中文 + 英文/葡萄牙文"""
        import matplotlib.patheffects as pe
        import pandas as pd
        if self.ax is None: raise ValueError("Canvas not found.")
        self._fetch_and_categorize_admins()
        
        target_levels = [5, 6]
        scale_factor = self.fig.get_figwidth() / 48.0
        font_size_map = {5: int(45 * scale_factor), 6: int(35 * scale_factor)}
        placed_labels = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio 

        for level in sorted(target_levels):
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
                
            pri_fontsize = font_size_map.get(level, int(10 * scale_factor))
            sec_fontsize = max(int(4 * scale_factor), int(pri_fontsize * 0.5))

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty: continue
                
                # 優先抓中文 (name:zh 或 name)
                pri_name = str(row.get('name:zh') or row.get('name')).strip()
                if not pri_name or pri_name.lower() in ['nan', 'none']: continue

                # 次要語言抓英文 (香港) 或 葡萄牙文 (澳門)
                sec_name = ""
                for tag in ['name:en', 'name:pt']:
                    val = row.get(tag)
                    if pd.notna(val) and str(val).strip().lower() not in ['', 'nan', 'none']:
                        sec_name = str(val).strip(); break 

                if isinstance(geom, MultiPolygon): geom = max(geom.geoms, key=lambda p: p.area)
                center_pt = geom.centroid if geom.contains(geom.centroid) else geom.representative_point()
                
                if not (self.minx <= center_pt.x <= self.maxx and self.miny <= center_pt.y <= self.maxy): continue
                if any(math.hypot(center_pt.x - px, center_pt.y - py) < min_dist for px, py in placed_labels): continue 
                placed_labels.append((center_pt.x, center_pt.y))

                halo = [pe.withStroke(linewidth=3.5 * scale_factor, foreground="#091c15")]
                self.ax.text(center_pt.x, center_pt.y, pri_name, color="white", fontsize=pri_fontsize, ha='center', va='bottom', zorder=6.0, clip_on=True, fontfamily='sans-serif', fontweight='bold', path_effects=halo)
                if sec_name: self.ax.text(center_pt.x, center_pt.y, sec_name, color="#cbd5e1", fontsize=sec_fontsize, ha='center', va='top', zorder=6.0, clip_on=True, fontfamily='sans-serif', fontweight='normal', path_effects=halo)
        return self

    def draw_historic_places(self, min_distance_ratio=0.012):
        """
        繪製港澳傳統地名 (如：旺角, 銅鑼灣, 氹仔)。
        同為雙語顯示，字體採用典雅的灰色配置。
        """
        import pandas as pd
        import matplotlib.patheffects as pe

        if getattr(self, 'gdf_historic_places', None) is None:
            try:
                tags = {'place': ['town', 'suburb', 'neighbourhood', 'quarter']}
                raw_places = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_historic_places = self._project(raw_places)
            except Exception:
                self.gdf_historic_places = None

        if self.gdf_historic_places is None or self.gdf_historic_places.empty: return self
        
        scale_factor = self.fig.get_figwidth() / 48.0
        pri_fontsize = max(6, int(14 * scale_factor))
        sec_fontsize = max(4, int(8 * scale_factor))
        placed_places = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        for _, row in self.gdf_historic_places.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            
            pri_name = str(row.get('name:zh') or row.get('name')).strip()
            if not pri_name or pri_name.lower() in ['nan', 'none']: continue

            sec_name = ""
            for tag in ['name:en', 'name:pt']:
                val = row.get(tag)
                if pd.notna(val) and str(val).strip().lower() not in ['', 'nan', 'none']:
                    sec_name = str(val).strip(); break 

            center_pt = geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom
            if not (self.minx <= center_pt.x <= self.maxx and self.miny <= center_pt.y <= self.maxy): continue

            if any(math.hypot(center_pt.x - px, center_pt.y - py) < min_dist for px, py in placed_places): continue
            placed_places.append((center_pt.x, center_pt.y))

            halo = [pe.withStroke(linewidth=2.5 * scale_factor, foreground="#091c15")]
            self.ax.text(center_pt.x, center_pt.y, pri_name, color="#e2e8f0", fontsize=pri_fontsize, ha='center', va='bottom', zorder=5.5, clip_on=True, fontfamily='sans-serif', fontweight='bold', path_effects=halo)
            if sec_name: self.ax.text(center_pt.x, center_pt.y, sec_name, color="#94a3b8", fontsize=sec_fontsize, ha='center', va='top', zorder=5.5, clip_on=True, fontfamily='serif', fontstyle='italic', path_effects=halo)

        return self

    def draw_ferry_routes(self):
        """
        🌟 港澳特化圖層：海運航線 (天星小輪, 噴射飛航等)。
        繪製為水域上方的亮藍色虛線，並帶有發光效果。
        """
        import matplotlib.patheffects as pe
        if getattr(self, 'gdf_ferry_routes', None) is None:
            logger.info("Fetching Ferry Routes...")
            try:
                # 抓取渡輪航線
                tags = {'route': 'ferry'}
                raw_ferry = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                # 過濾出線段
                self.gdf_ferry_routes = self._project(raw_ferry)[self._project(raw_ferry).geometry.type.isin(['LineString', 'MultiLineString'])].copy()
            except Exception as e:
                logger.warning(f"Failed to fetch ferry routes: {e}")
                self.gdf_ferry_routes = None

        if self.gdf_ferry_routes is not None and not self.gdf_ferry_routes.empty:
            scale_factor = self.fig.get_figwidth() / 72.0
            
            # 使用亮天藍色，繪製虛線，並加上深藍色光暈讓它在水面上跳脫出來
            halo = [pe.withStroke(linewidth=3.0 * scale_factor, foreground="#0369a1")]
            self.gdf_ferry_routes.plot(
                ax=self.ax, 
                color='#38bdf8', 
                linewidth=1.5 * scale_factor, 
                linestyle=(0, (4, 4)), 
                zorder=0.25, # 蓋在水域(0.2)上方，但在橋梁與陸地之下
                path_effects=halo
            )
            logger.info("Ferry routes rendered.")
        return self

    # 以下直接沿用優化過的軌道、車站、交流道等繪圖邏輯
    # (為了程式碼簡潔，這些與通用都市基礎設施高度重疊，可以直接呼叫我們之前寫好的邏輯。
    # 這裡示範覆寫 draw_stations 來支援中英雙語站名)

    def draw_stations(self, min_distance_ratio=0.010):
        """港澳車站：維持複合體邏輯，標籤改為中英/中葡雙語"""
        import pandas as pd
        import geopandas as gpd
        from shapely.ops import unary_union
        if getattr(self, 'gdf_stations', None) is None:
            try:
                tags = {'public_transport': 'stop_area', 'building': 'train_station', 'railway': ['station', 'platform']}
                gdf_proj = self._project(ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist))
                is_bus = pd.Series(False, index=gdf_proj.index)
                for col in ['highway', 'amenity']:
                    if col in gdf_proj.columns: is_bus = is_bus | gdf_proj[col].isin(['bus_station', 'bus_stop'])
                self.gdf_stations = gdf_proj[~is_bus].copy()
            except Exception: self.gdf_stations = None

        if self.gdf_stations is not None and not self.gdf_stations.empty:
            scale_factor, render_geoms = self.fig.get_figwidth() / 72.0, []
            for _, row in self.gdf_stations.iterrows():
                geom = row.geometry
                if geom.is_empty or geom.geom_type in ['Point', 'MultiPoint']: continue
                render_geoms.append(geom.buffer(3.0) if geom.geom_type in ['LineString', 'MultiLineString'] else geom)

            if render_geoms:
                gpd.GeoDataFrame(geometry=[unary_union(render_geoms)], crs=self.gdf_stations.crs).plot(ax=self.ax, facecolor='#f4f4f5', edgecolor='#52525b', linewidth=1.5 * scale_factor, zorder=3.8)

            station_groups, placed_stations, min_dist = {}, [], (self.maxx - self.minx) * min_distance_ratio
            for _, row in self.gdf_stations.iterrows():
                pri = str(row.get('name:zh') or row.get('name')).strip()
                if not pri or pri.lower() in ['nan', 'none']: continue
                
                sec = ""
                for tag in ['name:en', 'name:pt']:
                    val = row.get(tag)
                    if pd.notna(val) and str(val).strip().lower() not in ['', 'nan', 'none']:
                        sec = str(val).strip(); break
                        
                if not sec: continue
                station_groups.setdefault((pri, sec), []).append(row.geometry)

            bbox_style = dict(facecolor='#18181b', alpha=0.85, edgecolor='#facc15', linewidth=0.6 * scale_factor, boxstyle=f'square,pad={0.15 * scale_factor}')
            for (pri, sec), geoms in station_groups.items():
                merged = unary_union(geoms)
                pt = merged.centroid if merged.contains(merged.centroid) else merged.representative_point()
                if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy): continue
                if any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_stations): continue
                placed_stations.append((pt.x, pt.y))
                self.ax.text(pt.x, pt.y, f"{pri}\n{sec}", color="#fef08a", fontsize=max(4, int(4.5 * scale_factor)), ha='center', va='center', zorder=6.5, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=bbox_style, linespacing=1.2)
        return self
    
    def draw_piers(self):
        """碼頭與渡輪"""
        import pandas as pd
        if getattr(self, 'gdf_piers', None) is None:
            try:
                self.gdf_piers = self._project(ox.features_from_bbox(bbox=self.bbox, tags={'amenity': 'ferry_terminal', 'man_made': 'pier'}) if self.bbox else ox.features_from_point(self.center_point, tags={'amenity': 'ferry_terminal', 'man_made': 'pier'}, dist=self.dist))
            except Exception: self.gdf_piers = None
        if self.gdf_piers is None or self.gdf_piers.empty: return self
        placed_piers, min_dist = [], (self.maxx - self.minx) * 0.015
        for _, row in self.gdf_piers.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            pt = geom.centroid if not (geom.geom_type in ['Polygon', 'MultiPolygon'] and not geom.contains(geom.centroid)) else geom.representative_point()
            if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy) or any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_piers): continue
            placed_piers.append((pt.x, pt.y))
            self.ax.text(pt.x, pt.y, "⚓", color="#0ea5e9", fontsize=max(8, int(15 * (self.fig.get_figwidth() / 72.0))), ha='center', va='center', zorder=6.4, clip_on=True, fontfamily='DejaVu Sans')
        return self
    
    def draw_railways(self):
        if getattr(self, 'gdf_railways', None) is None:
            try:
                tags = {'railway': ['rail', 'subway', 'light_rail', 'tram', 'narrow_gauge', 'monorail', 'funicular']}
                raw_rail = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                gdf_r = self._project(raw_rail)
                self.gdf_railways = gdf_r[gdf_r.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
            except Exception:
                self.gdf_railways = None

        if self.gdf_railways is None or self.gdf_railways.empty: return self
        color_col = 'colour' if 'colour' in self.gdf_railways.columns else ('color' if 'color' in self.gdf_railways.columns else None)
        
        def get_valid_color(c):
            if pd.isna(c) or str(c).strip().lower() == 'nan' or not str(c).strip(): return None
            return str(c).strip().lower().split(';')[0]

        self.gdf_railways['parsed_color'] = self.gdf_railways[color_col].apply(get_valid_color) if color_col else None
        high_cap_gdf, other_rail_gdf = self.gdf_railways[self.gdf_railways['railway'].isin(['rail', 'narrow_gauge'])], self.gdf_railways[~self.gdf_railways['railway'].isin(['rail', 'narrow_gauge'])]
        base_lw, dash_lw = 2.5 * (self.fig.get_figwidth() / 72.0), 1.2 * (self.fig.get_figwidth() / 72.0)

        if not other_rail_gdf.empty:
            other_rail_gdf['render_color'] = other_rail_gdf['parsed_color'].fillna('#555555')
            for color, group in other_rail_gdf.groupby('render_color'): group.plot(ax=self.ax, color=color, linewidth=base_lw * 0.7, zorder=3.5, alpha=0.9)
        if not high_cap_gdf.empty:
            high_cap_gdf.plot(ax=self.ax, color='#18181b', linewidth=base_lw, zorder=3.6)
            for color, group in high_cap_gdf.assign(dash_color=high_cap_gdf['parsed_color'].fillna('#ffffff')).groupby('dash_color'):
                group.plot(ax=self.ax, color=color, linewidth=dash_lw, linestyle=(0, (4, 4)), zorder=3.7)
        return self
    
    def draw_highway_refs(self, min_distance_ratio=0.03):
        if self.gdf_edges is None or self.gdf_edges.empty: return self
        scale_factor, placed_refs, min_dist = self.fig.get_figwidth() / 72.0, [], (self.maxx - self.minx) * min_distance_ratio
        for _, row in self.gdf_edges.iterrows():
            ref = str(row.get('ref')).strip()
            if ref.lower() == 'nan' or not ref or row.get('highway') not in ['motorway', 'motorway_link', 'trunk', 'trunk_link']: continue
            geom = row.geometry
            if geom.is_empty or geom.length < (self.maxx - self.minx) * 0.01: continue
            pt = geom.interpolate(0.5, normalized=True)
            if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy) or any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_refs): continue
            placed_refs.append((pt.x, pt.y))
            self.ax.text(pt.x, pt.y, ref.split(';')[0], color="white", fontsize=max(6, int(12 * scale_factor)), ha='center', va='center', zorder=4.8, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=dict(facecolor='#27272a', alpha=0.85, edgecolor='#d4d4d8', linewidth=1.0 * scale_factor, boxstyle=f'square,pad={0.25 * scale_factor}'))
        return self
    
    def draw_exits(self, min_distance_ratio=0.015):
        if getattr(self, 'gdf_exits', None) is None:
            try:
                self.gdf_exits = self._project(ox.features_from_bbox(bbox=self.bbox, tags={'highway': 'motorway_junction'}) if self.bbox else ox.features_from_point(self.center_point, tags={'highway': 'motorway_junction'}, dist=self.dist))
            except Exception: self.gdf_exits = None
        if self.gdf_exits is None or self.gdf_exits.empty: return self
        scale_factor, placed_exits, min_dist = self.fig.get_figwidth() / 72.0, [], (self.maxx - self.minx) * min_distance_ratio
        for _, row in self.gdf_exits.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            pt = geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom
            if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy): continue
            name, ref = str(row.get('name', '')).strip(), str(row.get('ref', '')).strip()
            if not (name.lower() != 'nan' and name) and not (ref.lower() != 'nan' and ref): continue
            if any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_exits): continue
            placed_exits.append((pt.x, pt.y))
            self.ax.text(pt.x, pt.y, f"{ref if ref.lower()!='nan' else ''} {name if name.lower()!='nan' else ''}".strip(), color="#fef08a", fontsize=max(6, int(10 * scale_factor)), ha='left', va='center', zorder=4.9, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=dict(facecolor='#18181b', alpha=0.9, edgecolor='#facc15', linewidth=0.8 * scale_factor, boxstyle=f'square,pad={0.2 * scale_factor}'))
        return self

    # 由於港澳的高速公路與鐵路渲染邏輯與 JapanMapPlotter 大致相同 (黑白重軌、一般公路網)，
    # (在實務架構上可以將它們上提回 GenericMapPlotter 或透過 Mixin 引入)
    
if __name__ == "__main__":
    # 香港維多利亞港周邊 (九龍 + 港島北部)
    bbox_hk = (114.135, 22.270, 114.246846, 22.349612)
    
    plotter = CantonMapPlotter(
        bbox=bbox_hk,
        projection="utm" 
    )
    
    (plotter.setup_canvas(base_width=96)
            .draw_land()
            .draw_greenery()
            .draw_buildings()
            .draw_water()
            .draw_ferry_routes() # 🌟 特化圖層：天星小輪等航線！
            .draw_piers()        # 碼頭錨點
            .draw_airports()
            .draw_roads()
            .draw_railways()     # 港鐵系統
            .draw_stations()     # 雙語車站
            .draw_historic_places() # 🌟 如: "旺角 Mong Kok", "銅鑼灣 Causeway Bay"
            # .draw_highway_refs() # (若有需要可掛載)
            # .draw_exits()        # (若有需要可掛載)
            .draw_scale()
            .draw_admins()
            .draw_grid(10, 10, linewidth=3)
            .draw_coordinates()
            .draw_admin_labels() # 🌟 雙語行政區名 (如: 油尖旺區 Yau Tsim Mong District)
            .render())
            
    decorate_map_layout(plotter.fig, plotter.ax, plotter, "Hong Kong Victoria Harbour Map")
    plotter.fig.patch.set_facecolor("#091c15")
    plt.savefig("hong_kong_map.png", dpi=200, facecolor=plotter.fig.get_facecolor(), edgecolor='none')