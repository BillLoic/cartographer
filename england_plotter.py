from generic_plotter import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random, math, re
from shapely import MultiPolygon

# ==========================================
# EnglandMapPlotter: UK Geography & Suburbs
# ==========================================
class EnglandMapPlotter(GenericMapPlotter):
    """
    承繼 GenericMapPlotter，專門處理英國的行政區劃 (County/Borough/Ward)、
    單語標籤系統、傳統地名 (Town/Suburb)、以及英國鐵路與公路系統。
    """
    def __init__(self, bbox=None, center_point=None, dist=5000, 
                 custom_linewidths=None, custom_colors=None, 
                 bg_color="#8FA8DA", land_color="#a6dcca", 
                 # 英國特化層級: 5(County), 6(District), 8(Parish/Borough), 10(Ward)
                 admin_widths=(0, 0, 0, 0, 2.5, 1.5, 0, 1.0, 0, 0.5), 
                 projection="wgs84"):
        
        super().__init__(bbox, center_point, dist, custom_linewidths, custom_colors, bg_color, land_color, projection)
        
        if len(admin_widths) != 10:
            raise ValueError("admin_widths must be a tuple of length 10")
        self.admin_widths = admin_widths
        
        self.gdf_admins = {} 
        self.gdf_historic_places = None
        self.gdf_railways = None
        self.gdf_stations = None
        self.gdf_exits = None
        self.gdf_piers = None
        self._admins_fetched = False

    def _fetch_and_categorize_admins(self):
        if self._admins_fetched: return
            
        logger.info("Batch fetching UK administrative boundaries...")
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
            logger.warning(f"Failed to fetch UK boundaries: {e}")
            self._admins_fetched = True

    def draw_admins(self):
        """英國行政區：針對層級 6(區), 8(自治市), 10(選區) 進行淡色區塊渲染"""
        self._fetch_and_categorize_admins()
        for i, width in enumerate(self.admin_widths):
            level = i + 1 
            if width <= 0: continue
            
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
            
            if level in [6, 8, 10]:
                color_ids = list(range(len(gdf)))
                random.seed(42 + level); random.shuffle(color_ids)
                gdf = gdf.copy()
                gdf['color_id'] = color_ids
                alpha_val = {6: 0.04, 8: 0.08, 10: 0.15}.get(level, 0.1)
                gdf.plot(ax=self.ax, column='color_id', cmap='Set3', alpha=alpha_val, edgecolor='none', zorder=0.3 + (level * 0.001))
            
            gdf.plot(ax=self.ax, facecolor='none', edgecolor="#64748b", linewidth=width, linestyle='--', zorder=0.35 + (level * 0.01))
        return self

    def draw_admin_labels(self, min_distance_ratio=0.015):
        """英國行政區名稱：單行英文顯示"""
        import matplotlib.patheffects as pe
        if self.ax is None: raise ValueError("Canvas not found.")
        self._fetch_and_categorize_admins()
        
        target_levels = [5, 6, 8, 10]
        scale_factor = self.fig.get_figwidth() / 48.0
        font_size_map = {5: int(55 * scale_factor), 6: int(40 * scale_factor), 8: int(25 * scale_factor), 10: int(16 * scale_factor)}
        placed_labels = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio 

        for level in sorted(target_levels):
            gdf = self.gdf_admins.get(level)
            if gdf is None or gdf.empty: continue
                
            fontsize = font_size_map.get(level, int(10 * scale_factor))

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty: continue
                
                # 優先抓取 name:en，若無則抓 name
                name = str(row.get('name:en') or row.get('name')).strip()
                if not name or name.lower() in ['nan', 'none']: continue
                if level == 8: name = name.upper()

                if isinstance(geom, MultiPolygon): geom = max(geom.geoms, key=lambda p: p.area)
                center_pt = geom.centroid if geom.contains(geom.centroid) else geom.representative_point()
                
                if not (self.minx <= center_pt.x <= self.maxx and self.miny <= center_pt.y <= self.maxy): continue
                if any(math.hypot(center_pt.x - px, center_pt.y - py) < min_dist for px, py in placed_labels): continue 
                placed_labels.append((center_pt.x, center_pt.y))

                halo = [pe.withStroke(linewidth=3.5 * scale_factor, foreground="#091c15")]
                self.ax.text(center_pt.x, center_pt.y, name, color="white", fontsize=fontsize, ha='center', va='center', zorder=6.0, clip_on=True, fontfamily='sans-serif', fontweight='bold', path_effects=halo)
        return self

    def draw_historic_places(self, min_distance_ratio=0.015):
        """
        繪製英國獨特的傳統地名 (Town / Suburb)。
        使用灰色字體呈現歷史聚落的層次感。
        """
        import pandas as pd
        import matplotlib.patheffects as pe

        if getattr(self, 'gdf_historic_places', None) is None:
            try:
                tags = {'place': ['town', 'suburb', 'neighbourhood']}
                raw_places = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_historic_places = self._project(raw_places)
            except Exception:
                self.gdf_historic_places = None

        if self.gdf_historic_places is None or self.gdf_historic_places.empty: return self
        
        scale_factor = self.fig.get_figwidth() / 48.0
        fontsize = max(6, int(14 * scale_factor))
        placed_places = []
        min_dist = (self.maxx - self.minx) * min_distance_ratio

        for _, row in self.gdf_historic_places.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            
            name = str(row.get('name:en') or row.get('name')).strip()
            if not name or name.lower() in ['nan', 'none']: continue

            center_pt = geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom
            if not (self.minx <= center_pt.x <= self.maxx and self.miny <= center_pt.y <= self.maxy): continue

            if any(math.hypot(center_pt.x - px, center_pt.y - py) < min_dist for px, py in placed_places): continue
            placed_places.append((center_pt.x, center_pt.y))

            # 英國傳統地名：優雅的石板灰 (Slate Gray)，加上地圖底色光暈防遮擋
            halo = [pe.withStroke(linewidth=2.5 * scale_factor, foreground="#091c15")]
            self.ax.text(center_pt.x, center_pt.y, name, color="#94a3b8", fontsize=fontsize, ha='center', va='center', zorder=5.5, clip_on=True, fontfamily='serif', fontstyle='italic', path_effects=halo)

        logger.info("Historic towns and suburbs rendered (slate gray, italic).")
        return self

    def draw_railways(self):
        """英國軌道系統：區分 National Rail (黑白) 與 Underground/Tram (實線)"""
        import pandas as pd
        if getattr(self, 'gdf_railways', None) is None:
            try:
                tags = {'railway': ['rail', 'subway', 'light_rail', 'tram', 'narrow_gauge']}
                raw_rail = ox.features_from_bbox(bbox=self.bbox, tags=tags) if self.bbox else ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
                self.gdf_railways = self._project(raw_rail)[self._project(raw_rail).geometry.type.isin(['LineString', 'MultiLineString'])].copy()
            except Exception: self.gdf_railways = None

        if self.gdf_railways is None or self.gdf_railways.empty: return self
        
        color_col = 'colour' if 'colour' in self.gdf_railways.columns else None
        def get_valid_color(c):
            if pd.isna(c) or str(c).strip().lower() == 'nan' or not str(c).strip(): return None
            return str(c).strip().lower().split(';')[0]
        self.gdf_railways['parsed_color'] = self.gdf_railways[color_col].apply(get_valid_color) if color_col else None

        # 英國以 'rail' (National Rail) 為重軌；倫敦地鐵(subway)與輕軌(tram)為中低運量
        is_high_cap = self.gdf_railways['railway'] == 'rail'
        high_cap_gdf, other_rail_gdf = self.gdf_railways[is_high_cap].copy(), self.gdf_railways[~is_high_cap].copy()
        base_lw, dash_lw = 2.5 * (self.fig.get_figwidth() / 72.0), 1.2 * (self.fig.get_figwidth() / 72.0)

        if not other_rail_gdf.empty:
            other_rail_gdf['render_color'] = other_rail_gdf['parsed_color'].fillna('#555555')
            for color, group in other_rail_gdf.groupby('render_color'): group.plot(ax=self.ax, color=color, linewidth=base_lw * 0.7, zorder=3.5, alpha=0.9)
        if not high_cap_gdf.empty:
            high_cap_gdf.plot(ax=self.ax, color='#18181b', linewidth=base_lw, zorder=3.6)
            for color, group in high_cap_gdf.assign(dash_color=high_cap_gdf['parsed_color'].fillna('#ffffff')).groupby('dash_color'):
                group.plot(ax=self.ax, color=color, linewidth=dash_lw, linestyle=(0, (4, 4)), zorder=3.7)
        return self

    def draw_stations(self, min_distance_ratio=0.010):
        """英國車站：維持複合體幾何邏輯，但文字改為單行英文顯示"""
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
                name = str(row.get('name:en') or row.get('name')).strip()
                if not name or name.lower() in ['nan', 'none']: continue
                station_groups.setdefault(name, []).append(row.geometry)

            bbox_style = dict(facecolor='#18181b', alpha=0.85, edgecolor='#facc15', linewidth=0.6 * scale_factor, boxstyle=f'square,pad={0.15 * scale_factor}')
            for name, geoms in station_groups.items():
                merged = unary_union(geoms)
                pt = merged.centroid if merged.contains(merged.centroid) else merged.representative_point()
                if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy): continue
                if any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_stations): continue
                placed_stations.append((pt.x, pt.y))
                self.ax.text(pt.x, pt.y, name, color="#fef08a", fontsize=max(4, int(4.5 * scale_factor)), ha='center', va='center', zorder=6.5, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=bbox_style)
        return self

    def draw_highway_refs(self, min_distance_ratio=0.03):
        """英國道路編號：例如 M1, A40。極簡黑框白字"""
        import pandas as pd
        if self.gdf_edges is None or self.gdf_edges.empty: return self
        scale_factor, placed_refs, min_dist = self.fig.get_figwidth() / 72.0, [], (self.maxx - self.minx) * min_distance_ratio
        for _, row in self.gdf_edges.iterrows():
            ref = str(row.get('ref')).strip()
            if ref.lower() == 'nan' or not ref or row.get('highway') not in ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary']: continue
            geom = row.geometry
            if geom.is_empty or geom.length < (self.maxx - self.minx) * 0.01: continue
            pt = geom.interpolate(0.5, normalized=True)
            if not (self.minx <= pt.x <= self.maxx and self.miny <= pt.y <= self.maxy) or any(math.hypot(pt.x - px, pt.y - py) < min_dist for px, py in placed_refs): continue
            placed_refs.append((pt.x, pt.y))
            self.ax.text(pt.x, pt.y, ref.split(';')[0], color="white", fontsize=max(6, int(12 * scale_factor)), ha='center', va='center', zorder=4.8, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=dict(facecolor='#27272a', alpha=0.85, edgecolor='#d4d4d8', linewidth=1.0 * scale_factor, boxstyle=f'square,pad={0.25 * scale_factor}'))
        return self

    def draw_exits(self, min_distance_ratio=0.015):
        """英國高速公路交流道 (Junctions, 常常以 J10 表示)"""
        import pandas as pd
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
            # 若英國有出口編號，常加 J 前綴，如沒有則照舊
            if ref == "nan": ref = ""
            if ref.isdigit(): ref = f"J{ref}"
            self.ax.text(pt.x, pt.y, f"{ref} {name}".strip(), color="#fef08a", fontsize=max(6, int(10 * scale_factor)), ha='left', va='center', zorder=4.9, clip_on=True, fontfamily='sans-serif', fontweight='bold', bbox=dict(facecolor='#18181b', alpha=0.9, edgecolor='#facc15', linewidth=0.8 * scale_factor, boxstyle=f'square,pad={0.2 * scale_factor}'))
        return self

    def draw_piers(self):
        """英國碼頭與渡輪 (泰晤士河周邊常見)"""
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
    
if __name__ == "__main__":
    bbox = (-0.289903, 51.419445, -0.216056, 51.527777)
    
    plotter = EnglandMapPlotter(
        bbox=bbox,
        projection="utm"
    )
    
    (plotter.setup_canvas(base_width=48)
            .draw_land()
            .draw_greenery()
            .draw_buildings()
            .draw_water()
            .draw_piers()
            .draw_airports()
            .draw_roads()
            .draw_railways()
            .draw_stations()
            .draw_historic_places() # 🌟 英國特有歷史聚落
            .draw_highway_refs()
            .draw_exits()
            .draw_scale()
            .draw_admins()
            .draw_grid(10, 10, linewidth=3)
            .draw_coordinates()
            .draw_admin_labels()
            .render())
            
    decorate_map_layout(plotter.fig, plotter.ax, plotter, "London Metropoliton Sub-area Map")
    plotter.fig.patch.set_facecolor("#091c15")
    plt.savefig("London.png", dpi=200, facecolor=plotter.fig.get_facecolor(), edgecolor='none')