from generic_plotter import GenericMapPlotter
from osm_cache import OSMDataCache
from map_styles import *
from japan_plotter import *
import logging

if __name__ == "__main__":
    import osmnx as ox
    ox.settings.overpass_url = "http://localhost:8081/api"
    ox.settings.overpass_rate_limit = False
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s@%(filename)s:%(lineno)s] -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bbox = (
        140.344532, 
        35.736127, 
        140.409755,
        35.820426, 
    )

    style = JapanMapStyle()
    style.roads.widths_pt["motorway"] = 5.0
    style.japan_stations.corridor_width_m = 8.0
    style.japan_railways.heavy_width_pt = 6.0

    style.roads.adaptive_widths = True
    style.roads.reference_scale = 25_000
    style.roads.width_scale_exponent = 0.50

    plotter = JapanMapPlotter(
        bbox=bbox,
        style=style,
        projection="utm",
        cache_enabled=True,
        cache_ttl_days=30,
        query_padding=0.1
    )
    
    # 1. Setup canvas first so render_context is established
    plotter.setup_canvas(base_width=16)

    # 2. Optionally calculate optimal rows and cols (cells >= 4cm, square-like)
    opt_rows, opt_cols = plotter.calculate_optimal_grid(min_cell_cm=4.0)

    (
        plotter.setup_canvas(base_width=16)
        .draw_land()
        .draw_greenery()
        .draw_buildings()
        .draw_water()
        .draw_piers()
        .draw_airports()
        .draw_roads()
        .draw_railways()
        .draw_stations()
        .draw_highway_refs()
        .draw_exits()
        .draw_scale()
        .draw_admins()
        .draw_grid(opt_rows, opt_cols, linewidth=2.0)
        .draw_coordinates()
        .draw_landuse()
        .draw_admin_labels(
            levels=(4, 6, 7, 8, 9),
            #include_place_nodes=True,  # Automatically fills rural/suburban node labels
        )
        .render()
    )

    decorate_map_layout(
        plotter.fig,
        plotter.ax,
        plotter,
        "Tokyo Metropolitan Sub-area Map",
    )

    plotter.fig.patch.set_facecolor("#091c15")

    logger.info("Saving map to disk...")
    plt.savefig(
        "output_maps/tokyo_map_v2.png",
        dpi=200,
        facecolor=plotter.fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.25,
    )
    logger.info("Map generation process complete!")