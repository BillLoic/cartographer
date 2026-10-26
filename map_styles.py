"""
map_styles.py

Externalized cartographic styling engine.

All visual parameters (colors, stroke widths in points, font sizes in points,
z-orders, and physical default dimensions in metres) are defined here as
structured dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


# ======================================================================
# Generic Base & Infrastructure Styles
# ======================================================================

@dataclass
class BaseLayerStyle:
    """Colors, opacities, and z-orders for fundamental physical layers."""

    background: str = "#d1e2f8"          # Water / sea canvas background
    land: str = "#f8fafc"                # Base land polygon
    water: str = "#d1e2f8"               # Rivers, lakes, bays
    greenery: str = "#86efac"            # Parks, forests, golf courses
    greenery_alpha: float = 0.35

    building_residential: str = "#cbd5e1"
    building_residential_alpha: float = 0.45
    building_commercial: str = "#fca5a5"
    building_commercial_alpha: float = 0.65

    land_zorder: float = 0.10
    greenery_zorder: float = 0.15
    residential_building_zorder: float = 0.17
    commercial_building_zorder: float = 0.18
    water_zorder: float = 0.20


@dataclass
class RoadStyle:
    """
    Road network stroke widths (in points), colors, and scale-adaptive policy.
    Uses pandas.Series for fast vectorized mapping on GeoDataFrames.
    """

    widths_pt: pd.Series = field(
        default_factory=lambda: pd.Series(
            {
                "motorway": 4.0,
                "motorway_link": 3.0,
                "trunk": 3.5,
                "trunk_link": 2.5,
                "primary": 2.5,
                "primary_link": 2.0,
                "secondary": 1.5,
                "secondary_link": 1.0,
                "tertiary": 1.0,
                "tertiary_link": 1.0,
                "residential": 1.0,
                "unclassified": 1.0,
                "living_street": 1.0,
            },
            dtype=float,
        )
    )

    colors: pd.Series = field(
        default_factory=lambda: pd.Series(
            {
                "motorway": "#e63946",
                "motorway_link": "#c934ea",
                "trunk": "#f4a261",
                "trunk_link": "#ff6f00",
                "primary": "#e9c46a",
                "primary_link": "#e9c46a",
                "secondary": "#2a9d8f",
                "secondary_link": "#2a9d8f",
                "tertiary": "#19567c",
                "tertiary_link": "#19567c",
                "residential": "#64748b",
                "unclassified": "#64748b",
                "living_street": "#64748b",
            },
            dtype=object,
        )
    )

    fallback_width_pt: float = 0.5
    fallback_color: str = "#666666"

    tunnel_color: str = "#ff9999"
    bridge_color: str = "#ff9999"

    tunnel_zorder: float = 1.0
    ground_zorder: float = 2.0
    bridge_zorder: float = 3.0

    # Cartographic scale-adaptive policy
    adaptive_widths: bool = False
    reference_scale: float = 25_000.0
    width_scale_exponent: float = 0.50
    minimum_width_pt: float = 0.25
    maximum_width_pt: float = 18.0


@dataclass
class AirportStyle:
    """
    Airport surface and centerline styling.
    Runway/taxiway surface widths are physical (metres); centerlines are visual (points).
    """

    apron_color: str = "#d4d4d8"
    taxiway_surface_color: str = "#71717a"
    runway_surface_color: str = "#3f3f46"

    default_runway_width_m: float = 45.0
    default_taxiway_width_m: float = 15.0

    taxiway_centerline_color: str = "#fde047"
    taxiway_centerline_width_pt: float = 0.8

    runway_centerline_color: str = "#ffffff"
    runway_centerline_width_pt: float = 1.5

    apron_zorder: float = 0.40
    taxiway_zorder: float = 0.45
    runway_zorder: float = 0.46
    taxiway_centerline_zorder: float = 0.47
    runway_centerline_zorder: float = 0.48
    label_zorder: float = 4.90

    runway_label_font_size_pt: float = 11.0
    taxiway_label_font_size_pt: float = 7.5
    label_collision_ratio: float = 0.005


@dataclass
class GridStyle:
    """Alphanumeric reference grid styling."""

    color: str = "#4f46e5"       # High-contrast indigo for light land
    alpha: float = 0.30
    linewidth_pt: float = 1.0
    font_size_pt: float = 10.0
    zorder: float = 5.0


@dataclass
class CoordinateStyle:
    """Geographic DMS border ticks and labels."""

    color: str = "#f8fafc"
    halo_width_pt: float = 3.0   # Dark stroke halo ensures visibility on any background
    line_width_pt: float = 1.2
    font_size_pt: float = 9.0
    font_family: str = "monospace"
    zorder: float = 8.0
    text_zorder: float = 8.5


@dataclass
class ScaleBarStyle:
    """Physical map scale bar and North indicator."""

    x_offset_ratio: float = 0.08
    y_offset_ratio: float = 0.08
    bar_ground_length_m: float = 200.0
    color: str = "#0f172a"       # Dark slate so it stands out on light backgrounds
    font_size_pt: float = 10.0
    thick_line_width_pt: float = 2.2
    thin_line_width_pt: float = 1.5
    zorder: float = 10.0


@dataclass
class MapStyle:
    """Master style container for GenericMapPlotter."""

    base: BaseLayerStyle = field(default_factory=BaseLayerStyle)
    roads: RoadStyle = field(default_factory=RoadStyle)
    airports: AirportStyle = field(default_factory=AirportStyle)
    grid: GridStyle = field(default_factory=GridStyle)
    coordinates: CoordinateStyle = field(default_factory=CoordinateStyle)
    scale: ScaleBarStyle = field(default_factory=ScaleBarStyle)


# ======================================================================
# Japan-Specific Styles
# ======================================================================

@dataclass
class JapanAdminStyle:
    """Japanese administrative boundaries (Prefecture -> Ward -> Chome)."""

    boundary_widths_pt: tuple[float, ...] = (
        0.0, 0.0, 0.0, 3.5, 0.0, 3, 2.5, 0.0, 0.6, 0.0
    )
    fill_alpha: dict[int, float] = field(
        default_factory=lambda: {7: 0.05, 8: 0.20, 9: 0.40}
    )
    fill_colormap: str = "Pastel2"
    boundary_color: str = "#64748b"
    boundary_linestyle: str = "-."
    boundary_alpha: float = 0.85

    # Tuned for 16-inch+ high-resolution canvases
    label_font_sizes_pt: dict[int, float] = field(
        default_factory=lambda: {
            4: 112,
            6: 84,
            7: 60,
            8: 45,
            9: 34,
        }
    )
    label_secondary_ratio: float = 0.65
    label_color: str = "#1e293b"
    label_y_offset_ratio: float = 0.0012
    
    place_font_sizes_pt: dict[str, float] = field(
        default_factory=lambda: {
            "town": 16.0,
            "suburb": 14.0,
            "borough": 14.0,
            "village": 13.0,
            "quarter": 11.5,
            "neighbourhood": 22.0,
            "hamlet": 9.0,
            "isolated_dwelling": 8.5,
        }
    )


@dataclass
class JapanRailwayStyle:
    """Japanese heavy rail (JR/Private) and urban transit (Subway/Monorail)."""

    default_color: str = "#555555"
    urban_width_pt: float = 4.5
    urban_zorder: float = 3.5
    alpha: float = 0.90

    heavy_base_color: str = "#18181b"
    heavy_width_pt: float = 6.0
    heavy_zorder: float = 3.6

    heavy_dash_color: str = "#ffffff"
    heavy_dash_width_pt: float = 3.0
    dash_pattern: tuple[float, float] = (4.0, 4.0)
    heavy_dash_zorder: float = 3.7


@dataclass
class JapanStationStyle:
    """Station complex footprints and bilingual signboards."""

    corridor_width_m: float = 6.0  # Physical width in metres
    footprint_facecolor: str = "#f4f4f5"
    footprint_edgecolor: str = "#52525b"
    footprint_linewidth_pt: float = 1.5
    footprint_zorder: float = 3.8

    label_collision_ratio: float = 0.010
    label_facecolor: str = "#18181b"
    label_alpha: float = 0.88
    label_edgecolor: str = "#facc15"
    label_linewidth_pt: float = 0.8
    label_pad: float = 0.20
    label_color: str = "#fef08a"
    label_font_size_pt: float = 8.5
    label_zorder: float = 6.5


@dataclass
class JapanLabelStyle:
    """Highway reference shields, IC/JCT exits, and ferry piers."""

    highway_collision_ratio: float = 0.030
    highway_facecolor: str = "#15803d"  # Classic Japanese expressway green
    highway_edgecolor: str = "#ffffff"
    highway_linewidth_pt: float = 1.0
    highway_font_size_pt: float = 9.5
    highway_zorder: float = 4.8

    exit_collision_ratio: float = 0.015
    exit_facecolor: str = "#18181b"
    exit_edgecolor: str = "#facc15"
    exit_linewidth_pt: float = 0.8
    exit_font_size_pt: float = 8.5
    exit_zorder: float = 4.9

    pier_collision_ratio: float = 0.015
    pier_color: str = "#0284c7"
    pier_font_size_pt: float = 14.0
    pier_zorder: float = 6.4


@dataclass
class JapanMapStyle(MapStyle):
    """Master style container for JapanMapPlotter."""

    japan_admin: JapanAdminStyle = field(default_factory=JapanAdminStyle)
    japan_railways: JapanRailwayStyle = field(default_factory=JapanRailwayStyle)
    japan_stations: JapanStationStyle = field(default_factory=JapanStationStyle)
    japan_labels: JapanLabelStyle = field(default_factory=JapanLabelStyle)