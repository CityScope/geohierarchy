"""geohierarchy: hierarchical spatial data management with controlled column propagation."""

from .core import GeoHierarchy
from .aggregation import (
    AggregationStrategy,
    Sum,
    Mean,
    Max,
    Min,
    SmoothMean,
    aggregation_strategy,
)
from .exceptions import ColumnNotFoundError, AggregationStrategyError
from .utils import h3_cells, get_knn_mapping, get_id_mapping
from .edges import edges_to_level, edges_to_h3_by_distance
from .raster_resample import raster_to_polygons, raster_to_h3
from .spatial_cache import (
    build_raster_h3_mapping,
    apply_raster_h3_mapping,
    build_h3_to_polygon_mapping,
    build_street_to_h3_mapping,
    compose_street_to_polygon_mapping,
    build_census_level_mapping,
    raster_grid_key,
    geometry_version_key,
    load_mapping,
    save_mapping,
)

__all__ = [
    "GeoHierarchy",
    "AggregationStrategy",
    "Sum",
    "Mean",
    "Max",
    "Min",
    "SmoothMean",
    "aggregation_strategy",
    "ColumnNotFoundError",
    "AggregationStrategyError",
    "h3_cells",
    "get_knn_mapping",
    "get_id_mapping",
    "edges_to_level",
    "edges_to_h3_by_distance",
    "raster_to_polygons",
    "raster_to_h3",
    "build_raster_h3_mapping",
    "apply_raster_h3_mapping",
    "build_h3_to_polygon_mapping",
    "build_street_to_h3_mapping",
    "compose_street_to_polygon_mapping",
    "build_census_level_mapping",
    "raster_grid_key",
    "geometry_version_key",
    "load_mapping",
    "save_mapping",
]
