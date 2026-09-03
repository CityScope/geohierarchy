"""Fast vector tile generation using freestiler (Rust-powered).

This module provides the FASTEST methods for vector tile generation:
- freestile_file() for GeoParquet files (10-100x faster)
- freestile_query() for DuckDB-based tiling
- H3 PMTiles generation from WKB geometry in GeoParquet

All outputs are PMTiles (.pmtiles) for optimal static hosting.

For H3 cells WITH WKB geometry: Use generate_pmtiles_from_geoparquet() directly.
For H3 cells WITHOUT geometry: Use generate_pmtiles_from_h3_with_centers().

The old slow H3J/H3T implementation is in h3_slow.py (NOT RECOMMENDED).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

try:
    from freestiler import (
        freestile,
        freestile_file,
        freestile_query,
        freestile_h3,
    )

    HAS_FREESTILER = True
except ImportError:
    HAS_FREESTILER = False
    freestile = None
    freestile_file = None
    freestile_query = None


def generate_pmtiles_from_geoparquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    layer_name: str,
    min_zoom: int = 0,
    max_zoom: int = 14,
    property_cols: Optional[Sequence[str]] = None,
    tile_format: str = "mvt",
    **kwargs: Any,
) -> Path:
    """Generate PMTiles directly from a GeoParquet file.

    This is the FASTEST method for geometry data - no Python geometry processing,
    the Rust engine reads the GeoParquet directly.

    For H3 cells with WKB geometry, this is the recommended approach.
    No geometry computation is needed - uses existing geometry from the file.

    Args:
        input_path: Path to GeoParquet file
        output_path: Output .pmtiles file path
        layer_name: Name of the layer in the PMTiles
        min_zoom: Minimum zoom level (default: 0)
        max_zoom: Maximum zoom level (default: 14)
        property_cols: Columns to include as properties (default: all non-geometry)
        tile_format: 'mvt' or 'mlt' (default: 'mvt' for compatibility)
        **kwargs: Additional arguments passed to freestile_file()

    Returns:
        Path to the generated .pmtiles file

    Example:
        generate_pmtiles_from_geoparquet(
            "census.parquet",
            "census.pmtiles",
            layer_name="census",
            max_zoom=12
        )

        # For H3 cells with WKB geometry
        generate_pmtiles_from_geoparquet(
            "h3_grid.parquet",  # Must have geometry column with WKB
            "h3_grid.pmtiles",
            layer_name="h3_grid",
            min_zoom=0, max_zoom=14
        )
    """
    if not HAS_FREESTILER:
        raise ImportError(
            "freestiler is required for fast tile generation. "
            "Install with: pip install freestiler"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build property list if not specified
    if property_cols is None:
        # freestile_file will include all columns except geometry by default
        property_cols = []

    freestile_file(
        str(input_path.absolute()),
        str(output_path),
        layer_name=layer_name,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        tile_format=tile_format,
        **kwargs,
    )

    return output_path


def generate_xyz_tiles_from_geoparquet(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    layer_name: str,
    min_zoom: int = 0,
    max_zoom: int = 14,
    property_cols: Optional[Sequence[str]] = None,
    tile_format: str = "mvt",
    extract_pmtiles: bool = True,
    overwrite: bool = True,
    quiet: bool = False,
    **kwargs: Any,
) -> Path:
    """Generate XYZ .pbf tile directory from a GeoParquet file.

    This is the main entry point for fast XYZ tile generation that works
    with folium + Leaflet.VectorGrid. It:
    1. Uses freestiler to generate a PMTiles archive (FAST)
    2. Extracts XYZ .pbf directories from the PMTiles (if extract_pmtiles=True)

    The output directory structure is: {output_dir}/{layer_name}/{z}/{x}/{y}.pbf
    which is exactly what folium + Leaflet.VectorGrid expects.

    For H3 cells with WKB geometry, this is the recommended approach.
    No geometry computation is needed - uses existing geometry from the file.

    Args:
        input_path: Path to GeoParquet file
        output_dir: Root output directory for XYZ tiles
        layer_name: Name of the layer (used for subdirectory name)
        min_zoom: Minimum zoom level (default: 0)
        max_zoom: Maximum zoom level (default: 14)
        property_cols: Columns to include as properties
        tile_format: 'mvt' or 'mlt' (default: 'mvt')
        extract_pmtiles: If True (default), extract XYZ from PMTiles.
                        If False, just generate PMTiles file and return its path.
        overwrite: Overwrite existing output
        quiet: Suppress progress messages
        **kwargs: Additional arguments passed to freestile_file()

    Returns:
        Path to the output directory containing XYZ tiles
        If extract_pmtiles=False, returns path to .pmtiles file

    Example:
        # Generate XYZ tiles for use with folium
        generate_xyz_tiles_from_geoparquet(
            "census.parquet",
            "tiles",
            layer_name="census",
            max_zoom=12
        )
        # Creates: tiles/census/0/0/0.pbf, tiles/census/1/0/0.pbf, etc.

        # For H3 cells with WKB geometry
        generate_xyz_tiles_from_geoparquet(
            "h3_grid.parquet",
            "tiles",
            layer_name="h3_grid",
            min_zoom=0, max_zoom=14
        )
    """
    if not HAS_FREESTILER:
        raise ImportError(
            "freestiler is required for fast tile generation. "
            "Install with: pip install freestiler"
        )

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # If extract_pmtiles is False, just generate PMTiles
    if not extract_pmtiles:
        pmtiles_path = output_dir / f"{layer_name}.pmtiles"
        return generate_pmtiles_from_geoparquet(
            input_path,
            pmtiles_path,
            layer_name,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            property_cols=property_cols,
            tile_format=tile_format,
            **kwargs,
        )

    # Step 1: Generate PMTiles with freestiler
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pmtiles = Path(tmpdir) / f"{layer_name}.pmtiles"

        if not quiet:
            print(f"Generating PMTiles for {layer_name}...")

        freestile_file(
            str(input_path.absolute()),
            str(tmp_pmtiles),
            layer_name=layer_name,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            tile_format=tile_format,
            quiet=quiet,
            **kwargs,
        )

        if not quiet:
            print(f"  PMTiles: {tmp_pmtiles.stat().st_size / 1024 / 1024:.2f} MB")

        # Step 2: Extract XYZ from PMTiles
        # Note: output_dir is the root tiles directory, and we want
        # output_dir/layer_name/{z}/{x}/{y}.pbf structure
        # This matches what build_city_map expects (tiles_dir/layer_name/...)
        layer_output_dir = output_dir / layer_name

        if not quiet:
            print(f"Extracting XYZ tiles to {layer_output_dir}...")

        # Import here to avoid circular imports
        from .pmtiles_to_xyz import extract_xyz_from_pmtiles

        result = extract_xyz_from_pmtiles(
            tmp_pmtiles,
            output_dir,
            layer_name=layer_name,
            overwrite=overwrite,
            quiet=quiet,
        )

        return result


def generate_pmtiles_from_polars(
    pl_df,
    output_path: Union[str, Path],
    layer_name: str,
    geometry_col: str = "geometry",
    min_zoom: int = 0,
    max_zoom: int = 14,
    tile_format: str = "mvt",
    **kwargs: Any,
) -> Path:
    """Generate PMTiles from a Polars DataFrame with geometry.

    For DataFrames that already have geometry columns (as WKB or similar).

    Args:
        pl_df: Polars DataFrame with geometry
        output_path: Output .pmtiles file path
        layer_name: Name of the layer
        geometry_col: Name of the geometry column
        min_zoom: Minimum zoom level
        max_zoom: Maximum zoom level
        tile_format: 'mvt' or 'mlt'
        **kwargs: Additional arguments

    Returns:
        Path to the generated .pmtiles file
    """
    if not HAS_FREESTILER:
        raise ImportError(
            "freestiler is required. Install with: pip install freestiler"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary GeoParquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save as GeoParquet
        pl_df.write_parquet(tmp_path)

        # Generate tiles
        freestile_file(
            str(tmp_path),
            str(output_path),
            layer_name=layer_name,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            tile_format=tile_format,
            **kwargs,
        )
        return output_path
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_pmtiles_from_h3_with_centers(
    pl_df,
    output_path: Union[str, Path],
    h3_index_col: str = "h3_cell",
    layer_name: str = "h3_cells",
    min_zoom: int = 0,
    max_zoom: int = 14,
    base_zoom: int = 12,
    tile_format: str = "mvt",
    geometry_type: str = "polygon",  # or "point"
    **kwargs: Any,
) -> Path:
    """Generate PMTiles from H3 indices by converting to polygons/points.

    This creates a multi-resolution hexagon grid where:
    - Low zooms: Coarse hexagons (aggregated)
    - High zooms: Finer hexagons or individual cells
    - base_zoom and above: Individual H3 cells as polygons or points

    Use this ONLY if your H3 data does NOT have WKB geometry.
    If your data HAS WKB geometry, use generate_pmtiles_from_geoparquet() instead (faster).

    Note: This function computes geometry for each H3 cell, which can be slow
    for large datasets. For best performance, pre-compute WKB geometry and save
    as GeoParquet, then use generate_pmtiles_from_geoparquet().

    Args:
        pl_df: Polars DataFrame with H3 indices
        output_path: Output .pmtiles file
        h3_index_col: Name of H3 index column
        layer_name: Layer name
        min_zoom: Minimum zoom
        max_zoom: Maximum zoom
        base_zoom: Zoom where individual cells appear
        tile_format: 'mvt' or 'mlt'
        geometry_type: 'polygon' for hexagons, 'point' for centroids
        **kwargs: Additional arguments to freestiler_h3

    Returns:
        Path to .pmtiles file
    """
    if not HAS_FREESTILER:
        raise ImportError(
            "freestiler is required. Install with: pip install 'freestiler[h3]'"
        )

    import h3
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert H3 indices to geometries using Polars for efficiency

    # Create a list of H3 indices
    h3_indices = pl_df[h3_index_col].to_list()

    # Compute geometries
    if geometry_type == "point":
        # Use centroids
        geometries = []
        for idx in h3_indices:
            lat, lng = h3.cell_to_lat_lng(idx)
            geometries.append(Point(lng, lat))
    else:
        # Use polygon boundaries
        geometries = []
        for idx in h3_indices:
            boundary = h3.cell_to_boundary(idx)
            geometries.append(Polygon(boundary))

    # Create GeoDataFrame
    other_cols = [c for c in pl_df.columns if c != h3_index_col]
    gdf = gpd.GeoDataFrame(
        pl_df.select(other_cols).to_pandas(), geometry=geometries, crs="EPSG:4326"
    )

    # Add H3 index as a column for aggregation
    gdf["h3_cell"] = h3_indices

    # Use freestiler_h3 for multi-resolution hexagon tiling
    # This will create layers at different H3 resolutions
    freestile_h3(
        gdf,
        str(output_path),
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        base_zoom=base_zoom,
        layer_name=layer_name,
        tile_format=tile_format,
        agg={"count": "COUNT(*)"},  # Default aggregation
        **kwargs,
    )

    return output_path


# ============================================================================
# High-level API for HierarchyMap
# ============================================================================


def build_fast_hierarchy_map(
    hierarchy,
    levels: Optional[Sequence[str]] = None,
    tiles_dir: str = "tiles",
    min_zoom: int = 0,
    max_zoom: int = 25,
    h3_geometry_type: str = "polygon",
    **kwargs: Any,
) -> Dict[str, Path]:
    """Build tiles for a GeoHierarchy using the fastest available method.

    This function intelligently selects the best method for each level:
    - For levels with WKB geometry: Uses freestiler directly on GeoParquet (FASTEST)
    - For H3 cell levels WITHOUT geometry: Uses freestiler_h3 to compute geometry
    - All outputs are single .pmtiles files

    Note: H3J/H3T formats are NOT used by default. They don't scale well.
    If you need H3J/H3T, import from h3_slow.py explicitly.

    Args:
        hierarchy: A GeoHierarchy instance
        levels: Level names to process (default: all)
        tiles_dir: Output directory for tiles
        min_zoom: Global minimum zoom
        max_zoom: Global maximum zoom
        h3_geometry_type: 'polygon' or 'point' for H3 cells without geometry
        **kwargs: Additional arguments

    Returns:
        Dictionary mapping level names to output .pmtiles paths
    """
    if not HAS_FREESTILER:
        raise ImportError(
            "freestiler is required. Install with: pip install freestiler"
        )

    import polars as pl
    from pathlib import Path

    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    level_paths = {}

    if levels is None:
        levels = list(hierarchy.geometries.keys())

    from tqdm import tqdm

    level_pbar = tqdm(
        levels, desc="[tiles] building level tiles", unit="level", mininterval=1.0
    )
    for level_name in level_pbar:
        level_pbar.set_description(f"[tiles] level={level_name}")

        # Get the data as Polars DataFrame
        # (Assuming hierarchy can provide Polars data)
        try:
            pl_df = hierarchy.get_level_as_polars(level_name)
        except (AttributeError, KeyError):
            # Fall back to GeoPandas and convert
            gdf = hierarchy.get_level(level_name)
            pl_df = pl.from_pandas(gdf)

        output_path = tiles_dir / f"{level_name}.pmtiles"

        # Check if this level has geometry (WKB) column
        has_geometry = any(
            hasattr(pl_df[col], "dtype") and pl_df[col].dtype == pl.Binary
            for col in pl_df.columns
        )

        if has_geometry:
            # Has WKB geometry - use fast GeoParquet path
            # Save to temp file and use freestiler directly
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                pl_df.write_parquet(tmp_path)
                generate_pmtiles_from_geoparquet(
                    tmp_path,
                    output_path,
                    layer_name=level_name,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            # No geometry - check if it's H3 cell data
            h3_col = (
                "h3_cell"
                if "h3_cell" in pl_df.columns
                else ("h3" if "h3" in pl_df.columns else None)
            )

            if h3_col:
                # H3 cells without geometry - compute geometry using freestiler_h3
                generate_pmtiles_from_h3_with_centers(
                    pl_df,
                    output_path,
                    h3_index_col=h3_col,
                    layer_name=level_name,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                    geometry_type=h3_geometry_type,
                    **kwargs,
                )
            else:
                # Other data without geometry - try to use freestiler directly
                generate_pmtiles_from_polars(
                    pl_df,
                    output_path,
                    layer_name=level_name,
                    geometry_col="geometry",
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                )

        level_paths[level_name] = output_path

    return level_paths
