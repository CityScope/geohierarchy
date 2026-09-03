"""H3J/H3T tile generation - SLOW, for backward compatibility only.

These formats are NOT recommended for new code. Use PMTiles with WKB geometry instead.
H3J/H3T creates many files and doesn't scale well for large datasets.

Use fast_tiles.py (PMTiles) for all new development.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import polars as pl


def generate_h3j_from_h3_indices(
    pl_df,
    output_path: Union[str, Path],
    h3_index_col: str = "h3_cell",
    property_cols: Optional[Sequence[str]] = None,
    resolution: Optional[int] = None,
    **kwargs: Any,
) -> Path:
    """Generate H3J file from Polars DataFrame with H3 indices.

    H3J stores only the H3 indices and properties - NO geometry.
    Client-side rendering generates hexagon polygons from indices.

    WARNING: This format is SLOW and doesn't scale well for large datasets.
    Use generate_pmtiles_from_geoparquet() with WKB geometry instead.

    Args:
        pl_df: Polars DataFrame with H3 index column
        output_path: Output .h3j file path
        h3_index_col: Name of the H3 index column
        property_cols: Properties to include (default: all except h3_index_col)
        resolution: H3 resolution (for metadata)
        **kwargs: Additional metadata

    Returns:
        Path to the generated .h3j file

    Example:
        generate_h3j_from_h3_indices(
            pl_df,
            "h3_data.h3j",
            h3_index_col="h3_cell",
            property_cols=["population", "access_score"]
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select columns, excluding binary (WKB geometry) columns
    if property_cols is None:
        property_cols = [col for col in pl_df.columns if col != h3_index_col]

    # Filter out binary columns (like WKB geometry)
    cols_to_export = [h3_index_col] + [
        c
        for c in property_cols
        if not (hasattr(pl_df[c], "dtype") and pl_df[c].dtype == pl.Binary)
    ]

    # Build H3J structure
    h3j = {
        "metadata": {"name": output_path.stem, "h3_resolution": resolution, **kwargs},
        "cells": pl_df.select(cols_to_export).to_dicts(),
    }

    with open(output_path, "w") as f:
        json.dump(h3j, f)

    return output_path


def generate_h3t_tiles_from_h3_indices(
    pl_df,
    output_dir: Union[str, Path],
    h3_index_col: str = "h3_cell",
    property_cols: Optional[Sequence[str]] = None,
    min_zoom: int = 0,
    max_zoom: int = 15,
    resolution: Optional[int] = None,
    buffer_tiles: int = 1,
) -> Path:
    """Generate H3T tiled format from H3 indices using Polars.

    H3T is the tiled version of H3J - each tile file contains only
    the H3 cells that intersect that tile.

    WARNING: This format is SLOW and creates MANY files (doesn't scale well).
    Use generate_pmtiles_from_geoparquet() with WKB geometry instead.

    Args:
        pl_df: Polars DataFrame with H3 index column
        output_dir: Directory for {z}/{x}/{y}.h3t files
        h3_index_col: Name of the H3 index column
        property_cols: Properties to include
        min_zoom: Minimum tile zoom
        max_zoom: Maximum tile zoom
        resolution: H3 resolution
        buffer_tiles: Number of buffer tiles around each H3 cell

    Returns:
        Path to the output directory
    """
    import h3
    import mercantile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if property_cols is None:
        property_cols = [col for col in pl_df.columns if col != h3_index_col]

    # Filter out binary columns (like WKB geometry)
    safe_property_cols = [
        c
        for c in property_cols
        if not (hasattr(pl_df[c], "dtype") and pl_df[c].dtype == pl.Binary)
    ]

    # Columns to export (excluding binary)
    cols_to_export = [h3_index_col] + safe_property_cols

    # Get all unique H3 indices and their properties
    h3_data = pl_df.select(cols_to_export)

    # Extract unique H3 indices
    unique_h3 = h3_data[h3_index_col].unique().to_list()

    # Compute center lat/lng for all cells
    # This is the only part that uses a for loop, but it's O(n) not O(n*z)
    centers = []
    for idx in unique_h3:
        try:
            lat, lng = h3.cell_to_latlng(idx)
            centers.append({"h3_cell": idx, "lat": lat, "lng": lng})
        except Exception:
            continue

    if not centers:
        return output_dir

    centers_df = pl.DataFrame(centers)

    # For each zoom level, compute tile assignments and write files
    for z in range(min_zoom, max_zoom + 1):
        # Compute tile coordinates for all centers at this zoom
        tile_assignments = []
        for row in centers_df.iter_rows(named=True):
            try:
                tile = mercantile.tile(row["lng"], row["lat"], z)
                tile_assignments.append(
                    {"h3_cell": row["h3_cell"], "z": z, "x": tile.x, "y": tile.y}
                )
            except Exception:
                continue

        if not tile_assignments:
            continue

        tiles_df = pl.DataFrame(tile_assignments)

        # Group by tile coordinates
        grouped = tiles_df.group_by(["z", "x", "y"])

        # For each tile group, collect all H3 cells and their properties
        for group in grouped:
            z_val, x_val, y_val = group[0]  # (z, x, y)
            tile_h3_cells = group[1]["h3_cell"].to_list()

            # Get properties for these cells from original data
            cells_in_tile = h3_data.filter(pl.col(h3_index_col).is_in(tile_h3_cells))
            cells_list = cells_in_tile.to_dicts()

            # Write H3T file
            tile_dir = output_dir / str(z_val) / str(x_val)
            tile_dir.mkdir(parents=True, exist_ok=True)
            tile_path = tile_dir / f"{y_val}.h3t"

            h3t_data = {"cells": cells_list}
            with open(tile_path, "w") as f:
                json.dump(h3t_data, f)

    return output_dir
