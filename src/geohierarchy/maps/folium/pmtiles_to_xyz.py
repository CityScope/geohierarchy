"""Convert PMTiles archives to XYZ .pbf directory structure.

This module provides utilities to extract XYZ tile directories from PMTiles
archives, allowing them to be used with folium + Leaflet.VectorGrid which
expects the standard XYZ directory structure.

PMTiles format: https://github.com/protomaps/PMTiles
XYZ format expected by geohierarchy: {tiles_dir}/{level_name}/{z}/{x}/{y}.pbf
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

# Add pmtiles package to path if not already there
_pmtiles_path = str(
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _pmtiles_path not in sys.path:
    sys.path.insert(0, _pmtiles_path)

try:
    from pmtiles.convert import pmtiles_to_dir

    HAS_PMTILES_CONVERT = True
except ImportError:
    HAS_PMTILES_CONVERT = False


def extract_xyz_from_pmtiles(
    pmtiles_path: Union[str, Path],
    xyz_dir: Union[str, Path],
    layer_name: Optional[str] = None,
    min_zoom: Optional[int] = None,
    max_zoom: Optional[int] = None,
    overwrite: bool = True,
    quiet: bool = False,
) -> Path:
    """Extract XYZ .pbf tile directory from a PMTiles archive.

    This reads a PMTiles file and writes each tile to the standard XYZ
    directory structure that Leaflet.VectorGrid expects:
    {xyz_dir}/{layer_name}/{z}/{x}/{y}.pbf

    Args:
        pmtiles_path: Path to the input .pmtiles file
        xyz_dir: Output directory for XYZ tiles (will be created if needed)
        layer_name: Layer name to use in the directory structure.
                   If None, uses the PMTiles filename stem.
        min_zoom: Minimum zoom to extract (None = all zooms in PMTiles).
                  Currently not implemented - extracts all zooms.
        max_zoom: Maximum zoom to extract (None = all zooms in PMTiles).
                  Currently not implemented - extracts all zooms.
        overwrite: If True, overwrite existing XYZ directory
        quiet: Suppress progress messages

    Returns:
        Path to the output xyz_dir/layer_name

    Example:
        extract_xyz_from_pmtiles(
            "census.pmtiles",
            "tiles",
            layer_name="census",
        )
        # Creates: tiles/census/0/0/0.pbf, tiles/census/1/0/0.pbf, etc.
    """
    if not HAS_PMTILES_CONVERT:
        raise ImportError(
            "pmtiles package with convert module is required for XYZ extraction. "
            "Install with: pip install pmtiles"
        )

    pmtiles_path = Path(pmtiles_path)
    xyz_dir = Path(xyz_dir)

    if not pmtiles_path.exists():
        raise FileNotFoundError(f"PMTiles file not found: {pmtiles_path}")

    # Default layer_name to the PMTiles file stem
    if layer_name is None:
        layer_name = pmtiles_path.stem

    # The final structure is xyz_dir/layer_name/{z}/{x}/{y}.pbf
    final_layer_dir = xyz_dir / layer_name

    if final_layer_dir.exists():
        if overwrite:
            # Try to delete, but if it fails (e.g., another process is using it),
            # just continue and overwrite individual files
            try:
                shutil.rmtree(final_layer_dir)
            except (OSError, PermissionError) as e:
                # Directory might be in use by another process
                # We'll just overwrite files as we go
                if not quiet:
                    print(
                        f"  WARNING: Could not remove existing directory {final_layer_dir}: {e}"
                    )
                    print("  Will overwrite files individually")
        else:
            raise FileExistsError(
                f"XYZ layer directory already exists: {final_layer_dir}. "
                "Set overwrite=True to replace."
            )

    # Create a temporary directory for pmtiles_to_dir output
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = Path(tmpdir) / "pmtiles_output"

        # Extract using pmtiles_to_dir
        # This creates: tmp_output/{z}/{x}/{y}.mvt
        pmtiles_to_dir(str(pmtiles_path), str(tmp_output))

        # Now move and rename files to the correct structure
        # We need: xyz_dir/layer_name/{z}/{x}/{y}.pbf
        final_layer_dir.mkdir(parents=True, exist_ok=True)

        total_tiles = 0

        # Walk through the temporary output once to enumerate all tile
        # files up front, so the copy pass below can show a real
        # position/total progress bar (with ETA) instead of an unbounded
        # counter -- tile counts for a single level can run into the tens
        # or hundreds of thousands at high max_zoom, and this copy loop is
        # a genuinely slow, single-threaded Python step (unlike the Rust
        # freestiler tiling itself).
        walk_entries = [
            (root, file) for root, _dirs, files in os.walk(tmp_output) for file in files
        ]

        for root, file in tqdm(
            walk_entries,
            desc=f"[tiles] extracting {layer_name} xyz tiles",
            unit="tile",
            disable=quiet,
            mininterval=1.0,
        ):
            file_path = Path(root) / file

            # Skip metadata.json
            if file == "metadata.json":
                continue

            # Parse z, x, y from the path
            # root relative to tmp_output should be z/x
            rel_path = Path(root).relative_to(tmp_output)
            parts = list(rel_path.parts)

            if len(parts) == 2:
                # We're at z/x/ level
                z_str, x_str = parts
                y = file.split(".")[0]  # Remove .mvt extension
                y = int(y)

                # Create target directory: xyz_dir/layer_name/z/x/
                target_dir = final_layer_dir / z_str / x_str
                # Use exist_ok=True to handle concurrent writes gracefully
                target_dir.mkdir(parents=True, exist_ok=True)

                # Target file: y.pbf (note: VectorGrid uses y, not flipped)
                target_file = target_dir / f"{y}.pbf"

                # Copy and rename - handle case where target exists.
                # PMTiles archives store tiles gzip-compressed
                # (internal_compression) by default; a plain static file
                # host serving these bytes as-is under a .pbf extension
                # won't set Content-Encoding: gzip, so Leaflet.VectorGrid
                # (and any other client expecting raw MVT bytes) would
                # silently fail to decode the tile. Decompress on extract
                # so the .pbf files on disk are always raw protobuf.
                try:
                    raw = file_path.read_bytes()
                    if raw[:2] == b"\x1f\x8b":  # gzip magic
                        raw = gzip.decompress(raw)
                    target_file.write_bytes(raw)
                    total_tiles += 1
                except (FileExistsError, OSError) as e:
                    # File might already exist from a concurrent write
                    # This is fine, just skip it
                    if not quiet:
                        print(
                            f"  WARNING: Could not copy {file_path} to {target_file}: {e}"
                        )

        if not quiet:
            print(f"Extracted {total_tiles} tiles to {final_layer_dir}")

    return final_layer_dir


def extract_xyz_from_pmtiles_bulk(
    pmtiles_paths: dict[str, Union[str, Path]],
    xyz_dir: Union[str, Path],
    overwrite: bool = True,
    quiet: bool = False,
) -> dict[str, Path]:
    """Extract XYZ directories from multiple PMTiles files.

    Useful for extracting all layers at once.

    Args:
        pmtiles_paths: Dictionary mapping layer_name -> PMTiles file path
        xyz_dir: Root output directory
        overwrite: Overwrite existing directories
        quiet: Suppress progress messages

    Returns:
        Dictionary mapping layer_name -> extracted XYZ directory path
    """
    xyz_dir = Path(xyz_dir)
    xyz_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer_name, pmtiles_path in tqdm(
        list(pmtiles_paths.items()),
        desc="[tiles] extracting layers",
        unit="layer",
        disable=quiet,
        mininterval=1.0,
    ):
        result = extract_xyz_from_pmtiles(
            pmtiles_path,
            xyz_dir,
            layer_name=layer_name,
            overwrite=overwrite,
            quiet=quiet,
        )
        results[layer_name] = result

    return results
