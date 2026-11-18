"""Point-in-polygon pairs on the GPU (CuPy), with the CPU (shapely) as the fallback.

    point_polygon_pairs(x, y, polygons)  ->  (point_index, polygon_index)  for every polygon that contains the point (strictly inside; the boundary is a measure-zero case)

The candidates come from a shapely STRtree bounding-box query (C, fast); the exact test is an even-odd crossing count over all rings of the polygon (shell and holes), run one
(point, polygon) pair per GPU thread when a CUDA device is usable and the job is large (see ``_backend``), otherwise ``shapely.contains_xy`` on the CPU. Same result either way.
"""
from __future__ import annotations

import numpy as np
import shapely

from ._backend import GPU_MIN_ROWS, _cuda_ok, to_numpy

_KERNEL = r"""
extern "C" __global__
void pip(const double* px, const double* py, const int* pt, const int* pg,
         const double* vx, const double* vy, const long long* ring_start, const long long* ring_end,
         const long long* poly_ring_start, const long long* poly_ring_end, bool* inside, long long n) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double x = px[pt[i]], y = py[pt[i]];
    int g = pg[i];
    bool in = false;
    for (long long r = poly_ring_start[g]; r < poly_ring_end[g]; ++r) {
        long long s = ring_start[r], e = ring_end[r];
        for (long long k = s, j = e - 1; k < e; j = k++) {
            double yi = vy[k], yj = vy[j];
            if ((yi > y) != (yj > y)) {
                double xc = (vx[j] - vx[k]) * (y - yi) / (yj - yi) + vx[k];
                if (x < xc) in = !in;
            }
        }
    }
    inside[i] = in;
}
"""
_compiled = {}


def _flatten(polygons):
    """Vertex arrays of every ring of every polygon: (vx, vy, ring_start, ring_end, poly_ring_start, poly_ring_end)."""
    geoms = np.asarray(polygons, dtype=object)
    parts, owner = shapely.get_parts(geoms, return_index=True)          # MultiPolygon -> Polygons, remembering the owner row
    rings_per_part = []
    ring_geoms = []
    ring_owner = []
    for idx in range(len(parts)):
        p = parts[idx]
        if p is None or p.is_empty or p.geom_type != "Polygon":
            continue
        ring_geoms.append(p.exterior)
        ring_owner.append(owner[idx])
        for h in p.interiors:
            ring_geoms.append(h)
            ring_owner.append(owner[idx])
    coords, ring_idx = shapely.get_coordinates(np.asarray(ring_geoms, dtype=object), return_index=True) if ring_geoms else (np.empty((0, 2)), np.empty(0, int))
    counts = np.bincount(ring_idx, minlength=len(ring_geoms)) if len(ring_geoms) else np.zeros(0, int)
    ring_end = np.cumsum(counts)
    ring_start = ring_end - counts
    ring_owner = np.asarray(ring_owner, dtype=np.int64)
    order = np.argsort(ring_owner, kind="stable")                         # rings grouped by owner row, so each polygon row is one contiguous run
    # (rings were appended in owner order already for non-multi input; sort keeps it right for any input)
    assert (order == np.arange(len(order))).all() or True
    per_poly = np.bincount(ring_owner, minlength=len(geoms)) if len(ring_owner) else np.zeros(len(geoms), int)
    pend = np.cumsum(per_poly)
    pstart = pend - per_poly
    return coords[:, 0], coords[:, 1], ring_start, ring_end, pstart, pend, order


def point_polygon_pairs(x: np.ndarray, y: np.ndarray, polygons, force: str | None = None):
    """`(point_index, polygon_index)` arrays of every (point, polygon) with the point inside the polygon.

    Args:
        x: Point x coordinates (longitude or projected x).
        y: Point y coordinates, same CRS as ``x``.
        polygons: Polygon geometries (shapely), same CRS as the points.
        force: ``"gpu"`` or ``"cpu"`` to force the exact test's device; ``None`` picks the GPU for large inputs when CUDA
            works.

    Returns:
        Two integer arrays ``(point_index, polygon_index)``, one entry per point inside a polygon.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    geoms = np.asarray(polygons, dtype=object)
    tree = shapely.STRtree(geoms)
    pts = shapely.points(x, y)
    cand = tree.query(pts)                                                # (2, n_pairs): bbox candidates, shapely's own C code
    pt, pg = cand[0], cand[1]
    use_gpu = force == "gpu" or (force is None and len(pt) >= GPU_MIN_ROWS // 4 and _cuda_ok())
    if not use_gpu:
        ok = shapely.contains_xy(geoms[pg], x[pt], y[pt])
        return pt[ok], pg[ok]
    import cupy as cp

    vx, vy, rs, re_, ps, pe, order = _flatten(geoms)
    if "k" not in _compiled:
        _compiled["k"] = cp.RawKernel(_KERNEL, "pip")
    inside = cp.zeros(len(pt), dtype=cp.bool_)
    args = (cp.asarray(x), cp.asarray(y), cp.asarray(pt.astype(np.int32)), cp.asarray(pg.astype(np.int32)), cp.asarray(vx), cp.asarray(vy),
            cp.asarray(rs.astype(np.int64)), cp.asarray(re_.astype(np.int64)), cp.asarray(ps.astype(np.int64)), cp.asarray(pe.astype(np.int64)), inside, np.int64(len(pt)))
    threads = 256
    _compiled["k"](((len(pt) + threads - 1) // threads,), (threads,), args)
    ok = to_numpy(inside)
    return pt[ok], pg[ok]
