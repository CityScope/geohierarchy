"""Where bulk array work runs: on a CUDA GPU when one is usable and the job is big enough to pay for the transfer, otherwise on the CPU (NumPy).

    array_module(n)   CuPy when a CUDA device works and `n` >= GPU_MIN_ROWS, else NumPy
    to_numpy(a)       a NumPy array whatever `a` is

Same switch as `transitlos.flex.backend`: environment variable TRANSITLOS_DEVICE = cpu | cuda | auto (default auto). CuPy is imported lazily; a missing package, driver or
device falls back to the CPU, never to an error.
"""
from __future__ import annotations

import os

import numpy as np

GPU_MIN_ROWS = 2_000_000   # below this the host<->device copies cost more than the arithmetic
_state: dict = {}


def _cuda_ok() -> bool:
    if "ok" not in _state:
        ok = False
        if os.environ.get("TRANSITLOS_DEVICE", os.environ.get("FLEX_BACKEND", "auto")).lower() in ("auto", "cuda", "gpu", "cupy"):
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() > 0:
                    cp.zeros(1)
                    ok = True
            except Exception:  # noqa: BLE001
                ok = False
        _state["ok"] = ok
    return _state["ok"]


def array_module(n: int = 0):
    """The array library to use for ``n`` rows: CuPy when ``n`` reaches ``GPU_MIN_ROWS`` and a working CUDA device is
    found, NumPy otherwise.

    Args:
        n: Number of rows the caller will process.

    Returns:
        The ``cupy`` or ``numpy`` module.
    """
    if n >= GPU_MIN_ROWS and _cuda_ok():
        import cupy as cp

        return cp
    return np


def to_numpy(a):
    """Bring an array back to the host as a NumPy array (CuPy arrays are copied from the GPU).

    Args:
        a: A NumPy or CuPy array, or anything ``np.asarray`` accepts.

    Returns:
        A NumPy array.
    """
    return a.get() if hasattr(a, "get") else np.asarray(a)
