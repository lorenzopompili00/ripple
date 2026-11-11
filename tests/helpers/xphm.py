"""Shared inputs and tools for the IMRPhenomXPHM tests.

BBH box used throughout (``sample_bbh_box``): chirp mass 15-150 Msun
(log-uniform), component masses 10-120 Msun, q >= 1/8, spin magnitudes
<= 0.99 with isotropic directions, isotropic inclination, uniform phase,
distance 100 Mpc - 20 Gpc uniform in volume, f_ref = 20 Hz. Grids span
20-1024 Hz: uniform at 0.125 Hz, or multibanded with 1358 points.

Quick tier: one small batch (``reference()["theta"]``: 7 box draws, 2 edge cases and 2
MSA_ERROR draws) on a thinned multibanded grid.
Reference outputs of the base implementations (2c8d63b, before the
optimisations; 24fc1a1, with the NNLO fallback) are stored in
``tests/unit/waveforms/cbc/data/xphm_reference.npz`` (made by
``make_xphm_reference.py`` next to it), so the exactness tests need neither the
git history nor a compilation of the base code.

``compiled`` jits a function through ``jax.export`` and keeps the serialized
module on disk (in the JAX compilation cache directory), keyed by the ripplegw
sources: a warm run skips the ~40-80 s trace of the full model and, through
the persistent compilation cache, the compilation.
"""

import hashlib
import inspect
import os
from functools import cache
from pathlib import Path

import jax
import numpy as np

import ripplegw

F_REF = 20.0
DATA = Path(__file__).resolve().parents[1] / "unit" / "waveforms" / "cbc" / "data"
REFERENCE_FILE = DATA / "xphm_reference.npz"


# --------------------------------------------------------------------- grids and draws
def uniform_grid() -> np.ndarray:
    """20-1024 Hz at 0.125 Hz (8033 points)."""
    return 20.0 + 0.125 * np.arange(8033)


def multibanded_grid() -> np.ndarray:
    """Centres of the 1358-bin multibanded grid built on uniform_grid()."""
    nodes = np.array([20.0, 47.5, 60.25, 402.75, 585.75, 1023.75])
    df0, base_df = 0.125, 0.125
    df_bands = df0 * 2.0 ** np.arange(len(nodes) - 1)
    nbins = ((nodes[1:] - nodes[:-1]) / df_bands).astype(int)
    df = np.concatenate([np.full(n, d) for n, d in zip(nbins, df_bands)])
    lower = np.concatenate((nodes[:1], nodes[0] + np.cumsum(df[:-1])))
    upper = nodes[0] + np.cumsum(df) - base_df
    return (upper + lower) / 2


def quick_grid() -> np.ndarray:
    """Every 14th multibanded point: 97 points, 20-1018 Hz."""
    return multibanded_grid()[::14]


def sample_bbh_box(n: int, seed: int) -> np.ndarray:
    """[n, 11] draws from the BBH box (generate_xphm argument order:
    m1, m2, chi1xyz, chi2xyz, distance [Mpc], inclination, phi0).

    Masses by rejection on the component-mass bounds (batches of 4n candidates);
    the draw order is fixed, so (n, seed) always gives the same draws (e.g. the
    12288-draw validation set is sample_bbh_box(12288, seed=5)).
    """
    rng = np.random.default_rng(seed)
    mc, q = np.empty(0), np.empty(0)
    while len(mc) < n:
        m = 4 * n
        mc_ = np.exp(rng.uniform(np.log(15.0), np.log(150.0), m))
        q_ = rng.uniform(0.125, 1.0, m)
        mt_ = mc_ / (q_ / (1 + q_) ** 2) ** 0.6
        m1_, m2_ = mt_ / (1 + q_), mt_ * q_ / (1 + q_)
        keep = (m1_ <= 120) & (m2_ >= 10)
        mc, q = np.concatenate([mc, mc_[keep]]), np.concatenate([q, q_[keep]])
    mc, q = mc[:n], q[:n]
    mtot = mc / (q / (1 + q) ** 2) ** 0.6
    cols = [mtot / (1 + q), mtot * q / (1 + q)]
    for _ in range(2):
        a = rng.uniform(0.0, 0.99, n)
        cos_t = rng.uniform(-1.0, 1.0, n)
        phi = rng.uniform(0.0, 2 * np.pi, n)
        sin_t = np.sqrt(1 - cos_t**2)
        cols += [a * sin_t * np.cos(phi), a * sin_t * np.sin(phi), a * cos_t]
    inclination = np.arccos(rng.uniform(-1.0, 1.0, n))
    phase = rng.uniform(0.0, 2 * np.pi, n)
    distance = rng.uniform(100.0**3, 20000.0**3, n) ** (1 / 3)
    return np.stack([*cols, distance, inclination, phase], axis=-1)


# --------------------------------------------------------------------- reference data
@cache
def reference() -> dict:
    """Arrays of xphm_reference.npz (theta, freqs, base outputs, flags, gradients)."""
    with np.load(REFERENCE_FILE) as z:
        return {k: z[k] for k in z.files}


def multibanded_delta_f() -> np.ndarray:
    """Bin widths of multibanded_grid()."""
    nodes = np.array([20.0, 47.5, 60.25, 402.75, 585.75, 1023.75])
    return 0.125 * 2.0 ** (np.searchsorted(nodes, multibanded_grid()) - 1)


def aligo_psd(f):
    """Analytic aLIGO zero-detuned high-power PSD fit (Ajith 2011)."""
    x = np.asarray(f) / 245.4
    return 1e-48 * (
        0.0152 * x**-4 + 0.2935 * x**2.25 + 2.7951 * x**1.5 - 6.5080 * x**0.75 + 17.7622
    )


def mismatch(h1, h2, weights=None) -> np.ndarray:
    """1 - |<h1,h2>| / sqrt(<h1,h1><h2,h2>) per waveform (float64), no maximisation;
    ``weights`` (e.g. delta_f / PSD) default to a flat PSD."""
    h1, h2 = np.asarray(h1, np.complex128), np.asarray(h2, np.complex128)
    w = 1.0 if weights is None else np.asarray(weights)
    inner = np.sum(h1 * np.conj(h2) * w, axis=-1)
    norm = np.sqrt(np.sum(np.abs(h1) ** 2 * w, -1) * np.sum(np.abs(h2) ** 2 * w, -1))
    return np.abs(1.0 - np.abs(inner) / norm)


def max_rel_diff(h, h0) -> float:
    """max |h - h0| / max |h0| per waveform, maximised over the batch."""
    h, h0 = np.asarray(h), np.asarray(h0)
    scale = np.max(np.abs(h0), axis=-1, keepdims=True)
    return float(np.max(np.abs(h - h0) / scale))


# --------------------------------------------------------------------- compile once, cache on disk
@cache
def _source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(ripplegw.__file__).parent.rglob("*.py")):
        h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def _cache_dir() -> Path | None:
    d = jax.config.jax_compilation_cache_dir
    return Path(d) / "ripple_exported" if d else None


def compiled(name: str, fn, *example_args):
    """``jax.jit(fn)`` for these argument shapes, via a disk-cached ``jax.export``.

    The key covers ``name``, the source of ``fn``, the ripplegw sources, this
    file, the jax version,
    the backend, the x64 flag and the argument shapes/dtypes, so a stale
    entry is never reused. Without a cache directory this is plain jax.jit.
    """
    from jax import export

    d = _cache_dir()
    if d is None:
        return jax.jit(fn)
    sig = ",".join(f"{np.shape(a)}:{np.asarray(a).dtype}" for a in example_args)
    try:
        fn_src = inspect.getsource(fn)
    except (OSError, TypeError):
        fn_src = ""
    key = hashlib.sha256(
        f"{name}|{fn_src}|{_source_digest()}|{jax.__version__}|"
        f"{jax.default_backend()}|{jax.config.jax_enable_x64}|{sig}".encode()
    ).hexdigest()[:24]
    path = d / f"{name}-{key}.bin"
    if path.exists():
        exp = export.deserialize(bytearray(path.read_bytes()))
    else:
        exp = export.export(jax.jit(fn))(*example_args)
        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(exp.serialize())
        os.replace(tmp, path)  # atomic: safe with pytest-xdist workers
    return jax.jit(exp.call)
