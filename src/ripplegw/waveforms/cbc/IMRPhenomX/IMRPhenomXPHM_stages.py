"""IMRPhenomXPHM split into stages, for mixed precision and batched evaluation.

``generate_xphm`` (IMRPhenomXPHM.py) is the reference implementation; the
stages below call the same per-binary functions (coprecessing_setup,
precession_frame, precession_angles, coprecessing_modes, twist_and_sum), so
the physics lives in one place. The four stages can run in different
precisions:

=========  ==============================================================  ==========
stage      content                                                          "mixed"
=========  ==============================================================  ==========
setup      per binary: masses, amp0, MSA initialisation, pWF22 (+ NNLO      float64
           final spin), J-frame angles, MSA precession constants
angles     per frequency x emm = 1..4: MSA Euler angles (exact or           float64
           interpolated), NNLO angles on MSA_ERROR (``msa_fallback``)
modes      co-precessing XHM modes h_lm: the per-binary part of the         split
           XHM evaluation (mode coefficients, small linear solves) in
           float64, the per-frequency part in float32
twist      Wigner-d rotation, sum over modes, polarisation rotation         float32
=========  ==============================================================  ==========

``precision="float32"`` runs every stage in float32 (also with x64 enabled),
``precision="float64"`` every stage in float64 (same result as
generate_xphm to rounding). Before a float32 twist, alpha and epsilon are
reduced modulo 2 pi in float64, after the NNLO fallback (exact: the twist uses
only exp(i k alpha), exp(-i k epsilon) with integer k); for some MSA
configurations they reach 1e7. generate_xphm_batch evaluates the NNLO
fallback only for the MSA_ERROR binaries of a batch.

Why this split: float32 fails in the per-binary MSA setup (NaN where the
spin-magnitude roots are nearly degenerate), in the per-binary coefficients of
the (3,2) mode (badly conditioned linear solves) and in the per-frequency MSA
angles (double root of the S^2 cubic); the per-frequency mode evaluation and
the twist-up are accurate in float32.

Validated setting. The accuracy figures and the interpolation settings
(256 nodes, fallback threshold 1 rad, at most 5 % of a batch recomputed) were
tuned and validated for batched generation (batch 1024-16384) on a BBH box --
chirp mass 15-150 Msun (log-uniform), component masses 10-120 Msun, q >= 1/8,
spin magnitudes <= 0.99 with isotropic directions, f_ref = 20 Hz -- on a
20-1024 Hz grid (uniform at 0.125 Hz, or multibanded with 1358 points).
Observed over 12288 draws (sample maxima, not bounds), mismatch against
float64 (aLIGO-PSD-weighted inner product, no time/phase maximisation):

* "mixed", exact angles: max 7e-7;
* "mixed" + 256 nodes + fallback: max 8.4e-5, p99 9e-8, 5 of 12288 draws (4e-4)
  above 1e-5, about 1.7 % of binaries flagged. The largest error comes from a
  draw the flag does not catch (q ~ 1, large in-plane spins, irregularity
  0.04 rad), so in much larger samples a few draws above 1e-4 are expected.

Other applications (other mass/spin ranges, lower f_min, other grids, small
batches, gradients) may need other settings and should re-validate.
"""

import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float

import ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession as pPrec
from ripplegw.typing import FloatLike
from ripplegw.utils import mixed_precision as mp
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import (
    coprecessing_modes,
    coprecessing_setup,
    precession_angles,
    precession_frame,
    twist_and_sum,
)
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import lal_M_sec

Precision = Literal["float64", "mixed", "float32"]
_EMMS = (1, 2, 3, 4)
# (setup, angles, modes, twist) bits; "split" = per-binary f64 / per-frequency f32
_STAGE_BITS = {
    "float64": (64, 64, 64, 64),
    "mixed": (64, 64, "split", 32),
    "float32": (32, 32, 32, 32),
}
# generate_xphm_batch without interpolation fallback: the MSA_ERROR binaries are
# recomputed individually up to K = max(1, ceil(fraction * batch)) (see its docstring)
_MSA_FALLBACK_MAX_FRACTION = 0.01


# --------------------------------------------------------------------- stages
def stage_setup(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    distance: FloatLike,
    inclination: FloatLike,
    phi0: FloatLike,
    reference_frequency: float,
    msa_fallback: bool = True,
) -> dict[str, Any]:
    """Per-binary quantities (coprecessing_setup + precession_frame)."""
    spins = (chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    amp0, pWF22, msa_error = coprecessing_setup(
        mass_1, mass_2, *spins, distance, reference_frequency, msa_fallback=msa_fallback
    )
    theta_JN, zeta_pol, msa_setup, mf_cut = precession_frame(
        mass_1, mass_2, *spins, phi0, inclination, reference_frequency
    )
    return {
        "M_sec": lal_M_sec(mass_1, mass_2),
        "amp0": amp0,
        "pWF22": pWF22,
        "msa_error": msa_error,
        "theta_JN": theta_JN,
        "zeta_pol": zeta_pol,
        "msa_setup": msa_setup,
        "mf_cut": mf_cut,
        "params": (mass_1, mass_2, *spins, phi0, inclination),
    }


def stage_angles(
    Mf: Float[Array, " n_freq"],
    setup: dict[str, Any],
    reference_frequency: float,
    angle_interp_nodes: int | None = None,
    msa_fallback: bool = True,
) -> tuple:
    """(theta_JN, zeta_pol, alpha, epsilon, cos_beta) for emm = 1..4 (precession_angles)."""
    m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, phi0, inc = setup["params"]
    return precession_angles(
        Mf,
        m1,
        m2,
        c1x,
        c1y,
        c1z,
        c2x,
        c2y,
        c2z,
        phi0,
        inc,
        reference_frequency,
        setup["theta_JN"],
        setup["zeta_pol"],
        setup["msa_setup"],
        msa_fallback=msa_fallback,
        angle_interp_nodes=angle_interp_nodes,
    )


def stage_modes(
    Mf: Float[Array, " n_freq"], pWF22: dict[str, Any]
) -> Complex[Array, "5 n_freq"]:
    """Co-precessing modes (21, 22, 32, 33, 44) without amp0 (coprecessing_modes)."""
    return coprecessing_modes(Mf, pWF22)


def stage_twist(
    Mf: Float[Array, " n_freq"],
    hlm: Complex[Array, "5 n_freq"],
    angles: tuple,
    amp0: FloatLike,
    mf_cut: FloatLike,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """twist_and_sum with the amplitude prefactor applied to the modes."""
    theta_JN, zeta_pol, a, e, c = angles
    return twist_and_sum(Mf, hlm * amp0, a, e, c, theta_JN, zeta_pol, mf_cut)


# split_by_frequency(stage_modes) per (grid shape, pWF22 structure); bounded.
_SPLIT_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_SPLIT_CACHE_SIZE = 8


def _modes_split(
    Mf64: Float[Array, " n_freq"], pWF22: dict[str, Any]
) -> Complex[Array, "5 n_freq"]:
    """stage_modes with its frequency-independent part in float64, the rest in float32."""
    key = (Mf64.shape, str(jax.tree_util.tree_structure(pWF22)))
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = mp.split_by_frequency(stage_modes, Mf64, pWF22)
        while len(_SPLIT_CACHE) > _SPLIT_CACHE_SIZE:
            _SPLIT_CACHE.popitem(last=False)
    return _SPLIT_CACHE[key](Mf64.astype(jnp.float32), pWF22)


def _wrap(angles: tuple) -> tuple:
    theta_JN, zeta_pol, a, e, c = angles
    two_pi = 2 * jnp.pi
    return theta_JN, zeta_pol, jnp.remainder(a, two_pi), jnp.remainder(e, two_pi), c


def _check_precision(precision: str) -> tuple:
    if precision not in _STAGE_BITS:
        raise ValueError(
            f"precision must be one of {tuple(_STAGE_BITS)}, got {precision!r}"
        )
    if precision != "float32":
        mp.require_x64(f'precision="{precision}"')
    return _STAGE_BITS[precision]


def _modes_and_twist(Mf64, S, angles, bits):
    """Modes and twist stages of one binary in the requested precisions."""
    _, _, b_modes, b_twist = bits
    # Exact; changes only XLA's fusion. Without it XLA fuses the angle computation
    # into the twist kernels and moves more data: float64 batch on an A100 (MFD,
    # batch 4096) 124k -> 141k wf/s, bytes accessed 12.3 -> 9.0 GB. Also covers the
    # fallback branches of generate_xphm_batch (their lax.cond had the same effect).
    angles = jax.lax.optimization_barrier(angles)
    if b_twist == 32:
        angles = _wrap(angles)  # in float64, after NNLO, before the cast to float32
    if b_modes == "split":
        hlm = _modes_split(Mf64, S["pWF22"])
    else:
        hlm = mp.run_in(b_modes, stage_modes, Mf64, S["pWF22"])
    return mp.run_in(b_twist, stage_twist, Mf64, hlm, angles, S["amp0"], S["mf_cut"])


def _setup(params11, reference_frequency, bits, msa_fallback):
    return mp.run_in(
        bits[0],
        lambda *q: stage_setup(*q, reference_frequency, msa_fallback),
        *params11,
    )


def _setup_and_grid(params11, frequency_array, reference_frequency, bits, msa_fallback):
    S = _setup(params11, reference_frequency, bits, msa_fallback)
    Mf_a = mp.cast(frequency_array, bits[1]) * mp.cast(S["M_sec"], bits[1])
    if bits[2] == "split":
        Mf64 = jnp.asarray(frequency_array, jnp.float64) * S["M_sec"]
    else:
        Mf64 = Mf_a
    return S, Mf_a, Mf64


# --------------------------------------------------------------------- per binary
def generate_xphm_staged(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    distance: FloatLike,
    inclination: FloatLike,
    phi0: FloatLike,
    frequency_array: Float[Array, " n_freq"],
    reference_frequency: float,
    precision: Precision = "mixed",
    msa_fallback: bool = True,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """One IMRPhenomXPHM waveform through the staged path, exact precession angles.

    Same arguments as generate_xphm plus ``precision``. The output dtype is
    that of the twist stage (complex64 for "mixed" and "float32").
    """
    bits = _check_precision(precision)
    p = (
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        distance,
        inclination,
        phi0,
    )
    S, Mf_a, Mf64 = _setup_and_grid(
        p, frequency_array, reference_frequency, bits, msa_fallback
    )
    ang = mp.run_in(
        bits[1],
        lambda m, s: stage_angles(m, s, reference_frequency, None, msa_fallback),
        Mf_a,
        S,
    )
    return _modes_and_twist(Mf64, S, ang, bits)


# --------------------------------------------------------------------- batch
def generate_xphm_batch(
    params: Float[Array, "batch 11"],
    frequency_array: Float[Array, " n_freq"],
    reference_frequency: float,
    precision: Precision = "float64",
    angle_interp_nodes: int | None = None,
    angle_interp_fallback: bool = True,
    angle_interp_fallback_threshold: float = 1.0,
    angle_interp_fallback_max_fraction: float = 0.05,
    msa_fallback: bool = True,
    _fallback_max_rows: int | None = None,
) -> tuple[
    Complex[Array, "batch n_freq"],
    Complex[Array, "batch n_freq"],
    Array,
    Bool[Array, ""],
]:
    """A batch of IMRPhenomXPHM waveforms; optional mixed precision and interpolated angles.

    The defaults are exact (float64, exact angles). The fast setting validated in
    the module docstring is ``precision="mixed", angle_interp_nodes=256`` (with the
    default fallback); outside that setting, re-validate.

    Every binary is first computed without the NNLO fallback, with the
    interpolated angles if ``angle_interp_nodes`` is set. The binaries that need
    more are then recomputed individually, in one pass, with exact angles and
    ``msa_fallback``:

    * NNLO fallback (``msa_fallback``, as in generate_xphm): the MSA_ERROR
      binaries (~0.2 % of the validated box) get the NNLO setup and angles.
      Under vmap a per-binary selection (jnp.where, as in generate_xphm) would
      evaluate the NNLO branch for every binary.
    * Interpolation fallback (``angle_interp_nodes`` set, ``angle_interp_fallback``
      True): binaries whose node angles are irregular
      (pPrec.angle_node_irregularity > ``angle_interp_fallback_threshold`` rad,
      NaN counts as irregular) get the exact MSA angles. MSA_ERROR binaries are
      not counted as flagged when ``msa_fallback`` is set (they get the NNLO
      angles anyway).

    Up to K binaries are recomputed (static size K): K =
    ceil(``angle_interp_fallback_max_fraction`` * batch) with the interpolation
    fallback, else max(1, ceil(0.01 * batch)). If more than K need it, the whole
    batch is recomputed with exact angles and ``msa_fallback`` instead
    (``overflow`` = True; a batch-level lax.cond, so this costs nothing unless it
    happens). The result is the same either way: no flagged binary keeps
    interpolated angles, and with exact angles the batch equals generate_xphm to
    rounding.

    Args:
        params: [batch, 11] = (mass_1, mass_2, chi1x, chi1y, chi1z, chi2x, chi2y,
            chi2z, distance [Mpc], inclination, phi0), the argument order of
            generate_xphm.
        frequency_array: Frequencies in Hz.
        reference_frequency: Reference frequency in Hz.
        precision: "float64" (default), "mixed" or "float32".
        angle_interp_nodes: None (default, exact angles) or the number of
            interpolation nodes for the MSA angles (validated: 256).
        angle_interp_fallback: Recompute flagged binaries exactly (default True).
        angle_interp_fallback_threshold: Flag threshold in rad (default 1).
        angle_interp_fallback_max_fraction: Size of the individual recomputation
            with the interpolation fallback, K = ceil(fraction * batch) (default
            0.05; it also covers the MSA_ERROR binaries).
        msa_fallback: NNLO angles where the MSA system fails (as generate_xphm).
        _fallback_max_rows: Overrides K (for tests).

    Returns:
        hp, hc of shape [batch, n_freq], n_flagged (int, binaries flagged by the
        interpolation fallback; MSA_ERROR binaries not included) and overflow
        (bool, more than K binaries needed a recomputation and the whole batch
        was recomputed). For jax.jit, all arguments except ``params`` and
        ``frequency_array`` must be static (functools.partial).
    """
    bits = _check_precision(precision)
    B = params.shape[0]
    interp_fallback = angle_interp_nodes is not None and angle_interp_fallback
    if _fallback_max_rows is not None:
        K = int(_fallback_max_rows)
    elif interp_fallback:
        K = max(1, math.ceil(angle_interp_fallback_max_fraction * B))
    else:
        K = max(1, math.ceil(_MSA_FALLBACK_MAX_FRACTION * B))
    if K < 1:
        raise ValueError(f"_fallback_max_rows must be >= 1, got {K}")
    f = jnp.asarray(frequency_array)

    def setup(t, nnlo):
        return _setup([t[i] for i in range(11)], reference_frequency, bits, nnlo)

    def angles(Mf, s, n, nnlo):
        return mp.run_in(
            bits[1],
            lambda m, q: stage_angles(m, q, reference_frequency, n, nnlo),
            Mf,
            s,
        )

    # Main pass: no NNLO fallback, interpolated angles if requested.
    S, Mf_a, Mf64 = jax.vmap(
        lambda t: _setup_and_grid(
            [t[i] for i in range(11)], f, reference_frequency, bits, False
        )
    )(params)
    ang = jax.vmap(lambda m, s: angles(m, s, angle_interp_nodes, False))(Mf_a, S)
    n_flagged = jnp.zeros((), jnp.int32)
    overflow = jnp.zeros((), bool)
    need = jnp.zeros((B,), bool)
    if interp_fallback:

        def irregularity(m, s):
            return pPrec.angle_node_irregularity(m, _EMMS, s, angle_interp_nodes)

        irr = jax.vmap(lambda m, s: mp.run_in(bits[1], irregularity, m, s))(
            Mf_a, S["msa_setup"]
        )
        need = ~(irr <= angle_interp_fallback_threshold)
        if msa_fallback:
            need = need & ~S["msa_error"]
        n_flagged = jnp.sum(need).astype(jnp.int32)
    if msa_fallback:
        need = need | S["msa_error"]
    if interp_fallback or msa_fallback:
        overflow = jnp.sum(need) > K

        def recompute(t, m):  # setup and exact angles with the NNLO fallback
            s = setup(t, msa_fallback)
            return s, angles(m, s, None, msa_fallback)

        def rows(sa):
            idx = jnp.nonzero(need, size=K, fill_value=B)[0]  # B: out of range, dropped
            take = jnp.minimum(idx, B - 1)
            new = jax.vmap(recompute)(params[take], Mf_a[take])
            return jax.tree_util.tree_map(
                lambda full, x: full.at[idx].set(x, mode="drop"), sa, new
            )

        # Overflow: everything with exact angles (interpolated without the
        # interpolation fallback, as in the main pass) and the NNLO fallback.
        n_all = None if interp_fallback else angle_interp_nodes

        def everything(sa):
            s = jax.vmap(lambda t: setup(t, msa_fallback))(params)
            a = jax.vmap(lambda m, q: angles(m, q, n_all, msa_fallback))(Mf_a, s)
            return s, a

        S, ang = jax.lax.cond(overflow, everything, rows, (S, ang))
    hp, hc = jax.vmap(lambda m, s, a: _modes_and_twist(m, s, a, bits))(Mf64, S, ang)
    return hp, hc, n_flagged, overflow
