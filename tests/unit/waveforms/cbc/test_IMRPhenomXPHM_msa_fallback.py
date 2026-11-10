"""IMRPhenomXPHM MSA -> NNLO fallback (``msa_fallback``, default True).

LAL's default precession version (223, MSA angles) falls back to the NNLO
angles (version 102) when the MSA system fails to initialise
(|Omega_z^(5)| > 1000).  ripple does the same by default; ``msa_fallback=False``
gives the MSA angles for every binary (ripple's behaviour before the fallback).
The fixture ``msa_error_draws.json`` holds 27 such draws from a BBH prior box
plus 6 control draws.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomX_NNLO as pNNLO
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import build_pWF22
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import (
    IMRPhenomXPHM,
    generate_xphm,
)
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
)

F_REF = 20.0
FREQS = np.arange(20.0, 1024.0, 0.5)

_DRAWS = json.loads((Path(__file__).parent / "msa_error_draws.json").read_text())
FLAGGED = np.asarray(_DRAWS["msa_error"])
CONTROL = np.asarray(_DRAWS["control"])


def _batched(**kwargs):
    fn = lambda theta, f: generate_xphm(*theta, f, F_REF, **kwargs)
    return jax.jit(jax.vmap(fn, in_axes=(0, None)))


def _run(**kwargs):
    theta = jnp.asarray(np.concatenate([FLAGGED, CONTROL]))
    hp, hc = _batched(**kwargs)(theta, jnp.asarray(FREQS))
    return np.asarray(hp), np.asarray(hc)


@pytest.fixture(scope="module")
def fallback_waveforms():
    """Default call (msa_fallback=True) on the flagged then the control draws."""
    return _run()


@pytest.fixture(scope="module")
def msa_only_waveforms():
    """msa_fallback=False on the flagged then the control draws."""
    return _run(msa_fallback=False)


def _mismatch(hp, hc, lp, lc):
    """Phase-maximised mismatch of (hp, hc) concatenated, flat noise."""
    a = np.concatenate([hp, hc])
    b = np.concatenate([lp, lc])
    ov = np.abs(np.vdot(b, a)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return 1.0 - ov


def _lal(theta, prec_version):
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, dist, incl, phi = theta
    pars = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(pars, prec_version)
    ma = ls.SimInspiralCreateModeArray()
    for ell, m in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
        ls.SimInspiralModeArrayActivateMode(ma, ell, m)
        ls.SimInspiralModeArrayActivateMode(ma, ell, -m)
    ls.SimInspiralWaveformParamsInsertModeArray(pars, ma)
    fv = lal.CreateREAL8Vector(len(FREQS))
    fv.data = FREQS
    hp, hc = ls.SimInspiralChooseFDWaveformSequence(
        phi,
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        c1x,
        c1y,
        c1z,
        c2x,
        c2y,
        c2z,
        F_REF,
        dist * 1e6 * lal.PC_SI,
        incl,
        pars,
        ls.IMRPhenomXPHM,
        fv,
    )
    return hp.data.data.copy(), hc.data.data.copy()


def test_msa_error_flag():
    flag = jax.vmap(
        lambda th: pNNLO.msa_error_from_init(
            IMRPhenomX_Initialize_MSA_System(*th[:8], reference_frequency=F_REF)
        )
    )(jnp.asarray(np.concatenate([FLAGGED, CONTROL])))
    assert np.all(np.asarray(flag[: len(FLAGGED)]))
    assert not np.any(np.asarray(flag[len(FLAGGED) :]))


@pytest.mark.parametrize("prec_version", [223, 102])
def test_flagged_draws_match_lal(fallback_waveforms, prec_version):
    hp, hc = fallback_waveforms
    mm = np.array(
        [
            _mismatch(hp[i], hc[i], *_lal(FLAGGED[i], prec_version))
            for i in range(len(FLAGGED))
        ]
    )
    assert np.all(np.isfinite(mm))
    assert np.max(mm) < 1e-6, mm


def test_control_draws_match_lal(fallback_waveforms):
    hp, hc = fallback_waveforms
    n = len(FLAGGED)
    mm = np.array(
        [
            _mismatch(hp[n + i], hc[n + i], *_lal(CONTROL[i], 223))
            for i in range(len(CONTROL))
        ]
    )
    assert np.max(mm) < 1e-6, mm


def test_selection_is_exact_on_control_draws():
    """Where the MSA system is fine, the fallback returns its MSA inputs bit for bit."""
    rng = np.random.default_rng(0)
    f = jnp.asarray(FREQS[::8])
    msa_in = (
        jnp.asarray(rng.uniform()),
        jnp.asarray(rng.uniform()),
        *(jnp.asarray(rng.normal(size=(4, len(f)))) for _ in range(3)),
    )
    for theta in CONTROL:
        m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, _, incl, phi = theta
        out = jax.jit(pNNLO.apply_msa_fallback, static_argnums=12)(
            f,
            jnp.array([1, 2, 3, 4]),
            m1,
            m2,
            c1x,
            c1y,
            c1z,
            c2x,
            c2y,
            c2z,
            phi,
            incl,
            F_REF,
            *msa_in,
        )
        for a, b in zip(out, msa_in):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        pWF22 = build_pWF22(m1, m2, c1z, c2z, F_REF, chip=0.3)
        sel = pNNLO.select_fallback_pWF22(
            pWF22, jnp.asarray(False), m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, F_REF
        )
        for k, v in pWF22.items():
            np.testing.assert_array_equal(np.asarray(sel[k]), np.asarray(v))


def test_no_fallback_keeps_msa_angles(msa_only_waveforms):
    """msa_fallback=False is the MSA-only model: wrong where LAL falls back."""
    hp, hc = msa_only_waveforms
    mm = np.array(
        [_mismatch(hp[i], hc[i], *_lal(FLAGGED[i], 223)) for i in range(len(FLAGGED))]
    )
    assert np.median(mm) > 0.1, mm


def test_fallback_changes_nothing_on_control_draws(
    fallback_waveforms, msa_only_waveforms
):
    """Where the MSA system is fine, both settings agree to round-off.

    Not bitwise: the two settings compile to different XLA programs, and the
    MSA chain amplifies last-bit differences of fused kernels.  The model
    itself changes by ~1e-11 between vmap batch sizes (1 vs 8).
    """
    n = len(FLAGGED)
    for a, b in zip(fallback_waveforms, msa_only_waveforms):
        rel = np.max(np.abs(a[n:] - b[n:]), axis=1) / np.max(np.abs(b[n:]), axis=1)
        assert np.max(rel) < 1e-9, rel


def test_class_keyword():
    params = dict(
        zip(
            ("m1", "m2", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "iota"),
            FLAGGED[0][:10],
        )
    )
    m1, m2 = params.pop("m1"), params.pop("m2")
    params["M_c"] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    params["eta"] = m1 * m2 / (m1 + m2) ** 2
    params["phase_c"] = FLAGGED[0][10]
    wf = IMRPhenomXPHM(f_ref=F_REF)
    assert "msa_fallback=True" in repr(wf)
    assert IMRPhenomXPHM().msa_fallback is True
    assert IMRPhenomXPHM(msa_fallback=False).msa_fallback is False
    f = jnp.asarray(FREQS[::8])
    out = jax.jit(wf.__call__)(f, params)
    ref = jax.jit(lambda x: generate_xphm(*x, f, F_REF))(jnp.asarray(FLAGGED[0]))
    np.testing.assert_allclose(
        out["p"], ref[0], rtol=1e-6, atol=1e-6 * np.abs(ref[0]).max()
    )


# Eager gradient, as in tests/integration/test_jax.py: jit(grad) of XPHM takes
# tens of minutes and >100 GB of host memory to compile on the CPU.
# CONTROL[0] is not used: there the MSA-only model (msa_fallback=False, i.e.
# ripple 2c8d63b) already has NaN gradients w.r.t. masses and spins.
@pytest.mark.slow
@pytest.mark.parametrize(
    "theta", [FLAGGED[0], CONTROL[1]], ids=["msa_error", "control"]
)
def test_gradient_finite(theta):
    f = jnp.asarray(FREQS[::8])

    def loss(x):
        hp, hc = generate_xphm(*x, f, F_REF)
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2) * 1e44

    g = jax.grad(loss)(jnp.asarray(theta))
    assert np.all(np.isfinite(np.asarray(g))), g
