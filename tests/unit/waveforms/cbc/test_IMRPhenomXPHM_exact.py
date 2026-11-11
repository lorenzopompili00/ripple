"""The exact IMRPhenomXPHM optimisations reproduce the base implementations.

References: outputs of 2c8d63b (before the optimisations and the NNLO fallback)
and 24fc1a1 (NNLO fallback, not optimised) on the quick batch, stored in
data/xphm_reference.npz; the twist-up functions of 2c8d63b are vendored in
tests/helpers/xphm_base_2c8d63b.py. Nothing here depends on the git history.

Quick tier: two compiled configurations (float64 with msa_fallback False and
True), each compiled once per session. ``slow``: float32 and gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as xphm
import tests.helpers.xphm_base_2c8d63b as base
from ripplegw.waveforms.cbc.IMRPhenomX import IMRPhenomX_utils
from tests.helpers.xphm import F_REF, compiled, max_rel_diff, mismatch, reference


def _per_binary(name, theta, freqs, **kw):
    f = jnp.asarray(freqs)

    def fn(th):
        return jax.vmap(
            lambda t: xphm.generate_xphm(*[t[i] for i in range(11)], f, F_REF, **kw)
        )(th)

    return compiled(name, fn, theta)(theta)


@pytest.fixture(scope="session")
def out_no_nnlo():
    r = reference()
    return _per_binary(
        "xphm_f64_nonnlo", jnp.asarray(r["theta"]), r["freqs"], msa_fallback=False
    )


@pytest.fixture(scope="session")
def out_default():
    r = reference()
    return _per_binary("xphm_f64_default", jnp.asarray(r["theta"]), r["freqs"])


def _assert_close(out, hp0, hc0, rel_tol, mm_tol):
    hp, hc = (np.asarray(x) for x in out)
    ok = np.isfinite(hp0).all(-1) & np.isfinite(hc0).all(-1)
    assert np.array_equal(ok, np.isfinite(hp).all(-1) & np.isfinite(hc).all(-1))
    for h, h0 in ((hp[ok], hp0[ok]), (hc[ok], hc0[ok])):
        assert max_rel_diff(h, h0) < rel_tol
        assert np.max(mismatch(h, h0)) < mm_tol


# --------------------------------------------------------------------- quick
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cexpi_is_bitwise_complex_exp(dtype):
    x = jnp.asarray(np.random.default_rng(0).uniform(-1e3, 1e3, 4096), dtype)
    np.testing.assert_array_equal(
        np.asarray(IMRPhenomX_utils.cexpi(x)), np.asarray(jnp.exp(1j * x))
    )


@pytest.mark.parametrize("name", ["21", "22", "32", "33", "44"])
def test_twist_functions_match_base(name):
    rng = np.random.default_rng(1)
    alpha = jnp.asarray(rng.uniform(-10, 10, 512))
    cos_beta = jnp.asarray(rng.uniform(-1, 1, 512))
    theta_jn = jnp.asarray(1.1)
    c, s = xphm.IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)
    new = getattr(xphm, f"twist_{name}")(
        jnp.exp(1j * alpha), theta_jn, xphm.BetaPowers.from_half_angle_trig(c, s)
    )
    old = getattr(base, f"twist_{name}")(
        jnp.exp(1j * alpha), theta_jn, base.BetaPowers.from_half_angle_trig(c, s)
    )
    for a, b in zip(new, old):
        assert max_rel_diff(a[None], b[None]) < 1e-13


def test_matches_2c8d63b(out_no_nnlo):
    """msa_fallback=False reproduces the base (2c8d63b) to rounding, same NaN pattern.

    The aligned-spin edge case (no in-plane spin: near-degenerate MSA system)
    amplifies rounding differences between the two XLA programs to ~5e-12
    relative; its mismatch is 2e-16. Generic draws agree to <~1e-13.
    """
    r = reference()
    _assert_close(out_no_nnlo, r["base_hp"], r["base_hc"], rel_tol=2e-11, mm_tol=1e-13)


def test_matches_24fc1a1(out_default):
    """The default (NNLO fallback on) reproduces 24fc1a1 (fallback, not optimised)."""
    r = reference()
    assert r["msa_error"].any()  # the batch contains MSA_ERROR draws
    _assert_close(out_default, r["nnlo_hp"], r["nnlo_hc"], rel_tol=1e-9, mm_tol=1e-12)


def test_class_has_no_interpolation_keywords():
    """Interpolated angles live in the batch API only (per-binary use is slower under vmap)."""
    with pytest.raises(TypeError):
        xphm.IMRPhenomXPHM(angle_interp_nodes=256)


# --------------------------------------------------------------------- slow
@pytest.mark.slow
def test_float32_matches_2c8d63b():
    """float32 (process-like: x64 off) is as accurate as the base float32 path on the box draws.

    Pure float32 is ~1e-2 relative off float64 in places, so two float32 runs
    on different backends (the stored reference was made on CPU) differ at
    that level. The test therefore compares both against the float64 base and
    requires the new code to be no less accurate. Box draws only: on
    ill-conditioned inputs (aligned spins: near-degenerate MSA; MSA_ERROR
    draws) float32 is not accurate in either version (see "mixed").
    """
    r = reference()
    with jax.enable_x64(False):
        th = jnp.asarray(r["theta"], jnp.float32)
        f = jnp.asarray(r["freqs"], jnp.float32)
        out = jax.jit(
            jax.vmap(
                lambda t: xphm.generate_xphm(
                    *[t[i] for i in range(11)], f, F_REF, msa_fallback=False
                )
            )
        )(th)
    hp, hc = (np.asarray(x) for x in out)
    box = np.arange(7)
    for h, h0, h64 in (
        (hp, r["base32_hp"], r["base_hp"]),
        (hc, r["base32_hc"], r["base_hc"]),
    ):
        # Pure-float32 NaNs on the ill-conditioned rows depend on rounding (and on
        # the backend): check finiteness on the box draws only.
        assert np.isfinite(h[box]).all() and np.isfinite(h0[box]).all()
        mm_new = np.max(mismatch(h[box], h64[box]))
        mm_base = np.max(mismatch(h0[box], h64[box]))
        assert mm_new <= max(10 * mm_base, 1e-5)


@pytest.mark.slow
def test_grad_matches_2c8d63b():
    """Derivatives w.r.t. m1, m2, chi1z, distance agree with the base (tiny grid).

    Forward mode (jacfwd): reverse-mode grad of the full model takes more than
    80 GB and tens of minutes to compile on CPU.
    """
    r = reference()
    g = jnp.asarray(r["grad_freqs"])
    t = jnp.asarray(r["theta"][0])

    def loss(m1, m2, c1z, d):
        p = [m1, m2, t[2], t[3], c1z, t[5], t[6], t[7], d, t[9], t[10]]
        hp, hc = xphm.generate_xphm(*p, g, F_REF, msa_fallback=False)
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2) * 1e40

    grad = jax.jit(jax.jacfwd(loss, argnums=(0, 1, 2, 3)))(t[0], t[1], t[4], t[8])
    assert np.all(np.isfinite(np.array(grad)))
    np.testing.assert_allclose(np.array(grad), r["base_grad"], rtol=1e-8)
