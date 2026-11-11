"""Mixed precision and the batch API (generate_xphm_batch) of IMRPhenomXPHM.

Quick tier, on the quick batch (box draws, one with irregular angles, edge cases,
two MSA_ERROR draws) against the stored 24fc1a1 float64 outputs:

* float64 batch, NNLO fallback recomputed for the 2 MSA_ERROR rows only (K = 2):
  equal to generate_xphm (24fc1a1 behaviour) to rounding, MSA_ERROR rows included;
* "mixed", exact angles: max mismatch < 1e-6, finite, complex64; overflow (2
  MSA_ERROR draws > K = 1): the whole batch is recomputed with the NNLO fallback;
* "mixed" + 256 interpolation nodes + fallback, K = 3 (the irregular box draw
  and the 2 MSA_ERROR draws recomputed in one pass): max mismatch < 1e-4,
  exactly the irregular box draw flagged (MSA_ERROR draws are not), no
  overflow; its MSA_ERROR rows equal those of the overflow path.

Region sums of the aligned-spin (2,2) mode: an ansatz infinite outside its region
gives a finite amplitude and phase (float64, float32, split), with a finite gradient.

``slow``: a binary whose float32 intermediate amplitude is 1 / 0 at a grid point gives
a finite "mixed" waveform; batch float64 with the default K (overflow) = generate_xphm,
interpolation overflow path, per-binary "mixed", float32 batch, jaxpr check of
the float32 stages. ``gpu``: the 12288-draw validation of the fast setting on
the full multibanded grid.
"""

import functools
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as xphm
import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM_stages as stages
from ripplegw.utils import mixed_precision as mp
from tests.helpers.xphm import (
    F_REF,
    aligo_psd,
    compiled,
    max_rel_diff,
    mismatch,
    multibanded_delta_f,
    multibanded_grid,
    reference,
    sample_bbh_box,
)
from tests.unit.waveforms.cbc.test_IMRPhenomXPHM_exact import _per_binary

FAST = {"precision": "mixed", "angle_interp_nodes": 256}
# The quick batch has 2 MSA_ERROR rows and 1 irregular row; the default K (1)
# overflows, so these settings exercise the per-row recomputation.
MSA_ROWS = {"_fallback_max_rows": 2}  # exact angles: the 2 MSA_ERROR rows
FAST_ROWS = {"_fallback_max_rows": 3}  # interpolation: + the irregular row


def _batch(name, theta, freqs, **kw):
    fn = functools.partial(
        stages.generate_xphm_batch,
        frequency_array=jnp.asarray(freqs),
        reference_frequency=F_REF,
        **kw,
    )
    return compiled(name, fn, theta)(theta)


def _max_mismatch(hp, hc, hp0, hc0):
    return max(
        float(np.max(mismatch(np.asarray(hp), hp0))),
        float(np.max(mismatch(np.asarray(hc), hc0))),
    )


@pytest.fixture(scope="session")
def mixed_exact():
    r = reference()
    return _batch(
        "xphm_batch_mixed", jnp.asarray(r["theta"]), r["freqs"], precision="mixed"
    )


@pytest.fixture(scope="session")
def mixed_fast():
    r = reference()
    return _batch(
        "xphm_batch_mixed_interp256_rows",
        jnp.asarray(r["theta"]),
        r["freqs"],
        **FAST,
        **FAST_ROWS,
    )


# --------------------------------------------------------------------- quick
def test_float64_batch_nnlo_rows_equal_generate_xphm():
    """NNLO fallback on the MSA_ERROR rows only: the float64 batch reproduces
    generate_xphm with msa_fallback=True (24fc1a1 behaviour) on every row.

    Against generate_xphm on the same backend: rel 1e-9. Against the stored
    reference (made on CPU): mismatch < 1e-12, rel 1e-9 on CPU only (on other
    backends the MSA chain amplifies ulp differences; 8.8e-9 rel seen on an A100).
    """
    r = reference()
    assert r["msa_error"].sum() == 2
    hp, hc, n_flagged, overflow = _batch(
        "xphm_batch_f64_msarows", jnp.asarray(r["theta"]), r["freqs"], **MSA_ROWS
    )
    assert int(n_flagged) == 0 and not bool(overflow)
    # same configuration (and disk cache entry) as the out_default fixture
    hp1, hc1 = _per_binary("xphm_f64_default", jnp.asarray(r["theta"]), r["freqs"])
    assert max_rel_diff(hp, hp1) < 1e-9 and max_rel_diff(hc, hc1) < 1e-9
    assert _max_mismatch(hp, hc, r["nnlo_hp"], r["nnlo_hc"]) < 1e-12
    if jax.default_backend() == "cpu":
        assert max_rel_diff(hp, r["nnlo_hp"]) < 1e-9
        assert max_rel_diff(hc, r["nnlo_hc"]) < 1e-9
    # without the fallback the MSA_ERROR rows would be far off (24fc1a1 vs 2c8d63b)
    err = r["msa_error"]
    assert np.min(mismatch(r["nnlo_hp"][err], r["base_hp"][err])) > 1e-2


def test_mixed_exact_angles(mixed_exact):
    hp, hc, n_flagged, overflow = mixed_exact
    r = reference()
    assert hp.dtype == jnp.complex64
    assert np.all(np.isfinite(np.asarray(hp))) and np.all(np.isfinite(np.asarray(hc)))
    # overflow: 2 MSA_ERROR draws > K = max(1, ceil(0.01 * 11)) = 1, so the whole
    # batch is recomputed with the NNLO fallback
    assert int(n_flagged) == 0 and bool(overflow)
    assert _max_mismatch(hp, hc, r["nnlo_hp"], r["nnlo_hc"]) < 1e-6


def test_mixed_interpolated_angles_with_fallback(mixed_fast):
    hp, hc, n_flagged, overflow = mixed_fast
    r = reference()
    expected = int(np.sum((r["irregularity"] > 1.0) & ~r["msa_error"]))
    # the irregular MSA_ERROR draws (NNLO angles anyway) are not flagged
    assert np.sum((r["irregularity"] > 1.0) & r["msa_error"]) >= 1
    assert int(n_flagged) == expected == 1
    assert not bool(overflow)  # 1 flagged + 2 MSA_ERROR rows, K = 3
    assert np.all(np.isfinite(np.asarray(hp)))
    assert _max_mismatch(hp, hc, r["nnlo_hp"], r["nnlo_hc"]) < 1e-4
    # MSA_ERROR rows: exact NNLO angles, never interpolated
    err = r["msa_error"]
    h = (np.asarray(hp)[err], np.asarray(hc)[err])
    assert _max_mismatch(*h, r["nnlo_hp"][err], r["nnlo_hc"][err]) < 1e-6


def test_nnlo_rows_equal_nnlo_overflow(mixed_exact, mixed_fast):
    """The MSA_ERROR rows recomputed individually (mixed_fast, K = 3) equal the
    whole-batch recomputation (mixed_exact, overflow): both use exact NNLO angles
    in float64, then the same float32 modes and twist."""
    err = reference()["msa_error"]
    for a, b in zip(mixed_fast[:2], mixed_exact[:2]):
        assert max_rel_diff(np.asarray(a)[err], np.asarray(b)[err]) < 1e-5


def test_region_weight_and_ratio_helpers():
    """_weighted gives 0 where the region weight is 0, whatever the ansatz there;
    _ratio_or_inf gives inf for 1 / 0 with a gradient that _weighted stops (0, not
    NaN)."""
    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXAS as xas

    x = jnp.array([jnp.inf, 2.0, 3.0, jnp.nan, 5.0])
    w = jnp.array([0.0, 1.0, 0.5, 0.0, 0.25])
    np.testing.assert_array_equal(xas._weighted(x, w), [0.0, 2.0, 1.5, 0.0, 1.25])
    num, den = jnp.array([1.0, 3.0]), jnp.array([2.0, 0.0])
    np.testing.assert_array_equal(xas._ratio_or_inf(num, den), [0.5, jnp.inf])
    loss = lambda n, d: jnp.sum(
        xas._weighted(xas._ratio_or_inf(n, d), jnp.array([1.0, 0.0]))
    )
    dn, dd = jax.grad(loss, argnums=(0, 1))(num, den)
    np.testing.assert_array_equal(dn, [0.5, 0.0])
    np.testing.assert_array_equal(dd, [-0.25, 0.0])


@pytest.mark.parametrize("precision", ["float64", "float32", "split"])
def test_aligned_amplitude_and_phase_ignore_a_diverging_ansatz(monkeypatch, precision):
    """The aligned-spin (2,2) amplitude and phase (IMRPhenomXAS Amp and Phase, the
    co-precessing (2,2) mode of XPHM) stay finite when a region's ansatz is infinite
    outside its region (the intermediate amplitude at low frequencies, the inspiral
    phase at high frequencies), in float64, float32 and the float64 / float32 split."""
    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomX_utils as utils
    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXAS as xas

    amp_int, ins_phase = xas.get_intermediate_Amp, xas.get_inspiral_phase
    monkeypatch.setattr(
        xas,
        "get_intermediate_Amp",
        lambda fM, *a: jnp.where(fM < 0.005, jnp.inf, amp_int(fM, *a)),
    )
    monkeypatch.setattr(
        xas,
        "get_inspiral_phase",
        lambda fM, *a: jnp.where(fM > 0.08, jnp.inf, ins_phase(fM, *a)),
    )
    f = jnp.linspace(10.0, 500.0, 64)  # Mf 0.0025-0.12 at M = 50 Msun
    theta = jnp.array([30.0, 20.0, 0.3, -0.2])
    for fn, coeffs in (
        (xas.Amp, utils.PhenomX_amp_coeff_table),
        (xas.Phase, utils.PhenomX_phase_coeff_table),
    ):
        if precision == "split":
            g = mp.split_by_frequency(lambda x, t, fn=fn: fn(x, *t), f, (theta, coeffs))
            out = g(f, (theta, coeffs))
        else:
            out = mp.run_in(64 if precision == "float64" else 32, fn, f, theta, coeffs)
        assert np.all(np.isfinite(np.asarray(out))), (fn.__name__, precision)


@pytest.mark.slow
def test_mixed_finite_where_float32_intermediate_amplitude_diverges(monkeypatch):
    """A precessing binary (q ~ 0.37) whose co-precessing (2,2) intermediate amplitude
    1 / polynomial(Mf) is 1 / 0 in float32 at 27.125 Hz, an inspiral frequency outside
    the intermediate region (about 3 in 1e6 box draws have such a grid point on the
    multibanded grid): "mixed" gives a finite waveform there. Precondition (so that the
    test cannot pass vacuously if the float32 rounding changes): with the region sums
    as before (x * weight), the waveform is NaN at that point."""
    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXAS as xas

    def mixed():
        stages._SPLIT_CACHE.clear()  # the split jaxpr is cached per grid shape
        return stages.generate_xphm_staged(
            65.38460820979569,
            24.20576683604144,
            0.004386422900106385,
            0.3326009345682458,
            0.2579636889679578,
            0.045045695523561746,
            -0.4480164980632318,
            0.8662040845148204,
            100.0,
            1.48403763197006,
            2.7834250782436456,
            jnp.array([20.0, 27.125]),
            F_REF,
            precision="mixed",
        )

    try:
        hp, hc = mixed()
        assert np.all(np.isfinite(np.asarray(hp))) and np.all(
            np.isfinite(np.asarray(hc))
        )
        monkeypatch.setattr(xas, "_weighted", lambda x, w: x * w)
        hp, _ = mixed()
        assert not np.isfinite(np.asarray(hp)[1]), "the float32 1 / 0 no longer occurs"
    finally:
        monkeypatch.undo()
        stages._SPLIT_CACHE.clear()


def test_fallback_max_rows_validation():
    th = jnp.zeros((4, 11))
    with pytest.raises(ValueError, match="_fallback_max_rows"):
        stages.generate_xphm_batch(th, jnp.ones(3), F_REF, _fallback_max_rows=0)


def test_defaults_are_exact():
    sig = inspect.signature(stages.generate_xphm_batch)
    assert sig.parameters["precision"].default == "float64"
    assert sig.parameters["angle_interp_nodes"].default is None
    assert (
        inspect.signature(xphm.generate_xphm).parameters["precision"].default
        == "float64"
    )


def test_precision_validation():
    with pytest.raises(ValueError):
        stages._check_precision("bfloat16")
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="jax_enable_x64"):
        stages._check_precision("mixed")


def test_class_precision_keyword():
    wf = xphm.IMRPhenomXPHM(precision="mixed")
    assert "mixed" in repr(wf)
    assert xphm.IMRPhenomXPHM().precision == "float64"


# --------------------------------------------------------------------- slow
@pytest.mark.slow
def test_float64_batch_equals_generate_xphm():
    """The staged float64 batch path with the default K of the NNLO fallback
    (overflow on the quick batch: the whole batch recomputed) reproduces
    generate_xphm (shared setup code).

    Against generate_xphm on the same backend: rel 1e-9. Against the stored
    reference (made on CPU): rel 1e-9 on CPU; on other backends the MSA chain
    amplifies ulp differences (8.8e-9 rel seen on an A100), so there only the
    mismatch (< 1e-12) is checked.
    """
    r = reference()
    th = jnp.asarray(r["theta"])
    hp, hc, n_flagged, overflow = _batch("xphm_batch_f64", th, r["freqs"])
    assert int(n_flagged) == 0 and bool(overflow)
    # same configuration (and disk cache entry) as the quick-tier fixture
    hp1, hc1 = _per_binary("xphm_f64_default", th, r["freqs"])
    assert max_rel_diff(hp, hp1) < 1e-9 and max_rel_diff(hc, hc1) < 1e-9
    assert _max_mismatch(hp, hc, r["nnlo_hp"], r["nnlo_hc"]) < 1e-12
    if jax.default_backend() == "cpu":
        assert max_rel_diff(hp, r["nnlo_hp"]) < 1e-9
        assert max_rel_diff(hc, r["nnlo_hc"]) < 1e-9


@pytest.mark.slow
def test_overflow_recomputes_the_batch_exactly(mixed_exact):
    """More flagged binaries than K: the whole batch gets exact angles."""
    r = reference()
    hp, hc, n_flagged, overflow = _batch(
        "xphm_batch_mixed_interp256_overflow",
        jnp.asarray(r["theta"]),
        r["freqs"],
        angle_interp_fallback_threshold=0.0,  # flags every non-MSA_ERROR binary
        **FAST,
    )
    assert bool(overflow) and int(n_flagged) > 1
    assert max_rel_diff(hp, mixed_exact[0]) < 1e-5
    assert max_rel_diff(hc, mixed_exact[1]) < 1e-5


@pytest.mark.slow
def test_per_binary_mixed():
    r = reference()
    f = jnp.asarray(r["freqs"])
    hp, hc = jax.jit(
        jax.vmap(
            lambda t: xphm.generate_xphm(
                *[t[i] for i in range(11)], f, F_REF, precision="mixed"
            )
        )
    )(jnp.asarray(r["theta"]))
    assert hp.dtype == jnp.complex64
    assert _max_mismatch(hp, hc, r["nnlo_hp"], r["nnlo_hc"]) < 1e-6


@pytest.mark.slow
def test_float32_batch():
    """precision="float32" runs with x64 on; finite on the regular box draws."""
    r = reference()
    hp, _, _, _ = _batch(
        "xphm_batch_f32", jnp.asarray(r["theta"]), r["freqs"], precision="float32"
    )
    assert hp.dtype == jnp.complex64
    regular = ~r["msa_error"]
    assert np.isfinite(np.asarray(hp)[regular][:7]).all()


# Allowed float64 values with a frequency axis in the float32 stages: casts, call
# wrappers (their bodies are walked) and broadcasts / iotas of per-binary scalars.
_STRUCTURAL = {"convert_element_type", "jit", "pjit", "broadcast_in_dim", "iota"}


@pytest.mark.slow
def test_no_float64_per_frequency_arithmetic_in_float32_stages():
    r = reference()
    nf = len(r["freqs"])
    t = jnp.asarray(r["theta"][0])
    S = stages.stage_setup(*[t[i] for i in range(11)], F_REF)
    Mf64 = jnp.asarray(r["freqs"]) * S["M_sec"]
    ang = stages._wrap(stages.stage_angles(Mf64, S, F_REF))

    def f32_part(Mf64, S, ang):
        hlm = stages._modes_split(Mf64, S["pWF22"])
        return mp.run_in(32, stages.stage_twist, Mf64, hlm, ang, S["amp0"], S["mf_cut"])

    bad = []

    def walk(jx):
        for e in jx.eqns:
            if e.primitive.name not in _STRUCTURAL:
                for v in e.outvars:
                    dt, shape = (
                        str(getattr(v.aval, "dtype", "")),
                        getattr(v.aval, "shape", ()),
                    )
                    if dt in ("float64", "complex128") and nf in shape:
                        bad.append(e.primitive.name)
            for p in e.params.values():
                for q in p if isinstance(p, (list, tuple)) else [p]:
                    inner = getattr(q, "jaxpr", None)
                    if inner is not None:
                        walk(inner if hasattr(inner, "eqns") else inner.jaxpr)

    walk(jax.make_jaxpr(f32_part)(Mf64, S, ang).jaxpr)
    assert not bad, (
        f"float64 per-frequency arithmetic in the float32 stages: {set(bad)}"
    )


# --------------------------------------------------------------------- gpu
@pytest.mark.gpu
def test_fast_setting_on_12288_box_draws():
    """precision="mixed" + 256 nodes + fallback vs exact float64 generate_xphm on the
    full multibanded grid, 12288 draws (the validation set), aLIGO-weighted
    mismatch: no NaN, max <= 1e-4 (observed sample maximum 8.4e-5, p99 ~1e-7), no
    overflow at batch 4096."""
    freqs = jnp.asarray(multibanded_grid())
    weights = multibanded_delta_f() / aligo_psd(multibanded_grid())
    theta = jnp.asarray(sample_bbh_box(12288, seed=5))
    exact = jax.jit(
        jax.vmap(lambda t: xphm.generate_xphm(*[t[i] for i in range(11)], freqs, F_REF))
    )
    fast = jax.jit(
        functools.partial(
            stages.generate_xphm_batch,
            frequency_array=freqs,
            reference_frequency=F_REF,
            **FAST,
        )
    )
    mm, n_flag = [], 0
    for i in range(0, theta.shape[0], 4096):
        hp0, hc0 = (np.asarray(x) for x in exact(theta[i : i + 4096]))
        hp, hc, k, overflow = fast(theta[i : i + 4096])
        assert not bool(overflow)
        n_flag += int(k)
        assert np.isfinite(np.asarray(hp)).all() and np.isfinite(np.asarray(hc)).all()
        mm.append(
            np.maximum(
                mismatch(np.asarray(hp), hp0, weights),
                mismatch(np.asarray(hc), hc0, weights),
            )
        )
    mm = np.concatenate(mm)
    print(
        f"max {mm.max():.2e}  p99 {np.percentile(mm, 99):.2e}  flagged {n_flag / len(mm):.4f}"
    )
    assert mm.max() <= 1e-4
