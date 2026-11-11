"""Interpolated MSA precession angles (approximate, opt-in, batch API only).

Angle-level checks on the quick batch (one compiled function per session):
convergence of alpha, epsilon (through exp(i x), so 2 pi offsets from the
unwrapping do not count) and cos(beta) with the number of nodes, and a grid
with f = 0 and f < 0 bins (which must not poison the node range).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as xphm
import ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession as pPrec
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import lal_M_sec
from tests.helpers.xphm import F_REF, compiled, reference

EMMS = (1, 2, 3, 4)
EDGE_BINS = np.array([-5.0, 0.0, 5.0, 10.0])  # prepended to the quick grid


def _angle_errors(theta, freqs):
    f = jnp.asarray(freqs)
    f_edge = jnp.concatenate([jnp.asarray(EDGE_BINS), f])

    def one(t):
        _, _, setup, _ = xphm.precession_frame(
            *[t[i] for i in range(8)], t[10], t[9], F_REF
        )
        mf = f * lal_M_sec(t[0], t[1])
        exact = jax.vmap(
            pPrec.compute_evolved_spin_given_setup, in_axes=(None, 0, None)
        )(mf, jnp.array(EMMS), setup)

        def err(n):
            a, e, c = pPrec.compute_evolved_spin_interpolated(mf, EMMS, setup, n)
            return jnp.stack(
                [
                    jnp.max(jnp.abs(jnp.exp(1j * a) - jnp.exp(1j * exact[0]))),
                    jnp.max(jnp.abs(jnp.exp(1j * e) - jnp.exp(1j * exact[1]))),
                    jnp.max(jnp.abs(c - exact[2])),
                ]
            )

        mf_edge = f_edge * lal_M_sec(t[0], t[1])
        a0, e0, c0 = pPrec.compute_evolved_spin_interpolated(mf_edge, EMMS, setup, 256)
        finite_edge = jnp.all(jnp.isfinite(a0) & jnp.isfinite(e0) & jnp.isfinite(c0))
        # bins with f <= 0 take the lowest-node values, which are those of emm = 4
        # at the lowest positive frequency (x = 2 Mf / 4 is the bottom of the range)
        same_as_lowest = jnp.all(a0[3, :2] == a0[3, 2])
        irr_edge = pPrec.angle_node_irregularity(mf_edge, EMMS, setup, 256)
        irr = pPrec.angle_node_irregularity(mf, EMMS, setup, 256)
        return err(64), err(256), finite_edge, same_as_lowest, irr_edge, irr

    return compiled("xphm_interp_angle_errors", jax.vmap(one), theta)(theta)


@pytest.fixture(scope="session")
def angle_errors():
    r = reference()
    return tuple(
        np.asarray(x) for x in _angle_errors(jnp.asarray(r["theta"]), r["freqs"])
    )


def test_interpolated_angles_converge(angle_errors):
    """Regular binaries: 256 nodes are much closer to exact than 64, for all three angles."""
    e64, e256, *_ = angle_errors
    r = reference()
    regular = ~(r["irregularity"] > 1.0) & ~r["msa_error"]
    assert regular.sum() >= 8
    assert np.all(e256[regular] <= e64[regular] + 1e-12)
    assert np.max(e256[regular]) < 1e-2
    assert np.median(e256[regular]) < 1e-5


def test_zero_and_negative_frequency_bins(angle_errors):
    """f = 0 and f < 0 bins: interpolated angles and the flag stay finite."""
    _, _, finite_edge, same_as_lowest, irr_edge, _ = angle_errors
    assert finite_edge.all()
    assert same_as_lowest.all()
    assert np.all(np.isfinite(irr_edge))


def test_irregularity_matches_reference(angle_errors):
    """The stored flags (planted irregular and MSA_ERROR draws) are what the code computes."""
    np.testing.assert_allclose(angle_errors[5], reference()["irregularity"], rtol=1e-9)


def test_irregularity_flag_is_nan_safe():
    thr = 1.0
    irr = jnp.array([0.5, 2.0, jnp.nan])
    np.testing.assert_array_equal(np.asarray(~(irr <= thr)), [False, True, True])


def test_n_nodes_validation():
    with pytest.raises(ValueError):
        pPrec.compute_evolved_spin_interpolated(
            jnp.linspace(0.01, 0.1, 10), (1, 2), None, 3
        )
