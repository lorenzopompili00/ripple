"""Frozen copy of the IMRPhenomXPHM twist-up functions of ripple 2c8d63b.

Reference for the exactness tests of the rewritten twist_* functions
(test_IMRPhenomXPHM_exact.py). Copied verbatim from
src/ripplegw/waveforms/cbc/IMRPhenomX/IMRPhenomXPHM.py at 2c8d63b (BetaPowers,
twist_22, twist_21, twist_33, twist_32, twist_44,
IMRPhenomXWignerdCoefficients_cosbeta); do not edit.
"""

# ruff: noqa

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.typing import FloatLike
from ripplegw.utils.spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)


@dataclass
class BetaPowers:
    """
    Stores powers of cos(beta/2) and sin(beta/2) for Wigner-d coefficient calculations.

    Attributes:
        cBetah: cos(beta/2)
        cBetah2: cos^2(beta/2)
        cBetah3: cos^3(beta/2)
        cBetah4: cos^4(beta/2)
        cBetah5: cos^5(beta/2)
        cBetah6: cos^6(beta/2)
        cBetah7: cos^7(beta/2)
        cBetah8: cos^8(beta/2)
        sBetah: sin(beta/2)
        sBetah2: sin^2(beta/2)
        sBetah3: sin^3(beta/2)
        sBetah4: sin^4(beta/2)
        sBetah5: sin^5(beta/2)
        sBetah6: sin^6(beta/2)
        sBetah7: sin^7(beta/2)
        sBetah8: sin^8(beta/2)
    """

    cBetah: FloatLike
    cBetah2: FloatLike
    cBetah3: FloatLike
    cBetah4: FloatLike
    cBetah5: FloatLike
    cBetah6: FloatLike
    cBetah7: FloatLike
    cBetah8: FloatLike
    sBetah: FloatLike
    sBetah2: FloatLike
    sBetah3: FloatLike
    sBetah4: FloatLike
    sBetah5: FloatLike
    sBetah6: FloatLike
    sBetah7: FloatLike
    sBetah8: FloatLike

    @classmethod
    def from_half_angle_trig(cls, cBetah: FloatLike, sBetah: FloatLike):
        """
        Constructs a BetaPowers instance from cos(beta/2) and sin(beta/2).

        Args:
            cBetah: cos(beta/2)
            sBetah: sin(beta/2)

        Returns:
            BetaPowers instance with all power values computed
        """
        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah * cBetah2
        cBetah4 = cBetah * cBetah3
        cBetah5 = cBetah * cBetah4
        cBetah6 = cBetah * cBetah5
        cBetah7 = cBetah * cBetah6
        cBetah8 = cBetah * cBetah7

        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah * sBetah2
        sBetah4 = sBetah * sBetah3
        sBetah5 = sBetah * sBetah4
        sBetah6 = sBetah * sBetah5
        sBetah7 = sBetah * sBetah6
        sBetah8 = sBetah * sBetah7

        return cls(
            cBetah=cBetah,
            cBetah2=cBetah2,
            cBetah3=cBetah3,
            cBetah4=cBetah4,
            cBetah5=cBetah5,
            cBetah6=cBetah6,
            cBetah7=cBetah7,
            cBetah8=cBetah8,
            sBetah=sBetah,
            sBetah2=sBetah2,
            sBetah3=sBetah3,
            sBetah4=sBetah4,
            sBetah5=sBetah5,
            sBetah6=sBetah6,
            sBetah7=sBetah7,
            sBetah8=sBetah8,
        )


def twist_22(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Compute the twisting contributions for l=2, m'=2 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (2,2) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha

    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    # shape (5, N): rows indexed by m+2 for m in -2..2
    cexp_im_alpha_l2 = jnp.stack(
        [
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
        ],
        axis=0,
    )

    Y2mA = jnp.array(
        [
            compute_sminus2_l2(theta_JN, m=-2),
            compute_sminus2_l2(theta_JN, m=-1),
            compute_sminus2_l2(theta_JN, m=0),
            compute_sminus2_l2(theta_JN, m=1),
            compute_sminus2_l2(theta_JN, m=2),
        ]
    )

    # Wigner-d coefficients – shape (5, N)
    d22 = jnp.array(
        [
            beta_powers.sBetah4,
            2.0 * beta_powers.cBetah * beta_powers.sBetah3,
            jnp.sqrt(6) * beta_powers.sBetah2 * beta_powers.cBetah2,
            2.0 * beta_powers.cBetah3 * beta_powers.sBetah,
            beta_powers.cBetah4,
        ]
    )

    # Exploit symmetry d^2_{-m,-2} = (-1)^m d^2_{-m,2}. See eq. A2 of Precessing paper
    d2m2 = jnp.array([d22[4], -d22[3], d22[2], -d22[1], d22[0]])

    # Vectorised sum over m=-2..2.  The -m+2 index pattern equals reversed order.
    # Shapes: cexp_im_alpha_l2 (5,N), d2m2 (5,N), Y2mA (5,) -> broadcast to (5,N)
    A2m2emm = cexp_im_alpha_l2[::-1] * d2m2 * Y2mA[:, None]
    A22emmstar = cexp_im_alpha_l2 * d22 * jnp.conj(Y2mA)[:, None]
    hp_sum = jnp.sum(A2m2emm + A22emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A2m2emm - A22emmstar), axis=0)

    return hp_sum, hc_sum


def twist_21(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Compute the twisting contributions for l=2, m'=1 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (2,1) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha

    # shape (5, N): rows indexed by m+2 for m in -2..2
    cexp_im_alpha_l2 = jnp.stack(
        [
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
        ],
        axis=0,
    )

    Y2mA = jnp.array(
        [
            compute_sminus2_l2(theta_JN, m=-2),
            compute_sminus2_l2(theta_JN, m=-1),
            compute_sminus2_l2(theta_JN, m=0),
            compute_sminus2_l2(theta_JN, m=1),
            compute_sminus2_l2(theta_JN, m=2),
        ]
    )

    # Wigner-d coefficients for m'=1 – shape (5, N)
    d21 = jnp.array(
        [
            2.0 * beta_powers.cBetah * beta_powers.sBetah3,
            3.0 * beta_powers.cBetah2 * beta_powers.sBetah2 - beta_powers.sBetah4,
            jnp.sqrt(6)
            * (
                beta_powers.cBetah3 * beta_powers.sBetah
                - beta_powers.cBetah * beta_powers.sBetah3
            ),
            beta_powers.cBetah2 * (beta_powers.cBetah2 - 3.0 * beta_powers.sBetah2),
            -2.0 * beta_powers.cBetah3 * beta_powers.sBetah,
        ]
    )

    # Exploit symmetry d^2_{-m,-1} = -(-1)^m d^2_{m,1}. See eq. A2 of Precessing paper.
    d2m1 = jnp.array([-d21[4], d21[3], -d21[2], d21[1], -d21[0]])

    # Vectorised sum over m=-2..2.  The -m+2 index pattern equals reversed order.
    A2m1emm = cexp_im_alpha_l2[::-1] * d2m1 * Y2mA[:, None]
    A21emmstar = cexp_im_alpha_l2 * d21 * jnp.conj(Y2mA)[:, None]
    hp_sum = jnp.sum(A2m1emm + A21emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A2m1emm - A21emmstar), axis=0)

    return hp_sum, hc_sum


def twist_33(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Compute the twisting contributions for l=3, m'=3 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (3,3) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    # shape (7, N): rows indexed by m+3 for m in -3..3
    cexp_im_alpha_l3 = jnp.stack(
        [
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
        ],
        axis=0,
    )

    Y3mA = jnp.array(
        [
            compute_sminus2_l3(theta=theta_JN, m=-3),
            compute_sminus2_l3(theta=theta_JN, m=-2),
            compute_sminus2_l3(theta=theta_JN, m=-1),
            compute_sminus2_l3(theta=theta_JN, m=0),
            compute_sminus2_l3(theta=theta_JN, m=1),
            compute_sminus2_l3(theta=theta_JN, m=2),
            compute_sminus2_l3(theta=theta_JN, m=3),
        ]
    )

    # Wigner-d coefficients for m'=3 – shape (7, N)
    sqrt6 = jnp.sqrt(6.0)
    sqrt15 = jnp.sqrt(15.0)
    sqrt5 = jnp.sqrt(5.0)

    d33 = jnp.array(
        [
            beta_powers.sBetah6,
            sqrt6 * beta_powers.cBetah * beta_powers.sBetah5,
            sqrt15 * beta_powers.cBetah2 * beta_powers.sBetah4,
            2.0 * sqrt5 * beta_powers.cBetah3 * beta_powers.sBetah3,
            sqrt15 * beta_powers.cBetah4 * beta_powers.sBetah2,
            sqrt6 * beta_powers.cBetah5 * beta_powers.sBetah,
            beta_powers.cBetah6,
        ]
    )

    # Exploit symmetry d^3_{-m,-3} = -(-1)^m d^3_{m,3}. See eq. A2 of Precessing paper.
    d3m3 = jnp.array([d33[6], -d33[5], d33[4], -d33[3], d33[2], -d33[1], d33[0]])

    # Vectorised sum over m=-3..3.  The -m+3 index pattern equals reversed order.
    A3m3emm = cexp_im_alpha_l3[::-1] * d3m3 * Y3mA[:, None]
    A33emmstar = cexp_im_alpha_l3 * d33 * jnp.conj(Y3mA)[:, None]
    hp_sum = jnp.sum(A3m3emm - A33emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A3m3emm + A33emmstar), axis=0)

    return hp_sum, hc_sum


def twist_32(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Compute the twisting contributions for l=3, m'=2 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (3,2) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha

    # shape (7, N): rows indexed by m+3 for m in -3..3
    cexp_im_alpha_l3 = jnp.stack(
        [
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
        ],
        axis=0,
    )

    Y3mA = jnp.array(
        [
            compute_sminus2_l3(theta=theta_JN, m=-3),
            compute_sminus2_l3(theta=theta_JN, m=-2),
            compute_sminus2_l3(theta=theta_JN, m=-1),
            compute_sminus2_l3(theta=theta_JN, m=0),
            compute_sminus2_l3(theta=theta_JN, m=1),
            compute_sminus2_l3(theta=theta_JN, m=2),
            compute_sminus2_l3(theta=theta_JN, m=3),
        ]
    )

    # Wigner-d coefficients for m'=2 – shape (7, N)
    sqrt6 = jnp.sqrt(6.0)
    sqrt10 = jnp.sqrt(10.0)
    sqrt30 = jnp.sqrt(30.0)

    cBetah = beta_powers.cBetah
    cBetah2 = beta_powers.cBetah2
    cBetah3 = beta_powers.cBetah3
    cBetah4 = beta_powers.cBetah4
    cBetah5 = beta_powers.cBetah5
    sBetah = beta_powers.sBetah
    sBetah2 = beta_powers.sBetah2
    sBetah3 = beta_powers.sBetah3
    sBetah4 = beta_powers.sBetah4
    sBetah5 = beta_powers.sBetah5

    d32 = jnp.array(
        [
            sqrt6 * cBetah * sBetah5,
            sBetah4 * (5.0 * cBetah2 - sBetah2),
            sqrt10 * sBetah3 * (2.0 * cBetah3 - cBetah * sBetah2),
            sqrt30 * cBetah2 * (cBetah2 - sBetah2) * sBetah2,
            sqrt10 * cBetah3 * (cBetah2 * sBetah - 2.0 * sBetah3),
            cBetah4 * (cBetah2 - 5.0 * sBetah2),
            -1.0 * sqrt6 * cBetah5 * sBetah,
        ]
    )

    # Exploit symmetry d^3_{-m,-2} = (-1)^m d^3_{m,2}. See eq. A2 of Precessing paper.
    d3m2 = jnp.array([-d32[6], d32[5], -d32[4], d32[3], -d32[2], d32[1], -d32[0]])

    # Vectorised sum over m=-3..3.  The -m+3 index pattern equals reversed order.
    A3m2emm = cexp_im_alpha_l3[::-1] * d3m2 * Y3mA[:, None]
    A32emmstar = cexp_im_alpha_l3 * d32 * jnp.conj(Y3mA)[:, None]
    hp_sum = jnp.sum(A3m2emm - A32emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A3m2emm + A32emmstar), axis=0)

    return hp_sum, hc_sum


def twist_44(
    cexp_i_alpha: Complex[Array, " n_freq"],
    theta_JN: FloatLike,
    beta_powers: BetaPowers,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Compute the twisting contributions for l=4, m'=4 mode.

    This function computes the sum over m of the Wigner-d matrix elements
    and spherical harmonics for the (4,4) mode, following eq. 3.5-3.7
    in the Precessing paper.

    Args:
        cexp_i_alpha: Complex exponential e^{i*alpha} (array over frequencies)
        theta_JN: Angle between total angular momentum and line of sight
        beta_powers: BetaPowers object containing powers of cos(beta/2) and sin(beta/2)

    Returns:
        hp_sum: Plus polarization contribution
        hc_sum: Cross polarization contribution
    """
    # Complex exponential powers of alpha
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_3i_alpha = cexp_i_alpha * cexp_2i_alpha
    cexp_4i_alpha = cexp_i_alpha * cexp_3i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_m3i_alpha = cexp_mi_alpha * cexp_m2i_alpha
    cexp_m4i_alpha = cexp_mi_alpha * cexp_m3i_alpha

    # shape (9, N): rows indexed by m+4 for m in -4..4
    cexp_im_alpha_l4 = jnp.stack(
        [
            cexp_m4i_alpha,
            cexp_m3i_alpha,
            cexp_m2i_alpha,
            cexp_mi_alpha,
            jnp.ones_like(cexp_i_alpha),
            cexp_i_alpha,
            cexp_2i_alpha,
            cexp_3i_alpha,
            cexp_4i_alpha,
        ],
        axis=0,
    )

    Y4mA = jnp.array(
        [
            compute_sminus2_l4(theta=theta_JN, m=-4),
            compute_sminus2_l4(theta=theta_JN, m=-3),
            compute_sminus2_l4(theta=theta_JN, m=-2),
            compute_sminus2_l4(theta=theta_JN, m=-1),
            compute_sminus2_l4(theta=theta_JN, m=0),
            compute_sminus2_l4(theta=theta_JN, m=1),
            compute_sminus2_l4(theta=theta_JN, m=2),
            compute_sminus2_l4(theta=theta_JN, m=3),
            compute_sminus2_l4(theta=theta_JN, m=4),
        ]
    )

    # Wigner-d coefficients for m'=4 – shape (9, N)
    sqrt2 = jnp.sqrt(2.0)
    sqrt7 = jnp.sqrt(7.0)
    sqrt14 = jnp.sqrt(14.0)
    sqrt70 = jnp.sqrt(70.0)

    d44 = jnp.array(
        [
            beta_powers.sBetah8,
            2.0 * sqrt2 * beta_powers.cBetah * beta_powers.sBetah7,
            2.0 * sqrt7 * beta_powers.cBetah2 * beta_powers.sBetah6,
            2.0 * sqrt14 * beta_powers.cBetah3 * beta_powers.sBetah5,
            sqrt70 * beta_powers.cBetah4 * beta_powers.sBetah4,
            2.0 * sqrt14 * beta_powers.cBetah5 * beta_powers.sBetah3,
            2.0 * sqrt7 * beta_powers.cBetah6 * beta_powers.sBetah2,
            2.0 * sqrt2 * beta_powers.cBetah7 * beta_powers.sBetah,
            beta_powers.cBetah8,
        ]
    )

    # Exploit symmetry d^4_{-m,-4} = (-1)^m d^4_{m,4}. See eq. A2 of Precessing paper.
    d4m4 = jnp.array(
        [d44[8], -d44[7], d44[6], -d44[5], d44[4], -d44[3], d44[2], -d44[1], d44[0]]
    )

    # Vectorised sum over m=-4..4.  The -m+4 index pattern equals reversed order.
    A4m4emm = cexp_im_alpha_l4[::-1] * d4m4 * Y4mA[:, None]
    A44emmstar = cexp_im_alpha_l4 * d44 * jnp.conj(Y4mA)[:, None]
    hp_sum = jnp.sum(A4m4emm + A44emmstar, axis=0)
    hc_sum = jnp.sum(1j * (A4m4emm - A44emmstar), axis=0)

    return hp_sum, hc_sum


def IMRPhenomXWignerdCoefficients_cosbeta(
    cos_beta: Float[Array, " n_freq"],
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
    """
    Compute cos(beta/2) and sin(beta/2) from cos(beta).

    Uses half-angle formulas:
    - cos(beta/2) = sqrt((1 + cos(beta)) / 2)
    - sin(beta/2) = sqrt((1 - cos(beta)) / 2)

    Args:
        cos_beta (float or array): cos(beta).

    Returns:
        tuple[float or array, float or array]: (cos(beta/2), sin(beta/2)), both always non-negative.
    """
    # Note that the results here are indeed always non-negative
    cos_beta_half = jnp.sqrt(jnp.abs(1.0 + cos_beta) / 2.0)  # cos(beta/2)
    sin_beta_half = jnp.sqrt(jnp.abs(1.0 - cos_beta) / 2.0)  # sin(beta/2)

    return cos_beta_half, sin_beta_half
