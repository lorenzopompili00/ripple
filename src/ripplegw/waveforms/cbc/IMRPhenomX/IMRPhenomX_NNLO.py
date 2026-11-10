"""NNLO post-Newtonian Euler angles for IMRPhenomXP(HM) (LAL PrecVersion 102).

LAL's default MSA angles (PrecVersion 223) fall back to these NNLO angles when
the MSA system fails to initialise, i.e. when |Omega_z^(5)| > 1000
(IMRPhenomX_Initialize_MSA_System, LALSimIMRPhenomX_precession.c).  On a
fallback LAL changes, for that binary only:

* the orbital angular momentum coefficients (3PN spinning L, ``case 102``),
  hence LRef and the J-frame quantities (thetaJN, kappa, phiJ_Sf, zeta);
* the Euler angles: single-spin NNLO alpha/epsilon (Appendix G of
  arXiv:2004.06503) and cos(beta) from L(v) + S_L and S_perp (Eq. 4.4);
* the final spin: PhenomXPFinalSpinMod 3 requires an MSA PrecVersion, so LAL
  uses version 0 instead (XLALSimIMRPhenomXPrecessingFinalSpin2017 with chi_p).

Everything here is a direct port of the LAL C source
(IMRPhenomXGetAndSetPrecessionVariables, Get_alphaepsilon_atfref and
Get_alpha_beta_epsilon / IMRPhenomXWignerdCoefficients in
LALSimIMRPhenomXPHM.c) and is jit/vmap/grad compatible.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ripplegw.constants import MTSUN
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import build_pWF22
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
    lal_mass_conventions,
)
from ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession import (
    XLALSimIMRPhenomXLPNAnsatz,
    compute_thetaJN_kappa_and_zeta,
)

# LAL threshold on |Omega_z^(5)| above which the MSA system is flagged as
# pathological (pPrec->MSA_ERROR = 1).
MSA_OMEGAZ5_THRESHOLD = 1000.0

# Index of the raw Omega_z^(5) coefficient in the array returned by
# IMRPhenomX_Initialize_MSA_System.
MSA_INIT_OMEGAZ5_INDEX = 34


def msa_error_from_init(msa_init: Float[Array, " n_msa"]) -> Bool[Array, ""]:
    """LAL's MSA_ERROR flag from the output of IMRPhenomX_Initialize_MSA_System.

    Args:
        msa_init: Array returned by IMRPhenomX_Initialize_MSA_System.

    Returns:
        True where LAL would fall back from the MSA to the NNLO angles.
    """
    return jnp.abs(msa_init[MSA_INIT_OMEGAZ5_INDEX]) > MSA_OMEGAZ5_THRESHOLD


def lal_chi_p(
    m1: FloatLike,
    m2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
) -> FloatLike:
    """Effective precession spin chi_p as LAL forms it (pPrec->chi_p), m1 >= m2.

    chi_p = max(A1 m1^2 chi1_perp, A2 m2^2 chi2_perp) / (A1 m1^2), with
    A1 = 2 + 3 m2 / (2 m1), A2 = 2 + 3 m1 / (2 m2) (Schmidt et al. 2015).

    Args:
        m1, m2: Mass fractions m_i / M (or masses in any common unit).
        chi1x, chi1y, chi2x, chi2y: In-plane spin components.

    Returns:
        chi_p.
    """
    m1_2 = m1 * m1
    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * m1_2 * jnp.sqrt(chi1x * chi1x + chi1y * chi1y)
    ASp2 = A2 * (m2 * m2) * jnp.sqrt(chi2x * chi2x + chi2y * chi2y)
    return jnp.maximum(ASp1, ASp2) / (A1 * m1_2)


def nnlo_orbital_angular_momentum_coefficients(
    eta: FloatLike, chi1L: FloatLike, chi2L: FloatLike, delta: FloatLike
) -> Float[Array, "10"]:
    """3PN spinning orbital angular momentum coefficients (PrecVersion 102).

    LALSimIMRPhenomX_precession.c, IMRPhenomXGetAndSetPrecessionVariables,
    ``case 102``.  Differs from the 222/223 set in L3 and L5.

    Args:
        eta: Symmetric mass ratio.
        chi1L, chi2L: Aligned spin components.
        delta: (m1 - m2) / M.

    Returns:
        Array [L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L].
    """
    pi2 = jnp.pi * jnp.pi
    L0 = 1.0
    L1 = 0.0
    L2 = 3.0 / 2.0 + eta / 6.0
    L3 = (5 * (chi1L * (-2 - 2 * delta + eta) + chi2L * (-2 + 2 * delta + eta))) / 6.0
    L4 = (81 + (-57 + eta) * eta) / 24.0
    L5 = (
        -7
        * (
            chi1L * (72 + delta * (72 - 31 * eta) + eta * (-121 + 2 * eta))
            + chi2L * (72 + eta * (-121 + 2 * eta) + delta * (-72 + 31 * eta))
        )
    ) / 144.0
    L6 = (10935 + eta * (-62001 + eta * (1674 + 7 * eta) + 2214 * pi2)) / 1296.0
    L7 = 0.0
    L8 = 0.0
    L8L = 0.0
    return jnp.array([L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class NNLOPrecessionSetup:
    """Pre-computed NNLO angle constants, invariant to emm and Mf.

    Compute once via compute_nnlo_precession_setup and pass into
    compute_nnlo_angles_given_setup inside any jax.vmap over modes.
    """

    eta: FloatLike
    alpha1: FloatLike
    alpha2: FloatLike
    alpha3: FloatLike
    alpha4L: FloatLike
    alpha5: FloatLike
    epsilon1: FloatLike
    epsilon2: FloatLike
    epsilon3: FloatLike
    epsilon4L: FloatLike
    epsilon5: FloatLike
    alpha_offset: FloatLike
    epsilon_offset: FloatLike
    L_coeffs: Float[Array, "10"]
    Sperp: FloatLike
    SL: FloatLike


def _nnlo_series(
    c1: FloatLike,
    c2: FloatLike,
    c3: FloatLike,
    c4L: FloatLike,
    c5: FloatLike,
    omega: Float[Array, " n_freq"] | FloatLike,
) -> Float[Array, " n_freq"] | FloatLike:
    """c1/w + c2/w^(2/3) + c3/w^(1/3) + c4L log(w) + c5 w^(1/3), as in LAL."""
    omega_cbrt = jnp.cbrt(omega)
    omega_cbrt2 = omega_cbrt * omega_cbrt
    return (
        c1 / omega
        + c2 / omega_cbrt2
        + c3 / omega_cbrt
        + c4L * jnp.log(omega)
        + c5 * omega_cbrt
    )


def compute_nnlo_precession_setup(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    reference_frequency: float,
    kappa: FloatLike,
    phiJ_Sf: FloatLike,
) -> NNLOPrecessionSetup:
    """Compute the NNLO (PrecVersion 102) Euler-angle coefficients and offsets.

    Port of the ``case 101-104`` branch of the PN Euler angle coefficients in
    IMRPhenomXGetAndSetPrecessionVariables and of Get_alphaepsilon_atfref
    (mprime = 2, PhenomXPConvention 1: alpha0 = pi - kappa,
    epsilon0 = phiJ_Sf - pi).  kappa and phiJ_Sf must be the J-frame values
    computed with the 102 orbital angular momentum (see
    nnlo_orbital_angular_momentum_coefficients).

    Args:
        mass_1, mass_2: Component masses in solar masses (mass_1 >= mass_2).
        chi1x/y/z, chi2x/y/z: Dimensionless spin components in the L-frame.
        reference_frequency: Reference frequency (Hz).
        kappa: J-frame azimuthal angle of the line of sight (rad).
        phiJ_Sf: Azimuthal angle of J0 in the source frame (rad).

    Returns:
        NNLOPrecessionSetup.
    """
    _, _, eta, m1, m2, _, Mtot = lal_mass_conventions(mass_1, mass_2)
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    eta5 = eta * eta4
    eta6 = eta * eta5
    delta = jnp.abs(m1 - m2)
    delta2 = delta * delta
    delta3 = delta * delta2
    m1_2 = m1 * m1
    m1_3 = m1 * m1_2
    m1_4 = m1 * m1_3
    m1_5 = m1 * m1_4
    m1_6 = m1 * m1_5
    m1_8 = m1_4 * m1_4
    m2_2 = m2 * m2
    q = m1 / m2
    pi = jnp.pi

    # pWF->chiEff (XLALSimIMRPhenomXchiEff)
    chi_eff = (
        0.5 * (1.0 + jnp.sqrt(1.0 - 4.0 * eta)) * chi1z
        + 0.5 * (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) * chi2z
    )

    chip = lal_chi_p(m1, m2, chi1x, chi1y, chi2x, chi2y)
    chip2 = chip * chip

    chiL = (1.0 + q) * (chi_eff / q)
    chiL2 = chiL * chiL

    # Post-Newtonian Euler angles: alpha
    alpha1 = -35 / 192.0 + (5 * delta) / (64.0 * m1)
    alpha2 = ((15 * chiL * delta * m1) / 128.0 - (35 * chiL * m1_2) / 128.0) / eta
    alpha3 = (
        -5515 / 3072.0
        + eta
        * (-515 / 384.0 - (15 * delta2) / (256.0 * m1_2) + (175 * delta) / (256.0 * m1))
        + (4555 * delta) / (7168.0 * m1)
        + ((15 * chip2 * delta * m1_3) / 128.0 - (35 * chip2 * m1_4) / 128.0) / eta2
    )
    # Term proportional to log(w)
    alpha4L = (
        (5 * chiL * delta2) / 16.0
        - (5 * chiL * delta * m1) / 3.0
        + (2545 * chiL * m1_2) / 1152.0
        + ((-2035 * chiL * delta * m1) / 21504.0 + (2995 * chiL * m1_2) / 9216.0) / eta
        + (
            (5 * chiL * chip2 * delta * m1_5) / 128.0
            - (35 * chiL * chip2 * m1_6) / 384.0
        )
        / eta3
        - (35 * pi) / 48.0
        + (5 * delta * pi) / (16.0 * m1)
    )
    alpha5 = (
        5
        * (
            -190512 * delta3 * eta6
            + 2268
            * delta2
            * eta3
            * m1
            * (eta2 * (323 + 784 * eta) + 336 * (25 * chiL2 + chip2) * m1_4)
            + 7
            * m1_3
            * (
                8024297 * eta4
                + 857412 * eta5
                + 3080448 * eta6
                + 143640 * chip2 * eta2 * m1_4
                - 127008 * chip2 * (-4 * chiL2 + chip2) * m1_8
                + 6048
                * eta3
                * ((2632 * chiL2 + 115 * chip2) * m1_4 - 672 * chiL * m1_2 * pi)
            )
            + 3
            * delta
            * m1_2
            * (
                -5579177 * eta4
                + 80136 * eta5
                - 3845520 * eta6
                + 146664 * chip2 * eta2 * m1_4
                + 127008 * chip2 * (-4 * chiL2 + chip2) * m1_8
                - 42336
                * eta3
                * ((726 * chiL2 + 29 * chip2) * m1_4 - 96 * chiL * m1_2 * pi)
            )
        )
    ) / (6.5028096e7 * eta4 * m1_3)

    # Post-Newtonian Euler angles: epsilon
    epsilon1 = -35 / 192.0 + (5 * delta) / (64.0 * m1)
    epsilon2 = ((15 * chiL * delta * m1) / 128.0 - (35 * chiL * m1_2) / 128.0) / eta
    epsilon3 = (
        -5515 / 3072.0
        + eta
        * (-515 / 384.0 - (15 * delta2) / (256.0 * m1_2) + (175 * delta) / (256.0 * m1))
        + (4555 * delta) / (7168.0 * m1)
    )
    # Term proportional to log(w)
    epsilon4L = (
        (5 * chiL * delta2) / 16.0
        - (5 * chiL * delta * m1) / 3.0
        + (2545 * chiL * m1_2) / 1152.0
        + ((-2035 * chiL * delta * m1) / 21504.0 + (2995 * chiL * m1_2) / 9216.0) / eta
        - (35 * pi) / 48.0
        + (5 * delta * pi) / (16.0 * m1)
    )
    epsilon5 = (
        5
        * (
            -190512 * delta3 * eta3
            + 2268 * delta2 * m1 * (eta2 * (323 + 784 * eta) + 8400 * chiL2 * m1_4)
            - 3
            * delta
            * m1_2
            * (
                eta * (5579177 + 504 * eta * (-159 + 7630 * eta))
                + 254016 * chiL * m1_2 * (121 * chiL * m1_2 - 16 * pi)
            )
            + 7
            * m1_3
            * (
                eta * (8024297 + 36 * eta * (23817 + 85568 * eta))
                + 338688 * chiL * m1_2 * (47 * chiL * m1_2 - 12 * pi)
            )
        )
    ) / (6.5028096e7 * eta * m1_3)

    # Offsets at omega_ref = pi M f_ref (Get_alphaepsilon_atfref, mprime = 2);
    # LAL uses the mprime = 2 offset for every mode.
    alpha0 = jnp.pi - kappa
    epsilon0 = phiJ_Sf - jnp.pi
    omega_ref = jnp.pi * (Mtot * MTSUN) * reference_frequency
    alpha_offset = (
        _nnlo_series(alpha1, alpha2, alpha3, alpha4L, alpha5, omega_ref) - alpha0
    )
    epsilon_offset = (
        _nnlo_series(epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5, omega_ref)
        - epsilon0
    )

    return NNLOPrecessionSetup(
        eta=eta,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4L=alpha4L,
        alpha5=alpha5,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
        epsilon4L=epsilon4L,
        epsilon5=epsilon5,
        alpha_offset=alpha_offset,
        epsilon_offset=epsilon_offset,
        L_coeffs=nnlo_orbital_angular_momentum_coefficients(eta, chi1z, chi2z, delta),
        Sperp=chip * m1_2,
        SL=chi1z * m1_2 + chi2z * m2_2,
    )


def compute_nnlo_angles_given_setup(
    Mf: Float[Array, " n_freq"] | FloatLike,
    emm: int | Array,
    setup: NNLOPrecessionSetup,
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"], Float[Array, " n_freq"]]:
    """NNLO Euler angles for the co-precessing mode with azimuthal index emm.

    Port of Get_alpha_beta_epsilon (LALSimIMRPhenomXPHM.c) and
    IMRPhenomXWignerdCoefficients: the angles are evaluated at
    omega = pi Mf 2/emm, and cos(beta) = sign(L + S_L) / sqrt(1 + s^2) with
    s = S_perp / (L + S_L) and L the 3PN orbital angular momentum at
    v = omega^(1/3).  Same output convention and masking as
    compute_evolved_spin_given_setup (angles zeroed above Mf = 0.33).

    Args:
        Mf: Dimensionless frequency array.
        emm: Azimuthal mode number m' of the co-precessing mode.
        setup: NNLOPrecessionSetup from compute_nnlo_precession_setup.

    Returns:
        alpha, epsilon, cos_beta arrays over Mf.
    """
    inspiral_mask = (Mf <= 0.33) & (Mf > 0.0)
    # Double-where keeps log/cbrt finite (and their gradients) on masked bins.
    Mf_safe = jnp.where(inspiral_mask, Mf, 0.01)
    omega = jnp.pi * Mf_safe * 2.0 / emm
    alpha = (
        _nnlo_series(
            setup.alpha1, setup.alpha2, setup.alpha3, setup.alpha4L, setup.alpha5, omega
        )
        - setup.alpha_offset
    )
    epsilon = (
        _nnlo_series(
            setup.epsilon1,
            setup.epsilon2,
            setup.epsilon3,
            setup.epsilon4L,
            setup.epsilon5,
            omega,
        )
        - setup.epsilon_offset
    )

    v = jnp.cbrt(omega)
    L = XLALSimIMRPhenomXLPNAnsatz(v, setup.eta / v, *setup.L_coeffs)
    LpSL = L + setup.SL
    s = setup.Sperp / LpSL
    cos_beta = jnp.copysign(1.0, LpSL) / jnp.sqrt(1.0 + s * s)

    alpha_out = jnp.where(inspiral_mask, alpha, 0.0)
    epsilon_out = jnp.where(inspiral_mask, epsilon, 0.0)
    cos_beta_out = jnp.where(inspiral_mask, cos_beta, 0.0)
    return alpha_out, epsilon_out, cos_beta_out


# pWF22 entries that depend on the final spin (see build_pWF22).
_FINAL_SPIN_KEYS = ("afinal", "fRING22", "fDAMP22", "fMECO", "fISCO22", "chip")


def select_fallback_pWF22(
    pWF22: dict[str, Any],
    msa_error: Bool[Array, ""],
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    reference_frequency: float,
) -> dict[str, Any]:
    """Use LAL's fallback final spin where the MSA system fails.

    With the default PhenomXPFinalSpinMod (4 -> 3) LAL uses the MSA
    orbit-averaged final spin; after the MSA -> NNLO fallback PrecVersion is
    102, so IMRPhenomX_SetPrecessingRemnantParams uses final-spin version 0:
    XLALSimIMRPhenomXPrecessingFinalSpin2017(eta, chi1L, chi2L, chi_p).
    This also changes fRING/fDAMP of the co-precessing modes.

    Args:
        pWF22: Output of build_pWF22 with the MSA final spin.
        msa_error: MSA_ERROR flag (msa_error_from_init).
        mass_1, mass_2: Component masses in solar masses (mass_1 >= mass_2).
        chi1x/y/z, chi2x/y/z: Dimensionless spin components in the L-frame.
        reference_frequency: Reference frequency (Hz).

    Returns:
        pWF22 with the final-spin dependent entries replaced where msa_error.
    """
    _, _, _, x1, x2, _, _ = lal_mass_conventions(mass_1, mass_2)
    chip = lal_chi_p(x1, x2, chi1x, chi1y, chi2x, chi2y)
    pWF22_v0 = build_pWF22(mass_1, mass_2, chi1z, chi2z, reference_frequency, chip=chip)
    return {
        k: jnp.where(msa_error, pWF22_v0[k], v) if k in _FINAL_SPIN_KEYS else v
        for k, v in pWF22.items()
    }


def apply_msa_fallback(
    Mf: Float[Array, " n_freq"] | FloatLike,
    emms: Array,
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    phiRef_In: FloatLike,
    inclination: FloatLike,
    reference_frequency: float,
    theta_JN: FloatLike,
    zeta_polarisation: FloatLike,
    alpha: Float[Array, "n_emm n_freq"],
    epsilon: Float[Array, "n_emm n_freq"],
    cos_beta: Float[Array, "n_emm n_freq"],
) -> tuple[
    FloatLike,
    FloatLike,
    Float[Array, "n_emm n_freq"],
    Float[Array, "n_emm n_freq"],
    Float[Array, "n_emm n_freq"],
]:
    """Replace the MSA frame and Euler angles by LAL's NNLO fallback on MSA_ERROR.

    Evaluates the PrecVersion 102 J-frame (3PN spinning LRef -> thetaJN,
    kappa, phiJ_Sf, zeta) and NNLO angles for every emm, and selects them with
    jnp.where wherever LAL's MSA initialisation fails (|Omega_z^(5)| > 1000).
    Elsewhere the MSA inputs are returned unchanged (bit-identical).

    Args:
        Mf: Dimensionless frequency array.
        emms: Azimuthal indices m' of the rows of alpha/epsilon/cos_beta.
        mass_1, mass_2: Component masses in solar masses (mass_1 >= mass_2).
        chi1x/y/z, chi2x/y/z: Dimensionless spin components in the L-frame.
        phiRef_In: Reference orbital phase (rad).
        inclination: Inclination (rad).
        reference_frequency: Reference frequency (Hz).
        theta_JN, zeta_polarisation: MSA (PrecVersion 223) J-frame angles.
        alpha, epsilon, cos_beta: MSA angles, shape (len(emms), n_freq).

    Returns:
        theta_JN, zeta_polarisation, alpha, epsilon, cos_beta after the fallback.
    """
    msa_error = msa_error_from_init(
        IMRPhenomX_Initialize_MSA_System(
            mass_1=mass_1,
            mass_2=mass_2,
            chi1x=chi1x,
            chi1y=chi1y,
            chi1z=chi1z,
            chi2x=chi2x,
            chi2y=chi2y,
            chi2z=chi2z,
            reference_frequency=reference_frequency,
        )
    )

    # J-frame with the 102 orbital angular momentum, as in twistup
    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)
    x1 = mass_1 / (mass_1 + mass_2)
    x2 = mass_2 / (mass_1 + mass_2)
    L_coeffs = nnlo_orbital_angular_momentum_coefficients(eta, chi1z, chi2z, x1 - x2)
    v_ref = jnp.cbrt(jnp.pi * (mass_1 + mass_2) * MTSUN * reference_frequency)
    LRef = XLALSimIMRPhenomXLPNAnsatz(v_ref, eta / v_ref, *L_coeffs)
    theta_JN_n, _, _, phiJ_Sf_n, kappa_n, zeta_n = compute_thetaJN_kappa_and_zeta(
        x1, x2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, LRef, phiRef_In, inclination
    )

    setup = compute_nnlo_precession_setup(
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        reference_frequency,
        kappa_n,
        phiJ_Sf_n,
    )
    alpha_n, epsilon_n, cos_beta_n = jax.vmap(
        compute_nnlo_angles_given_setup, in_axes=(None, 0, None)
    )(Mf, emms, setup)

    return (
        jnp.where(msa_error, theta_JN_n, theta_JN),
        jnp.where(msa_error, zeta_n, zeta_polarisation),
        jnp.where(msa_error, alpha_n, alpha),
        jnp.where(msa_error, epsilon_n, epsilon),
        jnp.where(msa_error, cos_beta_n, cos_beta),
    )
