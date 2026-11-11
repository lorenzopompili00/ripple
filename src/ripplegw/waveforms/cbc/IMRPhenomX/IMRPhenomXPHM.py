from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomX_NNLO as pNNLO
import ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession as pPrec
from ripplegw.constants import MPC, MRSUN, MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.interfaces import DistanceScaledWaveform, FrequencyDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.utils.spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)
from ripplegw.waveforms.cbc.IMRPhenomX import IMRPhenomX_utils
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import (
    XLALSimIMRPhenomXHMGethlmModes,
    build_pWF22,
)
from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
    IMRPhenomX_Initialize_MSA_System,
    lal_M_sec,
)

# Co-precessing modes twisted up, in the order of the hlm stack in generate_xphm.
_TWIST_MODES = ((2, 1), (2, 2), (3, 2), (3, 3), (4, 4))


def compute_chip(
    m1: FloatLike,
    m2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
) -> FloatLike:
    """
    Effective precession spin parameter chi_p.

    LALSimIMRPhenomX_precession.c lines 260-275:
      A1 = 2 + 3*m2/(2*m1), A2 = 2 + 3*m1/(2*m2)
      chip = max(A1*m1^2*chi1_perp, A2*m2^2*chi2_perp) / (A1*m1^2)
    """
    chi1_perp = jnp.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = jnp.sqrt(chi2x**2 + chi2y**2)
    q = m2 / m1  # assumes m1 >= m2
    A1 = 2.0 + 1.5 * q
    A2 = 2.0 + 1.5 / q
    return jnp.maximum(A1 * chi1_perp * m1**2, A2 * chi2_perp * m2**2) / (A1 * m1**2)


def compute_chiTot_perp(
    m1: FloatLike,
    m2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
) -> FloatLike:
    """
    Total perpendicular spin parameter for afinal_prec (LAL PhenomXPFinalSpinMod=4, the default).

    LALSimIMRPhenomX_precession.c lines 237-241:
      STot_perp = |S1_perp + S2_perp|  where S_i = chi_i * (mi/M)^2
      chiTot_perp = STot_perp * M^2 / m1^2

    This is used by build_pWF22 (as its 'chip' argument) to compute afinal_prec via
      Sperp = chiTot_perp * mm1^2 = STot_perp
      afinal_prec = sign(a_aln) * sqrt(STot_perp^2 + a_aln^2)
    matching LAL's default fsflag=4 convention.
    """
    mm1 = m1 / (m1 + m2)
    mm2 = m2 / (m1 + m2)
    Sx = chi1x * mm1**2 + chi2x * mm2**2
    Sy = chi1y * mm1**2 + chi2y * mm2**2
    return jnp.sqrt(Sx**2 + Sy**2) / mm1**2


def generate_xphm(
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
    msa_fallback: bool = True,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Generate IMRPhenomXPHM plus and cross polarizations.

    Args:
        msa_fallback: Matches LAL: on MSA_ERROR (the MSA system fails to
            initialise, |Omega_z^(5)| > 1000), use the NNLO angles
            (PrecVersion 102) instead of the MSA angles (PrecVersion 223); see
            IMRPhenomX_NNLO.  Both branches are evaluated and selected per
            binary with jnp.where, so under jax.vmap the NNLO branch runs for
            every binary.  Default True; False gives the MSA angles for every
            binary (ripple's behaviour before the fallback).
    """
    # LAL: Mf = pWF->M_sec * f with M_sec built from round-tripped solar masses;
    # the ULP matters because Mf feeds the near-degenerate MSA S^2 cubic.
    Mf: Float[Array, " n_freq"] = frequency_array * lal_M_sec(mass_1, mass_2)

    amp0, pWF22, _ = coprecessing_setup(
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        distance,
        reference_frequency,
        msa_fallback=msa_fallback,
    )
    hlms = coprecessing_modes(Mf, pWF22) * amp0

    hp, hc = twistup(
        Mf,
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        phi0,
        inclination,
        reference_frequency,
        hlms,
        msa_fallback=msa_fallback,
    )

    return hp, hc


def coprecessing_setup(
    mass_1: FloatLike,
    mass_2: FloatLike,
    chi1x: FloatLike,
    chi1y: FloatLike,
    chi1z: FloatLike,
    chi2x: FloatLike,
    chi2y: FloatLike,
    chi2z: FloatLike,
    distance: FloatLike,
    reference_frequency: float,
    msa_fallback: bool = True,
) -> tuple[FloatLike, dict, Array]:
    """Per-binary inputs of the co-precessing modes.

    Returns:
        amp0: Overall amplitude prefactor (LAL's XLALSimPhenomUtilsFDamp0).
        pWF22: XAS/XHM parameter dict (build_pWF22), with LAL's fallback final
            spin where the MSA system fails and ``msa_fallback`` is set.
        msa_error: LAL's MSA_ERROR flag (|Omega_z^(5)| > 1000).
    """
    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    # Use XHM (PhenomXAS-based) modes as the co-precessing seed — matches LAL
    # default (TwistPhenomHM=0) and Bilby. XHM uses IMRPhenomX_TimeShift_22 for
    # the t0 convention, which is the correct PE coalescence-time reference.
    # Use MSA-averaged afinal_prec (fsflag=3) for PrecVersion=222, matching LAL default.
    _msa_init = IMRPhenomX_Initialize_MSA_System(
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
    pWF22 = build_pWF22(
        mass_1,
        mass_2,
        chi1z,
        chi2z,
        reference_frequency,
        msa_SAv2=_msa_init[15],
        msa_S1L_pav=_msa_init[32],
        msa_S2L_pav=_msa_init[33],
    )
    msa_error = pNNLO.msa_error_from_init(_msa_init)
    if msa_fallback:
        pWF22 = pNNLO.select_fallback_pWF22(
            pWF22,
            msa_error,
            mass_1,
            mass_2,
            chi1x,
            chi1y,
            chi1z,
            chi2x,
            chi2y,
            chi2z,
            reference_frequency,
        )
    return amp0, pWF22, msa_error


def coprecessing_modes(
    Mf: Float[Array, " n_freq"], pWF22: dict
) -> Complex[Array, "5 n_freq"]:
    """Co-precessing XHM modes (21, 22, 32, 33, 44), shape (5, N), without amp0.

    phi0=0.0 follows LAL convention=1 (pWF->phi0 = 0 for the co-precessing modes);
    odd-ell modes carry ripple's (-1)^ell sign.
    """
    hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
        Mf, pWF22, phi0=0.0, ell_mm_pairs=list(_TWIST_MODES)
    )
    return jnp.stack(
        [
            (-1 if ell % 2 != 0 else 1) * hlm_dict[(ell, mm)]
            for (ell, mm) in _TWIST_MODES
        ]
    )


def precession_frame(
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
) -> tuple[FloatLike, FloatLike, pPrec.MSAPrecessionSetup, FloatLike]:
    """Per-binary precession quantities of the twist-up (PrecVersion 223).

    Returns:
        theta_JN, zeta_polarisation, MSA precession setup, twist cutoff in Mf.
    """
    bigM = 1
    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)
    eta2 = jnp.power(eta, 2)
    chi1L = chi1z
    chi2L = chi2z
    total_mass = mass_1 + mass_2

    mass_1_fraction = mass_1 / total_mass
    mass_2_fraction = mass_2 / total_mass

    delta = mass_1_fraction - mass_2_fraction

    orbital_angular_momentum = (
        pPrec.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(
            eta, eta2, chi1L, chi2L, delta, jnp.power(jnp.pi, 2)
        )
    )
    Msec = (mass_1 + mass_2) * MTSUN
    piM = jnp.pi * Msec
    v_ref = jnp.cbrt(piM * reference_frequency)
    LRef = (
        bigM
        * bigM
        * pPrec.XLALSimIMRPhenomXLPNAnsatz(
            v_ref,
            eta / v_ref,
            orbital_angular_momentum[0],
            orbital_angular_momentum[1],
            orbital_angular_momentum[2],
            orbital_angular_momentum[3],
            orbital_angular_momentum[4],
            orbital_angular_momentum[5],
            orbital_angular_momentum[6],
            orbital_angular_momentum[7],
            orbital_angular_momentum[8],
            orbital_angular_momentum[9],
        )
    )

    # Fused call: compute J0, thetaJN, kappa, and zeta_polarization in one pass
    # (avoids recomputing J0/thetaJ_Sf/phiJ_Sf twice).
    theta_JN, _Nz_Jf, _Nx_Jf, phiJ_Sf, kappa, zeta_polarisations = (
        pPrec.compute_thetaJN_kappa_and_zeta(
            mass_1_fraction,
            mass_2_fraction,
            chi1x,
            chi1y,
            chi1z,
            chi2x,
            chi2y,
            chi2z,
            LRef,
            phiRef_In,
            inclination,
        )
    )

    # Compute MSA precession constants once (independent of emm and Mf) so they
    # are not redundantly recomputed for each of the 5 modes inside the vmap.
    _msa_setup = pPrec.compute_msa_precession_setup(
        mass_1,
        mass_2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        reference_frequency,
        kappa,
        phiJ_Sf,
    )
    # LALSim zeroes the contribution above Mf = f_max_prime*M_sec — inclusive
    # comparison, fCutDef in {0.3, 0.33}; see mf_twist_cutoff.
    mf_cut = pPrec.mf_twist_cutoff(eta, chi1z, chi2z, lal_M_sec(mass_1, mass_2))
    return theta_JN, zeta_polarisations, _msa_setup, mf_cut


def precession_angles(
    Mf: Float[Array, " n_freq"] | FloatLike,
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
    zeta_polarisations: FloatLike,
    msa_setup: pPrec.MSAPrecessionSetup,
    msa_fallback: bool = True,
    angle_interp_nodes: int | None = None,
) -> tuple:
    """(theta_JN, zeta, alpha, epsilon, cos_beta) for emm = 1..4, shapes (4, N).

    MSA angles, exact (default) or interpolated from ``angle_interp_nodes``
    nodes (pPrec.compute_evolved_spin_interpolated, approximate; meant for
    batched generation with a fallback for irregular binaries). Where the MSA system fails
    and ``msa_fallback`` is set, the NNLO frame and angles replace them.
    """
    # Compute precession angles for all 4 unique emm values in a single batched call.
    # Modes 22 and 32 share emm=2, so we only need emm = 1, 2, 3, 4.
    if angle_interp_nodes is None:
        _batched_angles = jax.vmap(
            pPrec.compute_evolved_spin_given_setup, in_axes=(None, 0, None)
        )(Mf, jnp.array([1, 2, 3, 4]), msa_setup)
    else:
        _batched_angles = pPrec.compute_evolved_spin_interpolated(
            Mf, (1, 2, 3, 4), msa_setup, angle_interp_nodes
        )
    # _batched_angles is a tuple of 3 arrays, each shape (4, N_freq)
    alpha_all = _batched_angles[0]  # (4, N)
    eps_all = _batched_angles[1]  # (4, N)
    cos_beta_all = _batched_angles[2]  # (4, N)
    if msa_fallback:
        theta_JN, zeta_polarisations, alpha_all, eps_all, cos_beta_all = (
            pNNLO.apply_msa_fallback(
                Mf,
                jnp.array([1, 2, 3, 4]),
                mass_1,
                mass_2,
                chi1x,
                chi1y,
                chi1z,
                chi2x,
                chi2y,
                chi2z,
                phiRef_In,
                inclination,
                reference_frequency,
                theta_JN,
                zeta_polarisations,
                alpha_all,
                eps_all,
                cos_beta_all,
            )
        )
    return theta_JN, zeta_polarisations, alpha_all, eps_all, cos_beta_all


def twistup(
    Mf: Float[Array, " n_freq"] | FloatLike,
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
    hlm: Complex[Array, "n_modes n_freq"],
    msa_fallback: bool = True,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Rotate the co-precessing frame hlm modes into the inertial (J-frame) polarisations hp, hc.
    Implementation follows the lalsimulation function IMRPhenomXPHMTwistUps.

    Computes the precession angles (alpha, beta, epsilon) at each frequency via MSA,
    applies the per-mode Wigner-d rotation and spherical-harmonic projection, sums over
    modes (21, 22, 32, 33, 44), and applies a final polarisation rotation by zeta.
    PrecVersion 223 (LAL's default); no multibanding of the angles.

    Args:
        Mf: Dimensionless frequency array (len N).
        mass_1, mass_2: Component masses in solar masses.
        chi1x/y/z, chi2x/y/z: Dimensionless spin components in the L-frame.
        phiRef_In: Reference orbital phase (rad).
        inclination: Inclination angle between J and line of sight (rad).
        reference_frequency: Reference frequency for spin evolution (Hz).
        hlm: Co-precessing frame modes, shape (n_modes, N).
        msa_fallback: Matches LAL: on MSA_ERROR, use the NNLO angles and
            J-frame (PrecVersion 102); see IMRPhenomX_NNLO.apply_msa_fallback.
            Selected with jnp.where: under jax.vmap both branches run for every
            binary (see generate_xphm). Default True; False uses the MSA angles
            for every binary.

    Returns:
        hp, hc: Plus and cross polarisations, shape (N,).
    """
    spins = (chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    theta_JN, zeta_polarisations, _msa_setup, mf_cut = precession_frame(
        mass_1, mass_2, *spins, phiRef_In, inclination, reference_frequency
    )
    theta_JN, zeta_polarisations, alpha_all, eps_all, cos_beta_all = precession_angles(
        Mf,
        mass_1,
        mass_2,
        *spins,
        phiRef_In,
        inclination,
        reference_frequency,
        theta_JN,
        zeta_polarisations,
        _msa_setup,
        msa_fallback=msa_fallback,
    )
    return twist_and_sum(
        Mf,
        hlm,
        alpha_all,
        eps_all,
        cos_beta_all,
        theta_JN,
        zeta_polarisations,
        mf_cut,
    )


def twist_and_sum(
    Mf: Float[Array, " n_freq"],
    hlm: Complex[Array, "n_modes n_freq"],
    alpha_all: Float[Array, "4 n_freq"],
    eps_all: Float[Array, "4 n_freq"],
    cos_beta_all: Float[Array, "4 n_freq"],
    theta_JN: FloatLike,
    zeta_polarisations: FloatLike,
    mf_cut: FloatLike,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Frequency-dependent part of ``twistup``: Wigner-d rotation, mode sum, polarisation.

    Args:
        Mf: Dimensionless frequency array (len N).
        hlm: Co-precessing modes (21, 22, 32, 33, 44), shape (5, N), including amp0.
        alpha_all, eps_all, cos_beta_all: Precession angles for emm = 1..4, shape (4, N).
        theta_JN: Angle between J and the line of sight.
        zeta_polarisations: Polarisation rotation angle.
        mf_cut: Twist cutoff in Mf (pPrec.mf_twist_cutoff); above it hp = hc = 0.

    Returns:
        hp, hc: Plus and cross polarisations, shape (N,).
    """
    # Wigner-d half-angle powers for the 4 unique emm values.
    cBetah_all, sBetah_all = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta_all)

    # Twist up mode by mode and accumulate.  Modes 22 and 32 share emm = 2, so
    # exp(i*alpha) and the beta powers are reused for them.  The mode sum is a
    # plain Python accumulation (no (n_modes, N) stacks): under vmap XLA fuses it
    # into a few elementwise kernels instead of materialising per-mode buffers.
    #
    # LAL grouping (LALSimIMRPhenomXPHM.c:2192, 2204-2205):
    # eps_phase = (cexp(-mprime*i*eps) * hlm) / 2, then hp += eps_phase * hp_sum.
    # Complex multiplication is not associative, so mirror the order exactly.
    # The sums over modes and over m are well-conditioned (O(1) summands, no
    # amplification), so their summation order only matters at the ULP level.
    _twist_fns = {
        (2, 1): twist_21,
        (2, 2): twist_22,
        (3, 2): twist_32,
        (3, 3): twist_33,
        (4, 4): twist_44,
    }
    _emm_of_mode = {(2, 1): 1, (2, 2): 2, (3, 2): 2, (3, 3): 3, (4, 4): 4}
    _per_emm: dict[int, tuple[Complex[Array, " n_freq"], BetaPowers]] = {}
    _hp = 0.0
    _hc = 0.0
    for i_mode, (ell, mprime) in enumerate(_TWIST_MODES):
        emm = _emm_of_mode[(ell, mprime)]
        if emm not in _per_emm:
            _per_emm[emm] = (
                IMRPhenomX_utils.cexpi(alpha_all[emm - 1]),
                BetaPowers.from_half_angle_trig(
                    cBetah_all[emm - 1], sBetah_all[emm - 1]
                ),
            )
        cexp_i_alpha, beta_powers = _per_emm[emm]
        hp_sum, hc_sum = _twist_fns[(ell, mprime)](cexp_i_alpha, theta_JN, beta_powers)
        eps_phase = (
            IMRPhenomX_utils.cexpi(-mprime * eps_all[emm - 1]) * hlm[i_mode]
        ) / 2.0
        _hp = _hp + eps_phase * hp_sum
        _hc = _hc + eps_phase * hc_sum

    # LALSim zeroes the contribution above Mf = f_max_prime*M_sec — inclusive
    # comparison, fCutDef in {0.3, 0.33}; see mf_twist_cutoff.  Setting
    # cos_beta=0 in compute_evolved_spin_given_setup does NOT produce a null
    # rotation (beta=pi/2 gives cBetah=sBetah=1/sqrt(2)), so we must
    # explicitly zero _hp/_hc here to match LALSim's behavior.
    inspiral_mask = Mf <= mf_cut
    _hp = jnp.where(inspiral_mask, _hp, 0.0)
    _hc = jnp.where(inspiral_mask, _hc, 0.0)

    hp, hc = apply_polarization_rotation(zeta_polarisations, _hp, _hc)

    return hp, hc


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


def _cexp_im_alpha(
    cexp_i_alpha: Complex[Array, " n_freq"], ell: int
) -> list[Complex[Array, " n_freq"]]:
    """[exp(i*m*alpha) for m = -ell..ell] from exp(i*alpha).

    Powers are built by repeated multiplication as before; exp(-i*alpha) is the
    complex conjugate of exp(i*alpha) (|exp(i*alpha)| = 1 to rounding), which
    replaces a complex division per element.
    """
    cexp_mi_alpha = jnp.conj(cexp_i_alpha)
    pos = [jnp.ones_like(cexp_i_alpha), cexp_i_alpha]
    neg = [jnp.ones_like(cexp_i_alpha), cexp_mi_alpha]
    for _ in range(2, ell + 1):
        pos.append(cexp_i_alpha * pos[-1])
        neg.append(cexp_mi_alpha * neg[-1])
    return neg[:0:-1] + pos


def _twist_sum(
    cexp_im_alpha: list[Complex[Array, " n_freq"]],
    ylm: list[FloatLike],
    d_lm: list[Float[Array, " n_freq"]],
    d_lminus: list[Float[Array, " n_freq"]],
    ell_odd: bool,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Sum over m of the Wigner-d / spherical-harmonic terms of one (ell, m') mode.

    Equivalent to the vectorised form (eqs. 3.5-3.7 of the precessing paper)

        A_minus = cexp_im_alpha[::-1] * d_lminus * Y
        A_plus  = cexp_im_alpha * d_lm * conj(Y)
        hp = sum(A_minus + s * A_plus),  hc = sum(1j * (A_minus - s * A_plus))

    with s = -1 for odd ell, written as a Python sum over m so that no
    (2*ell + 1, n_freq) intermediate arrays are created.
    """
    sgn = -1.0 if ell_odd else 1.0
    n = len(cexp_im_alpha)
    hp_sum = 0.0
    hc_sum = 0.0
    for k in range(n):
        a_minus = cexp_im_alpha[n - 1 - k] * (d_lminus[k] * ylm[k])
        a_plus = cexp_im_alpha[k] * (d_lm[k] * jnp.conj(ylm[k]))
        hp_sum = hp_sum + (a_minus + sgn * a_plus)
        hc_sum = hc_sum + 1j * (a_minus - sgn * a_plus)
    return hp_sum, hc_sum


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
    b = beta_powers
    ylm = [compute_sminus2_l2(theta_JN, m=m) for m in range(-2, 3)]
    # Wigner-d coefficients d^2_{m,2}, m = -2..2
    d22 = [
        b.sBetah4,
        2.0 * b.cBetah * b.sBetah3,
        jnp.sqrt(6) * b.sBetah2 * b.cBetah2,
        2.0 * b.cBetah3 * b.sBetah,
        b.cBetah4,
    ]
    # Exploit symmetry d^2_{-m,-2} = (-1)^m d^2_{-m,2}. See eq. A2 of Precessing paper
    d2m2 = [d22[4], -d22[3], d22[2], -d22[1], d22[0]]
    return _twist_sum(_cexp_im_alpha(cexp_i_alpha, 2), ylm, d22, d2m2, ell_odd=False)


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
    b = beta_powers
    ylm = [compute_sminus2_l2(theta_JN, m=m) for m in range(-2, 3)]
    # Wigner-d coefficients d^2_{m,1}, m = -2..2
    d21 = [
        2.0 * b.cBetah * b.sBetah3,
        3.0 * b.cBetah2 * b.sBetah2 - b.sBetah4,
        jnp.sqrt(6) * (b.cBetah3 * b.sBetah - b.cBetah * b.sBetah3),
        b.cBetah2 * (b.cBetah2 - 3.0 * b.sBetah2),
        -2.0 * b.cBetah3 * b.sBetah,
    ]
    # Exploit symmetry d^2_{-m,-1} = -(-1)^m d^2_{m,1}. See eq. A2 of Precessing paper.
    d2m1 = [-d21[4], d21[3], -d21[2], d21[1], -d21[0]]
    return _twist_sum(_cexp_im_alpha(cexp_i_alpha, 2), ylm, d21, d2m1, ell_odd=False)


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
    b = beta_powers
    ylm = [compute_sminus2_l3(theta=theta_JN, m=m) for m in range(-3, 4)]
    sqrt6 = jnp.sqrt(6.0)
    sqrt15 = jnp.sqrt(15.0)
    sqrt5 = jnp.sqrt(5.0)
    # Wigner-d coefficients d^3_{m,3}, m = -3..3
    d33 = [
        b.sBetah6,
        sqrt6 * b.cBetah * b.sBetah5,
        sqrt15 * b.cBetah2 * b.sBetah4,
        2.0 * sqrt5 * b.cBetah3 * b.sBetah3,
        sqrt15 * b.cBetah4 * b.sBetah2,
        sqrt6 * b.cBetah5 * b.sBetah,
        b.cBetah6,
    ]
    # Exploit symmetry d^3_{-m,-3} = -(-1)^m d^3_{m,3}. See eq. A2 of Precessing paper.
    d3m3 = [d33[6], -d33[5], d33[4], -d33[3], d33[2], -d33[1], d33[0]]
    return _twist_sum(_cexp_im_alpha(cexp_i_alpha, 3), ylm, d33, d3m3, ell_odd=True)


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
    b = beta_powers
    ylm = [compute_sminus2_l3(theta=theta_JN, m=m) for m in range(-3, 4)]
    sqrt6 = jnp.sqrt(6.0)
    sqrt10 = jnp.sqrt(10.0)
    sqrt30 = jnp.sqrt(30.0)
    c1, c2, c3, c4, c5 = b.cBetah, b.cBetah2, b.cBetah3, b.cBetah4, b.cBetah5
    s1, s2, s3, s4, s5 = b.sBetah, b.sBetah2, b.sBetah3, b.sBetah4, b.sBetah5
    # Wigner-d coefficients d^3_{m,2}, m = -3..3
    d32 = [
        sqrt6 * c1 * s5,
        s4 * (5.0 * c2 - s2),
        sqrt10 * s3 * (2.0 * c3 - c1 * s2),
        sqrt30 * c2 * (c2 - s2) * s2,
        sqrt10 * c3 * (c2 * s1 - 2.0 * s3),
        c4 * (c2 - 5.0 * s2),
        -1.0 * sqrt6 * c5 * s1,
    ]
    # Exploit symmetry d^3_{-m,-2} = (-1)^m d^3_{m,2}. See eq. A2 of Precessing paper.
    d3m2 = [-d32[6], d32[5], -d32[4], d32[3], -d32[2], d32[1], -d32[0]]
    return _twist_sum(_cexp_im_alpha(cexp_i_alpha, 3), ylm, d32, d3m2, ell_odd=True)


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
    b = beta_powers
    ylm = [compute_sminus2_l4(theta=theta_JN, m=m) for m in range(-4, 5)]
    sqrt2 = jnp.sqrt(2.0)
    sqrt7 = jnp.sqrt(7.0)
    sqrt14 = jnp.sqrt(14.0)
    sqrt70 = jnp.sqrt(70.0)
    # Wigner-d coefficients d^4_{m,4}, m = -4..4
    d44 = [
        b.sBetah8,
        2.0 * sqrt2 * b.cBetah * b.sBetah7,
        2.0 * sqrt7 * b.cBetah2 * b.sBetah6,
        2.0 * sqrt14 * b.cBetah3 * b.sBetah5,
        sqrt70 * b.cBetah4 * b.sBetah4,
        2.0 * sqrt14 * b.cBetah5 * b.sBetah3,
        2.0 * sqrt7 * b.cBetah6 * b.sBetah2,
        2.0 * sqrt2 * b.cBetah7 * b.sBetah,
        b.cBetah8,
    ]
    # Exploit symmetry d^4_{-m,-4} = (-1)^m d^4_{m,4}. See eq. A2 of Precessing paper.
    d4m4 = [d44[8], -d44[7], d44[6], -d44[5], d44[4], -d44[3], d44[2], -d44[1], d44[0]]
    return _twist_sum(_cexp_im_alpha(cexp_i_alpha, 4), ylm, d44, d4m4, ell_odd=False)


def apply_polarization_rotation(
    zeta_polarization: FloatLike, _hp, _hc
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Apply polarization rotation to waveform components.

    Args:
        zeta_polarization (float): Polarization angle.
        _hp (array_like): Plus polarization component (unrotated).
        _hc (array_like): Cross polarization component (unrotated).

    Returns:
        tuple[array_like, array_like]: Rotated plus (hp) and cross (hc) polarizations.
    """
    cosPolFac = jnp.cos(2.0 * zeta_polarization)
    sinPolFac = jnp.sin(2.0 * zeta_polarization)

    hp = cosPolFac * _hp + sinPolFac * _hc
    hc = cosPolFac * _hc - sinPolFac * _hp

    return hp, hc


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


@register("IMRPhenomXPHM", is_tidal=False, is_precessing=True, highest_mode=4)
class IMRPhenomXPHM(FrequencyDomainWaveform, DistanceScaledWaveform):
    """IMRPhenomXPHM frequency-domain waveform (precessing spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
        msa_fallback (bool): Use LAL's MSA -> NNLO angle fallback (default).
    """

    f_ref: float
    msa_fallback: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        msa_fallback: bool = True,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            msa_fallback (bool): Matches LAL: on MSA_ERROR, use the NNLO angles
                (PrecVersion 102), as LAL does for its default PrecVersion 223.
                Selected per binary with jnp.where (under jax.vmap both branches
                run). Defaults to True; False uses the MSA angles for every binary.
        """
        self.f_ref = f_ref
        self.msa_fallback = msa_fallback

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXPHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params: Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = generate_xphm(
            m1,
            m2,
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            frequency,
            self.f_ref,
            msa_fallback=self.msa_fallback,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXPHM(f_ref={self.f_ref}, msa_fallback={self.msa_fallback})"
