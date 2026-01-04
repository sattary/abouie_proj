"""
Stochastic baseline: The limit to beat.

Implements the analytic steady-state formula from Vashaee & Abouie (2025).
This is the theoretical limit of the Poissonian collision model.
"""

import jax.numpy as jnp
from typing import NamedTuple
from src.physics import SystemParams, thermal_occupation


class StochasticParams(NamedTuple):
    """Parameters for the stochastic collision model."""

    # System
    omega_c: float  # Cavity frequency (GHz)
    omega_a: float  # Atom frequency (GHz)
    kappa: float  # Cavity decay rate (GHz)

    # Bath
    T_bath: float  # Bath temperature (K)

    # Atom reservoir
    T_atom: float  # Atom pair temperature (K)
    lambda_ex: float  # Exchange coupling (GHz)

    # Collision parameters
    g: float  # Coupling strength (GHz)
    tau: float  # Interaction time (ns)
    R: float  # Collision rate (GHz, i.e., 1/ns)

    # Two-atom enhancement
    chi: float = 2.0  # Coherent enhancement factor (1-2)


def compute_partition_function(beta_hw: float, beta_hl: float) -> float:
    """Partition function Z = 2[cosh(βℏω) + cosh(βℏλ)] (Eq. 3)"""
    return 2 * (jnp.cosh(beta_hw) + jnp.cosh(beta_hl))


def compute_stream_coefficients_one_atom(
    params: StochasticParams,
) -> tuple[float, float]:
    """
    One-subsystem stream coefficients r1, r2 (Eq. 6).
    Only one atom of the pair couples to the cavity.

    r1 = ρ_e + ρ_d (upward rate coefficient)
    r2 = ρ_g + ρ_d (downward rate coefficient)
    r1 + r2 = 1
    """
    beta_hw = params.omega_a * 0.048 / params.T_atom
    beta_hl = params.lambda_ex * 0.048 / params.T_atom

    Z = compute_partition_function(beta_hw, beta_hl)

    rho_e = jnp.exp(-beta_hw) / Z
    rho_g = jnp.exp(beta_hw) / Z
    rho_d = jnp.cosh(beta_hl) / Z

    r1 = rho_e + rho_d
    r2 = rho_g + rho_d

    return float(r1), float(r2)


def compute_stream_coefficients_two_atom(
    params: StochasticParams,
) -> tuple[float, float]:
    """
    Two-subsystem stream coefficients r1^(2), r2^(2) (Eq. 32).
    Both atoms couple to the cavity.

    r1^(2) = ρ_e + ρ_d + ρ_nd
    r2^(2) = ρ_g + ρ_d + ρ_nd
    r1^(2) + r2^(2) = 1 + 2*ρ_nd (NOT normalized to 1!)
    """
    beta_hw = params.omega_a * 0.048 / params.T_atom
    beta_hl = params.lambda_ex * 0.048 / params.T_atom

    Z = compute_partition_function(beta_hw, beta_hl)

    rho_e = jnp.exp(-beta_hw) / Z
    rho_g = jnp.exp(beta_hw) / Z
    rho_d = jnp.cosh(beta_hl) / Z
    rho_nd = -jnp.sinh(beta_hl) / Z  # Coherence term (negative for λ > 0)

    r1 = rho_e + rho_d + rho_nd
    r2 = rho_g + rho_d + rho_nd

    return float(r1), float(r2)


def compute_detuning_filter(
    delta: float,
    kappa: float,
    tau: float,
) -> float:
    """
    Spectral overlap filter L(Δ) (Eq. 14).

    L(Δ) = 1 / (1 + (2Δ/Γ_over)²)
    Γ_over = κ + 1/τ
    """
    gamma_over = kappa + 1.0 / tau
    return 1.0 / (1.0 + (2 * delta / gamma_over) ** 2)


def compute_steady_state_occupation_one_atom(
    params: StochasticParams,
    delta: float = 0.0,
) -> float:
    """
    Steady-state cavity occupation for one-atom coupling (Eq. 25).

    n* = (κn̄₁ + R·r₁·φ²·L(Δ)) / (κ + R·(r₂-r₁)·φ²·L(Δ))

    Args:
        params: Stochastic model parameters
        delta: Detuning Δ = ω_a - ω_c (GHz)

    Returns:
        Steady-state photon number n*
    """
    # Bath occupation
    n_bar = thermal_occupation(params.omega_c, params.T_bath)

    # Stream coefficients
    r1, r2 = compute_stream_coefficients_one_atom(params)

    # Per-collision angle squared
    phi_sq = (params.g * params.tau) ** 2

    # Detuning filter
    L_delta = compute_detuning_filter(delta, params.kappa, params.tau)

    # Effective rates
    R_phi_L = params.R * phi_sq * L_delta

    # Steady state (Eq. 25)
    numerator = params.kappa * n_bar + R_phi_L * r1
    denominator = params.kappa + R_phi_L * (r2 - r1)

    return float(numerator / denominator)


def compute_steady_state_occupation_two_atom(
    params: StochasticParams,
    delta: float = 0.0,
) -> float:
    """
    Steady-state cavity occupation for two-atom coupling.

    Uses r1^(2), r2^(2) and φ₂² = χ·φ² (Eq. 31).

    Args:
        params: Stochastic model parameters
        delta: Detuning Δ = ω_a - ω_c (GHz)

    Returns:
        Steady-state photon number n*
    """
    # Bath occupation
    n_bar = thermal_occupation(params.omega_c, params.T_bath)

    # Stream coefficients (two-atom)
    r1, r2 = compute_stream_coefficients_two_atom(params)

    # Per-collision angle squared with enhancement
    phi_sq = params.chi * (params.g * params.tau) ** 2

    # Detuning filter
    L_delta = compute_detuning_filter(delta, params.kappa, params.tau)

    # Effective rates
    R_phi_L = params.R * phi_sq * L_delta

    # Steady state
    numerator = params.kappa * n_bar + R_phi_L * r1
    denominator = params.kappa + R_phi_L * (r2 - r1)

    return float(numerator / denominator)


def occupation_to_temperature(n: float, omega: float) -> float:
    """
    Convert photon occupation to effective temperature (Eq. 26).

    T_cav = ℏω / (k_B · ln(1 + 1/n))

    Using ℏω/k_B = ω[GHz] * 0.048 K/GHz
    """
    if n <= 0:
        return 0.0
    hw_over_kb = omega * 0.048  # in Kelvin
    return hw_over_kb / jnp.log(1 + 1 / n)


def compute_stochastic_limit(
    params: StochasticParams,
    delta: float = 0.0,
    two_atom: bool = True,
) -> tuple[float, float]:
    """
    Compute the stochastic limit: (n*, T_cav).

    This is THE LIMIT TO BEAT with Floquet protocols.

    Args:
        params: Stochastic model parameters
        delta: Detuning (GHz)
        two_atom: Use two-atom model (True) or one-atom (False)

    Returns:
        (n_star, T_cav): Steady-state occupation and effective temperature
    """
    if two_atom:
        n_star = compute_steady_state_occupation_two_atom(params, delta)
    else:
        n_star = compute_steady_state_occupation_one_atom(params, delta)

    T_cav = occupation_to_temperature(n_star, params.omega_c)

    return n_star, float(T_cav)


if __name__ == "__main__":
    # Reproduce paper's baseline parameters (Table 2)
    params = StochasticParams(
        omega_c=5.0,  # 5 GHz
        omega_a=5.0,  # Resonant
        kappa=0.00001,  # 10 kHz = 0.00001 GHz
        T_bath=1.0,  # 1 K
        T_atom=0.05,  # 50 mK
        lambda_ex=5.0,  # 5 GHz exchange
        g=0.5,  # 0.5 GHz coupling
        tau=0.05,  # 50 ns = 0.05 / (2π) in natural units
        R=5.0,  # 5 MHz = 0.005 GHz
        chi=2.0,  # Full coherent enhancement
    )

    print("=" * 60)
    print("STOCHASTIC BASELINE (The Limit to Beat)")
    print("=" * 60)

    # One-atom model
    r1_1, r2_1 = compute_stream_coefficients_one_atom(params)
    n_star_1, T_cav_1 = compute_stochastic_limit(params, delta=0.0, two_atom=False)

    print(f"\nOne-Atom Model (Δ=0):")
    print(f"  Stream coefficients: r1={r1_1:.4f}, r2={r2_1:.4f}")
    print(f"  Steady-state <n>*: {n_star_1:.4f}")
    print(f"  Effective T_cav: {T_cav_1 * 1000:.1f} mK")
    print(f"  Ratio T_cav/T_atom: {T_cav_1 / params.T_atom:.2f}")

    # Two-atom model
    r1_2, r2_2 = compute_stream_coefficients_two_atom(params)
    n_star_2, T_cav_2 = compute_stochastic_limit(params, delta=0.0, two_atom=True)

    print(f"\nTwo-Atom Model (Δ=0):")
    print(f"  Stream coefficients: r1={r1_2:.4f}, r2={r2_2:.4f}")
    print(f"  Steady-state <n>*: {n_star_2:.4f}")
    print(f"  Effective T_cav: {T_cav_2 * 1000:.1f} mK")
    print(f"  Ratio T_cav/T_atom: {T_cav_2 / params.T_atom:.2f}")

    # Detuning sweep
    print(f"\nDetuning Sweep (Two-Atom):")
    for delta in [0.0, 0.01, 0.05, 0.1, 0.5]:
        n, T = compute_stochastic_limit(params, delta=delta, two_atom=True)
        print(f"  Δ={delta:.2f} GHz: n*={n:.4f}, T_cav={T * 1000:.1f} mK")

    print("\n" + "=" * 60)
    print("Target: Floquet cycle must achieve n* < {:.4f}".format(n_star_2))
    print("=" * 60)
