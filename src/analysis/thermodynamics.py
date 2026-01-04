"""
Thermodynamic analysis for Floquet cooling cycles.
Computes COP, cooling power, and entropy efficiency.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, NamedTuple

from src.physics import SystemParams, thermal_occupation
from src.floquet import FloquetCycleParams
from src.baseline import StochasticParams, occupation_to_temperature


class ThermodynamicMetrics(NamedTuple):
    """Thermodynamic performance metrics."""

    n_cav: float  # Steady-state occupation
    T_cav: float  # Cavity temperature (K)
    Q_cool: float  # Cooling power (heat extracted per cycle)
    W_drive: float  # Work input per cycle (estimated)
    COP: float  # Coefficient of Performance
    eta_S: float  # Entropy efficiency (eta_S = S_out / S_in)
    carnot_limit: float  # Carnot COP for comparison


def compute_thermodynamics(
    params: SystemParams,
    cycle: FloquetCycleParams,
    n_final: float,
    n_initial: float = None,
) -> ThermodynamicMetrics:
    """
    Compute thermodynamic metrics for a Floquet cooling cycle.

    Args:
        params: System parameters
        cycle: Floquet cycle configuration
        n_final: Steady-state cavity occupation
        n_initial: Initial occupation (default: thermal at T_bath)

    Returns:
        ThermodynamicMetrics with all computed values
    """
    omega = params.omega_c
    hbar_omega = omega * 0.048  # in K (hbar*omega / k_B)

    # Initial state
    if n_initial is None:
        n_initial = thermal_occupation(params.omega_c, params.T_bath)

    # Effective cavity temperature
    T_cav = occupation_to_temperature(n_final, omega)

    # Cooling power: heat removed per cycle
    # Q_cool = hbar * omega * (n_initial - n_final) per cycle
    # In units where everything is normalized to omega
    delta_n = n_initial - n_final
    Q_cool = hbar_omega * max(0, delta_n)  # Only positive cooling

    # Work input estimate: integral of g(t)^2 over cycle
    # W ~ integral_0^T g(t)^2 dt (simplified model)
    g_rms = float(jnp.sqrt(jnp.mean(cycle.g_sequence**2)))
    W_drive = g_rms**2 * cycle.T_cycle * 0.1  # Approximate scaling

    # Coefficient of Performance
    if W_drive > 1e-10:
        COP = Q_cool / W_drive
    else:
        COP = float("inf") if Q_cool > 0 else 0.0

    # Carnot limit: COP_carnot = T_cold / (T_hot - T_cold)
    T_cold = min(T_cav, params.T_atom)
    T_hot = params.T_bath
    if T_hot > T_cold and T_cold > 0:
        carnot_limit = T_cold / (T_hot - T_cold)
    else:
        carnot_limit = float("inf")

    # Entropy efficiency
    # eta_S = S_extracted / S_input
    # Simplified: proportional to (delta_T / T_cold) / W_drive
    if T_cold > 0 and W_drive > 0:
        delta_T = params.T_bath - T_cav
        eta_S = (delta_T / T_cold) / (W_drive + 1e-10)
    else:
        eta_S = 0.0

    return ThermodynamicMetrics(
        n_cav=n_final,
        T_cav=T_cav,
        Q_cool=Q_cool,
        W_drive=W_drive,
        COP=COP,
        eta_S=eta_S,
        carnot_limit=carnot_limit,
    )


def compute_cooling_power_vs_temperature(
    params: SystemParams,
    cycle: FloquetCycleParams,
    n_final: float,
    T_targets: list = None,
) -> list:
    """
    Compute cooling power curve vs target temperature.
    """
    if T_targets is None:
        T_targets = np.linspace(0.01, params.T_bath, 20)

    results = []
    n_bath = thermal_occupation(params.omega_c, params.T_bath)

    for T in T_targets:
        n_target = thermal_occupation(params.omega_c, T)
        delta_n = n_bath - n_target
        Q = params.omega_c * 0.048 * max(0, delta_n)

        results.append(
            {
                "T": float(T),
                "n_target": float(n_target),
                "Q_max": float(Q),
            }
        )

    return results


def print_thermodynamic_report(metrics: ThermodynamicMetrics):
    """Print formatted thermodynamic report."""
    print("=" * 50)
    print("THERMODYNAMIC ANALYSIS")
    print("=" * 50)
    print(f"Cavity occupation: <n> = {metrics.n_cav:.4f}")
    print(f"Cavity temperature: T_cav = {metrics.T_cav * 1000:.1f} mK")
    print(f"Cooling power: Q_cool = {metrics.Q_cool * 1000:.3f} mK/cycle")
    print(f"Work input: W_drive = {metrics.W_drive:.4f}")
    print(f"COP: {metrics.COP:.2f}")
    print(f"Carnot limit: {metrics.carnot_limit:.2f}")
    print(f"Entropy efficiency: eta_S = {metrics.eta_S:.4f}")

    if metrics.COP < metrics.carnot_limit:
        efficiency = metrics.COP / metrics.carnot_limit * 100
        print(f"Carnot efficiency: {efficiency:.1f}%")


def analyze_grape_result(
    params: SystemParams,
    cycle: FloquetCycleParams,
    n_final: float,
    n_stochastic: float,
) -> dict:
    """
    Complete thermodynamic analysis comparing GRAPE to stochastic.
    """
    # Compute metrics for GRAPE
    grape_metrics = compute_thermodynamics(params, cycle, n_final)

    # Compute metrics for stochastic baseline
    stoch_metrics = compute_thermodynamics(params, cycle, n_stochastic)

    # Improvement factors
    T_improvement = (
        (stoch_metrics.T_cav - grape_metrics.T_cav) / stoch_metrics.T_cav * 100
    )
    n_improvement = (n_stochastic - n_final) / n_stochastic * 100

    return {
        "grape": grape_metrics,
        "stochastic": stoch_metrics,
        "T_improvement_pct": T_improvement,
        "n_improvement_pct": n_improvement,
    }


if __name__ == "__main__":
    from src.floquet import create_bang_bang_cycle
    from src.baseline import compute_stochastic_limit

    # System
    params = SystemParams(kappa=0.05, gamma1=0.01, T_bath=0.5, T_atom=0.05)

    # Create cycle (simulated GRAPE result)
    cycle = create_bang_bang_cycle(
        T_cycle=0.5,
        n_steps=20,
        g_on=0.8,
        g_off=0.2,
        delta_on=0.1,
        delta_off=-0.1,
    )

    # Simulated results
    n_grape = 1.0  # From GRAPE

    stoch_params = StochasticParams(
        omega_c=5.0,
        omega_a=5.0,
        kappa=0.05,
        T_bath=0.5,
        T_atom=0.05,
        lambda_ex=5.0,
        g=0.5,
        tau=0.05,
        R=5.0,
        chi=2.0,
    )
    n_stoch, T_stoch = compute_stochastic_limit(stoch_params)

    print("GRAPE Result:")
    grape_metrics = compute_thermodynamics(params, cycle, n_grape)
    print_thermodynamic_report(grape_metrics)

    print("\nStochastic Baseline:")
    stoch_metrics = compute_thermodynamics(params, cycle, n_stoch)
    print_thermodynamic_report(stoch_metrics)

    print("\nComparison:")
    analysis = analyze_grape_result(params, cycle, n_grape, n_stoch)
    print(f"n improvement: {analysis['n_improvement_pct']:.1f}%")
    print(f"T improvement: {analysis['T_improvement_pct']:.1f}%")
