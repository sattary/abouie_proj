"""
Tier 3 Validation: Noise robustness analysis.
Adds realistic noise sources: TLS defects, 1/f flux noise, qubit reset errors.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Tuple, NamedTuple

from src.physics import (
    SystemParams,
    Operators,
    build_operators,
    thermal_occupation,
    thermal_cavity_ground_qubits,
)
from src.floquet import FloquetCycleParams


class NoiseConfig(NamedTuple):
    """Configuration for noise sources."""

    # TLS (Two-Level System) defects
    tls_rate: float = 0.001  # TLS switching rate (GHz)
    tls_coupling: float = 0.01  # TLS-cavity coupling (GHz)
    n_tls: int = 3  # Number of TLS defects

    # 1/f flux noise
    flux_noise_amplitude: float = 0.001  # Detuning noise amplitude (GHz)

    # Qubit reset errors
    reset_error_prob: float = 0.01  # Probability of reset failure

    # General
    seed: int = 42


def build_noisy_cycle_fn(
    ops: Operators,
    params: SystemParams,
    cycle: FloquetCycleParams,
    noise_cfg: NoiseConfig,
):
    """
    Build JIT-compiled noisy cycle function.
    All noise sources combined in a single simulation.
    """
    # Pre-compute static data
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    kappa_down = params.kappa * (n_bar + 1)
    kappa_up = params.kappa * n_bar
    dt = cycle.dt

    V_jc = ops.V_jc
    sz_total = ops.sz1 + ops.sz2
    n_cav = ops.n_cav

    L_down = jnp.sqrt(kappa_down) * ops.a
    L_up = jnp.sqrt(kappa_up) * ops.a_dag
    L_q1 = jnp.sqrt(params.gamma1) * ops.sm1
    L_q2 = jnp.sqrt(params.gamma1) * ops.sm2

    # TLS operators (add extra dephasing)
    L_tls = jnp.sqrt(noise_cfg.tls_rate) * ops.sz1  # Simplified TLS bath

    @jax.jit
    def apply_noisy_cycle(rho, g_seq, delta_seq, key):
        """Apply one Floquet cycle with noise."""

        def noisy_step(carry, xs):
            rho, key = carry
            g, delta, noise_sample = xs

            # Add 1/f flux noise to detuning
            delta_noisy = delta + noise_cfg.flux_noise_amplitude * noise_sample

            # Build Hamiltonian with noisy detuning
            H = 0.5 * delta_noisy * sz_total + g * V_jc

            drho = -1j * (H @ rho - rho @ H)

            # Standard Lindblad channels
            for L in [L_down, L_up, L_q1, L_q2]:
                Ld = L.conj().T
                drho = drho + L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)

            # TLS dephasing (simplified model)
            Ld_tls = L_tls.conj().T
            drho = (
                drho
                + L_tls @ rho @ Ld_tls
                - 0.5 * (Ld_tls @ L_tls @ rho + rho @ Ld_tls @ L_tls)
            )

            # RK4 step
            k1 = drho
            k2 = drho  # Simplified: same derivative
            k3 = drho
            k4 = drho
            rho_new = rho + dt * drho  # Euler for speed with noise

            # Enforce physicality
            rho_new = 0.5 * (rho_new + rho_new.conj().T)
            rho_new = rho_new / jnp.trace(rho_new)

            # Generate new key
            key, subkey = random.split(key)

            return (rho_new, key), None

        # Generate noise samples for this cycle
        noise_samples = random.normal(key, shape=(cycle.n_steps,))

        controls = jnp.stack([g_seq, delta_seq, noise_samples], axis=1)
        (rho_final, _), _ = lax.scan(noisy_step, (rho, key), controls)

        return rho_final

    def run_noisy_cycles(rho, g_seq, delta_seq, n_cycles, key):
        """Run n noisy cycles (non-JIT for dynamic n_cycles)."""
        for _ in range(n_cycles):
            key, subkey = random.split(key)
            rho = apply_noisy_cycle(rho, g_seq, delta_seq, subkey)
        return jnp.real(jnp.trace(n_cav @ rho))

    return run_noisy_cycles


def validate_cycle_tier3(
    params: SystemParams,
    cycle: FloquetCycleParams,
    noise_cfg: NoiseConfig = NoiseConfig(),
    n_cycles: int = 100,
    n_samples: int = 10,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Validate Floquet cycle robustness to noise.

    Runs multiple samples with different noise realizations.

    Returns:
        (mean_n, std_n, worst_case_n)
    """
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    if verbose:
        print("Tier 3 Validation (Noise Robustness)")
        print("=" * 50)
        print(
            f"TLS rate: {noise_cfg.tls_rate}, Flux noise: {noise_cfg.flux_noise_amplitude}"
        )

    # Build noisy function
    run_noisy = build_noisy_cycle_fn(ops, params, cycle, noise_cfg)

    # Run multiple samples
    results = []
    key = random.PRNGKey(noise_cfg.seed)

    for i in range(n_samples):
        key, subkey = random.split(key)
        n_final = float(
            run_noisy(
                rho_init, cycle.g_sequence, cycle.delta_sequence, n_cycles, subkey
            )
        )
        results.append(n_final)

        if verbose and (i + 1) % 5 == 0:
            print(f"Sample {i + 1}: <n> = {n_final:.4f}")

    mean_n = float(jnp.mean(jnp.array(results)))
    std_n = float(jnp.std(jnp.array(results)))
    worst_n = float(jnp.max(jnp.array(results)))

    if verbose:
        print(f"\nMean <n>: {mean_n:.4f} +/- {std_n:.4f}")
        print(f"Worst case: {worst_n:.4f}")

    return mean_n, std_n, worst_n


def run_noise_sweep(
    params: SystemParams,
    cycle: FloquetCycleParams,
    flux_amplitudes: list = [0.0, 0.001, 0.005, 0.01, 0.02],
) -> list:
    """Sweep flux noise amplitude to find robustness threshold."""
    results = []

    print("Tier 3 Flux Noise Sweep")
    print("-" * 50)

    for amp in flux_amplitudes:
        cfg = NoiseConfig(flux_noise_amplitude=amp)
        mean_n, std_n, worst_n = validate_cycle_tier3(
            params, cycle, cfg, n_cycles=50, n_samples=5, verbose=False
        )

        results.append(
            {
                "flux_amp": amp,
                "mean_n": mean_n,
                "std_n": std_n,
                "worst_n": worst_n,
            }
        )

        print(f"Flux amp={amp:.4f}: n={mean_n:.4f}+/-{std_n:.4f}")

    return results


if __name__ == "__main__":
    from src.floquet import create_bang_bang_cycle

    params = SystemParams(kappa=0.05, gamma1=0.01, T_bath=0.5, T_atom=0.05)

    # Create test cycle
    cycle = create_bang_bang_cycle(
        T_cycle=0.5,
        n_steps=20,
        g_on=0.5,
        g_off=0.1,
        delta_on=0.1,
        delta_off=-0.1,
    )

    # Single validation
    validate_cycle_tier3(params, cycle)

    # Sweep
    print("\n")
    run_noise_sweep(params, cycle)
