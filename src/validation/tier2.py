"""
Tier 2 Validation: Full TDSE without Rotating Wave Approximation (RWA).
Validates that Floquet cycles work with counter-rotating terms included.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple
from functools import partial

from src.physics import (
    SystemParams,
    Operators,
    build_operators,
    thermal_occupation,
    thermal_cavity_ground_qubits,
)
from src.floquet import FloquetCycleParams


def _build_tier2_cycle_fn(
    ops: Operators,
    params: SystemParams,
    dt: float,
    include_counter_rotating: bool,
):
    """
    Build JIT-compiled cycle function for Tier 2 validation.
    All operator-dependent logic is captured in closure.
    """
    # Pre-compute all static data
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    kappa_down = params.kappa * (n_bar + 1)
    kappa_up = params.kappa * n_bar

    V_jc = ops.V_jc
    sz_total = ops.sz1 + ops.sz2
    a = ops.a
    a_dag = ops.a_dag
    S_minus = ops.S_minus
    S_plus = ops.S_plus
    n_cav = ops.n_cav

    L_down = jnp.sqrt(kappa_down) * ops.a
    L_up = jnp.sqrt(kappa_up) * ops.a_dag
    L_q1 = jnp.sqrt(params.gamma1) * ops.sm1
    L_q2 = jnp.sqrt(params.gamma1) * ops.sm2

    @jax.jit
    def apply_cycle(rho, g_seq, delta_seq):
        """Apply one Floquet cycle."""

        def lindblad_rhs(r, g, delta):
            # Build Hamiltonian
            H_rwa = 0.5 * delta * sz_total + g * V_jc
            if include_counter_rotating:
                H_counter = g * (a @ S_minus + a_dag @ S_plus)
                H = H_rwa + H_counter
            else:
                H = H_rwa

            drho = -1j * (H @ r - r @ H)

            for L in [L_down, L_up, L_q1, L_q2]:
                Ld = L.conj().T
                drho = drho + L @ r @ Ld - 0.5 * (Ld @ L @ r + r @ Ld @ L)

            return drho

        def rk4_step(rho, controls):
            g, delta = controls[0], controls[1]

            k1 = lindblad_rhs(rho, g, delta)
            k2 = lindblad_rhs(rho + 0.5 * dt * k1, g, delta)
            k3 = lindblad_rhs(rho + 0.5 * dt * k2, g, delta)
            k4 = lindblad_rhs(rho + dt * k3, g, delta)

            rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho_new = 0.5 * (rho_new + rho_new.conj().T)
            rho_new = rho_new / jnp.trace(rho_new)

            return rho_new, None

        controls = jnp.stack([g_seq, delta_seq], axis=1)
        rho_final, _ = lax.scan(rk4_step, rho, controls)
        return rho_final

    @jax.jit
    def run_n_cycles(rho, g_seq, delta_seq, n_cycles):
        """Run n cycles and return final occupation."""
        rho_final = lax.fori_loop(
            0, n_cycles, lambda i, r: apply_cycle(r, g_seq, delta_seq), rho
        )
        return jnp.real(jnp.trace(n_cav @ rho_final))

    return run_n_cycles


def validate_cycle_tier2(
    params: SystemParams,
    cycle: FloquetCycleParams,
    n_cycles: int = 200,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Validate a Floquet cycle using full TDSE (no RWA).

    Compares results with and without counter-rotating terms.

    Returns:
        (n_rwa, n_full, relative_error)
    """
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    if verbose:
        print("Tier 2 Validation (Full TDSE)")
        print("=" * 50)

    # Build JIT-compiled functions for each mode
    run_rwa = _build_tier2_cycle_fn(
        ops, params, cycle.dt, include_counter_rotating=False
    )
    run_full = _build_tier2_cycle_fn(
        ops, params, cycle.dt, include_counter_rotating=True
    )

    # Run RWA version
    n_rwa = float(run_rwa(rho_init, cycle.g_sequence, cycle.delta_sequence, n_cycles))
    if verbose:
        print(f"RWA result: <n> = {n_rwa:.4f}")

    # Run full version with counter-rotating terms
    n_full = float(run_full(rho_init, cycle.g_sequence, cycle.delta_sequence, n_cycles))
    if verbose:
        print(f"Full TDSE: <n> = {n_full:.4f}")

    # Compute relative error
    if abs(n_rwa) > 1e-6:
        rel_error = abs(n_full - n_rwa) / abs(n_rwa)
    else:
        rel_error = abs(n_full - n_rwa)

    if verbose:
        print(f"Relative error: {rel_error * 100:.2f}%")
        if rel_error < 0.1:
            print("PASS: Counter-rotating terms have <10% effect")
        else:
            print("WARNING: Counter-rotating terms significant")

    return n_rwa, n_full, rel_error


def run_tier2_sweep(
    params: SystemParams,
    cycle: FloquetCycleParams,
    g_multipliers: list = [0.5, 1.0, 1.5, 2.0],
) -> list:
    """
    Sweep g(t) strength to find where RWA breaks down.
    Strong driving increases counter-rotating term effects.
    """
    results = []

    print("Tier 2 g-sweep")
    print("-" * 50)

    for mult in g_multipliers:
        scaled_cycle = FloquetCycleParams(
            T_cycle=cycle.T_cycle,
            n_steps=cycle.n_steps,
            g_sequence=cycle.g_sequence * mult,
            delta_sequence=cycle.delta_sequence,
        )

        n_rwa, n_full, err = validate_cycle_tier2(
            params, scaled_cycle, n_cycles=100, verbose=False
        )
        results.append(
            {
                "g_mult": mult,
                "n_rwa": n_rwa,
                "n_full": n_full,
                "rel_error": err,
            }
        )

        print(f"g*{mult:.1f}: RWA={n_rwa:.4f}, Full={n_full:.4f}, err={err * 100:.1f}%")

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
    n_rwa, n_full, err = validate_cycle_tier2(params, cycle, n_cycles=100)

    # Sweep to find RWA breakdown
    print("\n")
    results = run_tier2_sweep(params, cycle)
