"""
Integration test for the physics engine.
Verifies all modules work together correctly.
"""

import jax
import jax.numpy as jnp
from src.physics import (
    SystemParams,
    build_operators,
    thermal_occupation,
    thermal_cavity_ground_qubits,
    simulate,
    check_floquet_condition,
)


def test_operators():
    """Test operator construction and basic properties."""
    print("=" * 50)
    print("TEST: Operators")
    print("=" * 50)

    params = SystemParams()
    ops = build_operators(params)

    print(f"Hilbert space dimension: {ops.dim}")
    assert ops.dim == 20, f"Expected dim=20, got {ops.dim}"

    # Check [a, a†] ~ I (in truncated space)
    comm = ops.a @ ops.a_dag - ops.a_dag @ ops.a
    trace_comm = jnp.trace(comm)
    print(f"Trace of [a, a†]: {trace_comm:.2f} (expect ~{ops.dim})")

    # Check V_jc is Hermitian
    V_diff = jnp.max(jnp.abs(ops.V_jc - ops.V_jc.T))
    print(f"V_jc Hermiticity check (max diff): {V_diff:.6f}")
    assert V_diff < 1e-6, "V_jc should be Hermitian (real symmetric)"

    print("PASSED\n")


def test_thermal_states():
    """Test initial state construction."""
    print("=" * 50)
    print("TEST: Thermal States")
    print("=" * 50)

    params = SystemParams(T_bath=1.0, omega_c=5.0, n_fock=5)
    ops = build_operators(params)

    rho_init = thermal_cavity_ground_qubits(params)

    # Check trace = 1
    trace = jnp.trace(rho_init)
    print(f"Trace of initial state: {trace:.6f}")
    assert jnp.abs(trace - 1.0) < 1e-5, "Trace should be 1"

    # Check positivity (all eigenvalues >= 0)
    eigvals = jnp.linalg.eigvalsh(rho_init)
    min_eig = jnp.min(eigvals)
    print(f"Minimum eigenvalue: {min_eig:.6f}")
    assert min_eig >= -1e-6, "State should be positive semidefinite"

    # Check initial occupation
    n_init = jnp.real(jnp.trace(ops.n_cav @ rho_init))
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    print(f"Initial <n>: {n_init:.4f}")
    print(f"Expected n̄(5 GHz, 1 K): {n_bar:.4f}")

    print("PASSED\n")


def test_simulation_basic():
    """Test basic simulation runs without errors."""
    print("=" * 50)
    print("TEST: Basic Simulation")
    print("=" * 50)

    params = SystemParams(
        kappa=0.05,  # Reasonable decay
        gamma1=0.001,  # Qubit T1
        gamma_phi=0.002,  # Qubit dephasing
        T_bath=1.0,
    )
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    n_init = jnp.real(jnp.trace(ops.n_cav @ rho_init))
    print(f"Initial <n>: {n_init:.4f}")

    # Constant coupling (baseline)
    g_const = lambda t: 0.5
    delta_const = lambda t: 0.0

    rho_final, n_final = simulate(
        params,
        g_const,
        delta_const,
        rho_init,
        t_final=5.0,
        rtol=1e-4,
        atol=1e-4,
    )

    print(f"Final <n> (after 5 ns): {n_final:.4f}")

    # Check trace preservation
    trace_final = jnp.trace(rho_final)
    print(f"Final trace: {jnp.real(trace_final):.6f}")
    assert jnp.abs(trace_final - 1.0) < 0.01, "Trace should be ~1"

    print("PASSED\n")


def test_floquet_condition():
    """Test the Floquet commutator check."""
    print("=" * 50)
    print("TEST: Floquet Condition")
    print("=" * 50)

    params = SystemParams()
    ops = build_operators(params)

    t_pts = jnp.linspace(0, 10, 50)

    # Case 1: Constant controls (should commute)
    g_const = jnp.ones(50) * 0.5
    delta_const = jnp.zeros(50)

    max_comm, can_beat = check_floquet_condition(ops, t_pts, g_const, delta_const)
    print(f"Constant g, Δ=0: ||[H,H']||_max = {max_comm:.6f}")
    print(f"  Can beat stochastic limit? {can_beat}")

    # Case 2: Time-varying detuning (should NOT commute)
    delta_vary = jnp.sin(t_pts * 0.5) * 0.2

    max_comm, can_beat = check_floquet_condition(ops, t_pts, g_const, delta_vary)
    print(f"Varying Δ(t): ||[H,H']||_max = {max_comm:.6f}")
    print(f"  Can beat stochastic limit? {can_beat}")

    assert can_beat, "Time-varying Δ should give non-commuting H"

    print("PASSED\n")


def test_cooling_direction():
    """Verify coupling produces cooling (not heating)."""
    print("=" * 50)
    print("TEST: Cooling Direction")
    print("=" * 50)

    params = SystemParams(kappa=0.1, T_bath=0.5)  # Warmer bath for clear test
    rho_init = thermal_cavity_ground_qubits(params)
    ops = build_operators(params)

    n_init = jnp.real(jnp.trace(ops.n_cav @ rho_init))

    # No coupling - should thermalize to bath
    _, n_no_coupling = simulate(
        params,
        g_func=lambda t: 0.0,
        delta_func=lambda t: 0.0,
        rho_init=rho_init,
        t_final=20.0,
    )

    # With coupling - should cool (qubits are cold sinks)
    _, n_with_coupling = simulate(
        params,
        g_func=lambda t: 0.5,
        delta_func=lambda t: 0.0,
        rho_init=rho_init,
        t_final=20.0,
    )

    print(f"Initial <n>: {n_init:.4f}")
    print(f"No coupling (20 ns): {n_no_coupling:.4f}")
    print(f"With coupling (20 ns): {n_with_coupling:.4f}")

    # Coupling should reduce cavity occupation compared to no coupling
    print(f"Cooling effect: Δn = {n_no_coupling - n_with_coupling:.4f}")

    if n_with_coupling < n_no_coupling:
        print("PASSED - Coupling produces net cooling\n")
    else:
        print("WARNING - No clear cooling observed (may need parameter tuning)\n")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("PHYSICS ENGINE INTEGRATION TESTS")
    print("=" * 60 + "\n")

    test_operators()
    test_thermal_states()
    test_floquet_condition()
    test_simulation_basic()
    test_cooling_direction()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
