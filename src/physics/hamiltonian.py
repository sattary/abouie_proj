"""
Time-dependent Hamiltonian for cavity-qubit system.
Supports dual control: coupling g(t) and detuning Δ(t).
"""

import jax.numpy as jnp
from typing import Callable, Tuple
from .operators import Operators, SystemParams


def build_hamiltonian_func(
    ops: Operators,
    params: SystemParams,
) -> Callable[[float, Callable, Callable], jnp.ndarray]:
    """
    Returns a function H(t, g_func, delta_func) that computes the Hamiltonian.

    The full Hamiltonian in the lab frame is:
        H = ω_c a†a + (ω_a/2) σz1 + (ω_a/2) σz2 + g(t)(a†S- + aS+)

    In the interaction picture (rotating at ω_c), with detuning Δ(t) = ω_a(t) - ω_c:
        H_int = (Δ(t)/2)(σz1 + σz2) + g(t)(a†S- + aS+)

    This is the form we implement for control.
    """
    # Pre-compute static parts
    V_jc = ops.V_jc  # a†S- + aS+
    sz_total = ops.sz1 + ops.sz2  # σz1 + σz2

    def hamiltonian(t: float, g_func: Callable, delta_func: Callable) -> jnp.ndarray:
        """
        Compute H(t) given control functions.

        Args:
            t: Current time
            g_func: Function returning coupling strength g(t)
            delta_func: Function returning detuning Δ(t)

        Returns:
            Hamiltonian matrix (dim x dim)
        """
        g_t = g_func(t)
        delta_t = delta_func(t)

        # H = (Δ/2)(σz1 + σz2) + g(a†S- + aS+)
        H = 0.5 * delta_t * sz_total + g_t * V_jc

        return H

    return hamiltonian


def build_hamiltonian_from_arrays(
    ops: Operators,
    t_points: jnp.ndarray,
    g_array: jnp.ndarray,
    delta_array: jnp.ndarray,
) -> Callable[[float], jnp.ndarray]:
    """
    Build Hamiltonian function from discrete control arrays.
    Uses linear interpolation for smooth pulses.

    Args:
        ops: Operators object
        t_points: Time points (shape: n_points)
        g_array: Coupling values at each time (shape: n_points)
        delta_array: Detuning values at each time (shape: n_points)

    Returns:
        H(t) function
    """
    V_jc = ops.V_jc
    sz_total = ops.sz1 + ops.sz2

    def hamiltonian(t: float) -> jnp.ndarray:
        g_t = jnp.interp(t, t_points, g_array)
        delta_t = jnp.interp(t, t_points, delta_array)
        return 0.5 * delta_t * sz_total + g_t * V_jc

    return hamiltonian


def compute_commutator_norm(H1: jnp.ndarray, H2: jnp.ndarray) -> float:
    """
    Compute ||[H(t1), H(t2)]||_F (Frobenius norm).

    Non-zero commutator is the KEY condition for beating the stochastic limit.
    This quantifies the "coherent advantage" of the Floquet protocol.
    """
    comm = H1 @ H2 - H2 @ H1
    return jnp.sqrt(jnp.sum(jnp.abs(comm) ** 2))


def check_floquet_condition(
    ops: Operators,
    t_points: jnp.ndarray,
    g_array: jnp.ndarray,
    delta_array: jnp.ndarray,
) -> Tuple[float, bool]:
    """
    Verify the Floquet hypothesis: [H(t1), H(t2)] != 0 for some t1, t2.

    Returns:
        max_comm_norm: Maximum commutator norm over all pairs
        can_beat_stochastic: True if non-trivial commutator exists
    """
    H_func = build_hamiltonian_from_arrays(ops, t_points, g_array, delta_array)

    max_norm = 0.0
    n_samples = min(len(t_points), 20)  # Sample subset for efficiency
    indices = jnp.linspace(0, len(t_points) - 1, n_samples, dtype=jnp.int32)

    for i in range(n_samples):
        t1 = t_points[indices[i]]
        H1 = H_func(t1)
        for j in range(i + 1, n_samples):
            t2 = t_points[indices[j]]
            H2 = H_func(t2)
            norm = compute_commutator_norm(H1, H2)
            max_norm = jnp.maximum(max_norm, norm)

    # Threshold for "non-trivial" commutator
    threshold = 1e-6
    can_beat = float(max_norm) > threshold

    return float(max_norm), can_beat


if __name__ == "__main__":
    from .operators import build_operators

    params = SystemParams()
    ops = build_operators(params)

    # Test 1: Constant controls (should COMMUTE)
    t_pts = jnp.linspace(0, 10, 50)
    g_const = jnp.ones(50) * 0.5
    delta_const = jnp.zeros(50)  # Resonant

    max_comm, can_beat = check_floquet_condition(ops, t_pts, g_const, delta_const)
    print(f"Constant g, Δ=0: max ||[H,H']|| = {max_comm:.6f}, can beat? {can_beat}")

    # Test 2: Time-varying detuning (should NOT commute)
    delta_varying = jnp.sin(t_pts * 0.5) * 0.1  # Oscillating detuning

    max_comm, can_beat = check_floquet_condition(ops, t_pts, g_const, delta_varying)
    print(f"Varying Δ(t): max ||[H,H']|| = {max_comm:.6f}, can beat? {can_beat}")

    # Test 3: Time-varying g AND delta
    g_varying = jnp.sin(t_pts * 0.3) * 0.3 + 0.5

    max_comm, can_beat = check_floquet_condition(ops, t_pts, g_varying, delta_varying)
    print(f"Varying g(t) & Δ(t): max ||[H,H']|| = {max_comm:.6f}, can beat? {can_beat}")
