"""
Initial quantum states for cavity-qubit system.
"""

import jax.numpy as jnp
from .operators import SystemParams, thermal_occupation, build_operators


def thermal_state(n_levels: int, n_bar: float) -> jnp.ndarray:
    """
    Build a thermal density matrix for a harmonic oscillator.

    ρ_th = Σ_n p_n |n><n|
    where p_n = (1 - e^{-β}) * e^{-nβ} = n̄^n / (1+n̄)^{n+1}
    """
    if n_bar <= 0:
        # Ground state
        rho = jnp.zeros((n_levels, n_levels))
        rho = rho.at[0, 0].set(1.0)
        return rho

    # Boltzmann distribution: p_n = (n̄/(1+n̄))^n / (1+n̄)
    # Equivalent to geometric distribution
    r = n_bar / (1 + n_bar)
    p = jnp.array([(1 - r) * r**n for n in range(n_levels)])
    p = p / jnp.sum(p)  # Normalize (truncation correction)

    return jnp.diag(p)


def ground_state_qubit() -> jnp.ndarray:
    """
    Qubit ground state |g><g|.
    Convention: |g> = [1, 0]^T, |e> = [0, 1]^T
    """
    return jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)


def excited_state_qubit() -> jnp.ndarray:
    """Qubit excited state |e><e|."""
    return jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)


def thermal_cavity_ground_qubits(params: SystemParams) -> jnp.ndarray:
    """
    Standard initial state: thermal cavity + both qubits in ground state.

    ρ_init = ρ_cav(T_bath) ⊗ |g><g| ⊗ |g><g|
    """
    # Cavity thermal state at bath temperature
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    rho_cav = thermal_state(params.n_fock, n_bar)

    # Ground state qubits
    rho_q = ground_state_qubit()

    # Full tensor product
    rho_init = jnp.kron(rho_cav, jnp.kron(rho_q, rho_q))

    return rho_init.astype(jnp.complex64)


def cold_cavity_ground_qubits(params: SystemParams) -> jnp.ndarray:
    """
    Cavity in ground state + both qubits in ground state.
    Used for testing / ideal starting point.
    """
    rho_cav = jnp.zeros((params.n_fock, params.n_fock))
    rho_cav = rho_cav.at[0, 0].set(1.0)
    rho_q = ground_state_qubit()

    return jnp.kron(rho_cav, jnp.kron(rho_q, rho_q)).astype(jnp.complex64)


def hot_cavity_ground_qubits(n_fock: int, n_target: float) -> jnp.ndarray:
    """
    Cavity with specified mean photon number + ground qubits.

    Args:
        n_fock: Number of Fock states
        n_target: Target mean photon number
    """
    rho_cav = thermal_state(n_fock, n_target)
    rho_q = ground_state_qubit()

    return jnp.kron(rho_cav, jnp.kron(rho_q, rho_q)).astype(jnp.complex64)


def correlated_qubit_pair(
    params: SystemParams,
    lambda_exchange: float = 0.0,
) -> jnp.ndarray:
    """
    Correlated two-qubit state from XY Heisenberg model (paper Eq. 2-3).

    This is the state of the qubit reservoir in the paper's collision model.

    ρ_pair = (1/Z) * exp(-β * H_pair)
    H_pair = ω(σz1 + σz2)/2 + λ(σ+σ- + σ-σ+)

    Args:
        params: System parameters (uses T_atom, omega_a)
        lambda_exchange: Exchange coupling strength

    Returns:
        4x4 density matrix for qubit pair
    """
    omega = params.omega_a
    T = params.T_atom

    if T <= 0:
        # Zero temperature: ground state |gg>
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=jnp.complex64,
        )

    beta = 1.0 / (T * 0.0208)  # k_B in units where omega is in GHz, T in K
    # Actually: k_B = 0.0208 meV/K, and we need hbar*omega/k_B
    # For omega in GHz: hbar*omega = 4.136 ueV/GHz
    # k_B*T in meV = 0.0862 * T[K]
    # So beta*hbar*omega = (omega * 4.136e-3 meV/GHz) / (0.0862 * T meV/K)
    # = omega * 0.048 / T
    beta_hw = omega * 0.048 / T
    beta_hl = lambda_exchange * 0.048 / T if lambda_exchange != 0 else 0.0

    # Partition function (Eq. 3)
    Z = 2 * (jnp.cosh(beta_hw) + jnp.cosh(beta_hl))

    # State weights (Eq. 2)
    rho_e = jnp.exp(-beta_hw) / Z  # |ee> weight
    rho_g = jnp.exp(beta_hw) / Z  # |gg> weight
    rho_d = jnp.cosh(beta_hl) / Z  # Diagonal in {|eg>, |ge|}
    rho_nd = -jnp.sinh(beta_hl) / Z  # Off-diagonal coherence

    # Density matrix in basis {|ee>, |eg>, |ge>, |gg>}
    rho = jnp.array(
        [
            [rho_e, 0, 0, 0],
            [0, rho_d, rho_nd, 0],
            [0, rho_nd, rho_d, 0],
            [0, 0, 0, rho_g],
        ],
        dtype=jnp.complex64,
    )

    return rho


if __name__ == "__main__":
    params = SystemParams(T_bath=1.0, T_atom=0.05, omega_c=5.0, n_fock=5)
    ops = build_operators(params)

    # Test thermal cavity state
    rho_init = thermal_cavity_ground_qubits(params)
    print(f"Initial state shape: {rho_init.shape}")
    print(f"Trace: {jnp.trace(rho_init):.6f}")

    # Compute initial occupation
    n_init = jnp.real(jnp.trace(ops.n_cav @ rho_init))
    n_bar_expected = thermal_occupation(params.omega_c, params.T_bath)
    print(f"Initial <n>: {n_init:.4f}")
    print(f"Expected n̄ at T={params.T_bath}K: {n_bar_expected:.4f}")

    # Test correlated pair
    rho_pair = correlated_qubit_pair(params, lambda_exchange=0.5)
    print(f"\nCorrelated pair state:")
    print(rho_pair)
    print(f"Pair trace: {jnp.trace(rho_pair):.6f}")
