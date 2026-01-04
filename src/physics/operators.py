"""
Static quantum operators for cavity-qubit system.
All operators are pre-computed as JAX arrays for efficiency.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple


class SystemParams(NamedTuple):
    """Physical parameters for the cavity-qubit system."""

    n_fock: int = 5  # Number of Fock states for cavity
    n_qubits: int = 2  # Number of qubits (always 2 for this paper)
    omega_c: float = 5.0  # Cavity frequency (GHz)
    omega_a: float = 5.0  # Qubit frequency (GHz)
    g_max: float = 0.5  # Max coupling strength (GHz)
    kappa: float = 0.01  # Cavity decay rate (GHz)
    gamma1: float = 0.001  # Qubit T1 decay rate (GHz)
    gamma_phi: float = 0.002  # Qubit dephasing rate (GHz)
    T_bath: float = 1.0  # Bath temperature (K)
    T_atom: float = 0.05  # Qubit reservoir temperature (K)


@dataclass
class Operators:
    """Pre-computed operators for the full Hilbert space."""

    # Cavity operators
    a: jnp.ndarray  # Annihilation operator
    a_dag: jnp.ndarray  # Creation operator
    n_cav: jnp.ndarray  # Number operator

    # Qubit 1 operators
    sm1: jnp.ndarray  # Sigma minus
    sp1: jnp.ndarray  # Sigma plus
    sz1: jnp.ndarray  # Sigma z

    # Qubit 2 operators
    sm2: jnp.ndarray
    sp2: jnp.ndarray
    sz2: jnp.ndarray

    # Collective operators
    S_minus: jnp.ndarray  # sm1 + sm2
    S_plus: jnp.ndarray  # sp1 + sp2

    # Interaction operator (for Jaynes-Cummings)
    V_jc: jnp.ndarray  # a_dag @ S_minus + a @ S_plus

    # Identity
    eye: jnp.ndarray

    # Dimensions
    dim: int


def build_operators(params: SystemParams = SystemParams()) -> Operators:
    """
    Build all static operators for the cavity + 2-qubit system.

    Hilbert space ordering: |cavity> ⊗ |qubit1> ⊗ |qubit2>
    Total dimension: n_fock * 2 * 2
    """
    n_fock = params.n_fock
    dim = n_fock * 4  # 4 = 2 * 2 for two qubits

    # Cavity annihilation operator in Fock basis
    # a|n> = sqrt(n)|n-1>
    a_cav = jnp.diag(jnp.sqrt(jnp.arange(1, n_fock, dtype=jnp.float32)), k=1)

    # Qubit operators (2x2)
    sm_q = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)  # |g><e|
    sp_q = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)  # |e><g|
    sz_q = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float32)  # |e><e| - |g><g|

    # Identity matrices
    id_c = jnp.eye(n_fock, dtype=jnp.float32)
    id_q = jnp.eye(2, dtype=jnp.float32)

    # Tensor products to full space: |cavity> ⊗ |q1> ⊗ |q2>
    # A = a ⊗ I ⊗ I
    a = jnp.kron(a_cav, jnp.kron(id_q, id_q))
    a_dag = a.T
    n_cav = a_dag @ a

    # SM1 = I ⊗ sm ⊗ I
    sm1 = jnp.kron(id_c, jnp.kron(sm_q, id_q))
    sp1 = sm1.T
    sz1 = jnp.kron(id_c, jnp.kron(sz_q, id_q))

    # SM2 = I ⊗ I ⊗ sm
    sm2 = jnp.kron(id_c, jnp.kron(id_q, sm_q))
    sp2 = sm2.T
    sz2 = jnp.kron(id_c, jnp.kron(id_q, sz_q))

    # Collective qubit operators
    S_minus = sm1 + sm2
    S_plus = sp1 + sp2

    # Jaynes-Cummings interaction: a†S- + aS+
    V_jc = a_dag @ S_minus + a @ S_plus

    # Full identity
    eye = jnp.eye(dim, dtype=jnp.float32)

    return Operators(
        a=a,
        a_dag=a_dag,
        n_cav=n_cav,
        sm1=sm1,
        sp1=sp1,
        sz1=sz1,
        sm2=sm2,
        sp2=sp2,
        sz2=sz2,
        S_minus=S_minus,
        S_plus=S_plus,
        V_jc=V_jc,
        eye=eye,
        dim=dim,
    )


def thermal_occupation(omega: float, T: float) -> float:
    """
    Bose-Einstein occupation number.

    n_bar = 1 / (exp(hbar*omega / k_B*T) - 1)

    Using natural units where hbar = k_B = 1, omega in GHz, T in K.
    Conversion: hbar*omega/k_B = omega[GHz] * 0.048 / T[K]
    """
    if T <= 0:
        return 0.0

    # hbar * omega / k_B in units of K (for omega in GHz)
    # hbar = 1.054e-34 J*s, k_B = 1.38e-23 J/K
    # For omega in rad/s: hbar*omega/k_B = omega * 7.6e-12 K/(rad/s)
    # For omega in GHz (2pi * 1e9 rad/s): factor = 0.048 K/GHz
    hbar_omega_over_kB = omega * 0.048  # in Kelvin

    x = hbar_omega_over_kB / T
    if x > 50:  # Avoid overflow
        return 0.0
    return 1.0 / (jnp.exp(x) - 1.0)


if __name__ == "__main__":
    # Quick test
    params = SystemParams()
    ops = build_operators(params)

    print(f"System dimension: {ops.dim}")
    print(f"Cavity a shape: {ops.a.shape}")
    print(f"V_jc shape: {ops.V_jc.shape}")

    # Check commutation [a, a†] = 1 (in subspace)
    comm = ops.a @ ops.a_dag - ops.a_dag @ ops.a
    print(f"[a, a†] trace (should be ~n_fock*4): {jnp.trace(comm):.2f}")

    # Check thermal occupation
    n_bar = thermal_occupation(5.0, 1.0)  # 5 GHz at 1 K
    print(f"Thermal occupation at 5 GHz, 1 K: {n_bar:.4f}")

    n_bar_cold = thermal_occupation(5.0, 0.05)  # 5 GHz at 50 mK
    print(f"Thermal occupation at 5 GHz, 50 mK: {n_bar_cold:.6f}")
