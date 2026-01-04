"""
Lindblad master equation dynamics with full thermal dissipation.
Includes cavity decay, qubit T1/T2, and thermal bath heating.
"""

import jax
import jax.numpy as jnp
import diffrax
from diffrax import PIDController
from typing import Callable, Tuple, NamedTuple
from .operators import Operators, SystemParams, thermal_occupation, build_operators


class DissipationRates(NamedTuple):
    """Dissipation rates for the system."""

    kappa_down: float  # Cavity decay rate (κ(n̄+1))
    kappa_up: float  # Cavity heating rate (κn̄)
    gamma1_q1: float  # Qubit 1 T1 decay
    gamma1_q2: float  # Qubit 2 T1 decay
    gamma_phi_q1: float  # Qubit 1 dephasing
    gamma_phi_q2: float  # Qubit 2 dephasing


def compute_dissipation_rates(params: SystemParams) -> DissipationRates:
    """
    Compute physical dissipation rates from system parameters.
    Includes thermal occupation for proper detailed balance.
    """
    # Thermal occupation of cavity mode at bath temperature
    n_bar = thermal_occupation(params.omega_c, params.T_bath)

    # Cavity rates (satisfying detailed balance)
    kappa_down = params.kappa * (n_bar + 1)  # Emission: D[a]
    kappa_up = params.kappa * n_bar  # Absorption: D[a†]

    return DissipationRates(
        kappa_down=kappa_down,
        kappa_up=kappa_up,
        gamma1_q1=params.gamma1,
        gamma1_q2=params.gamma1,
        gamma_phi_q1=params.gamma_phi,
        gamma_phi_q2=params.gamma_phi,
    )


def build_lindblad_superoperator(
    ops: Operators,
    rates: DissipationRates,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Build the dissipative part of the Lindblad equation.

    L_D[ρ] = Σ_k γ_k (L_k ρ L_k† - 0.5{L_k†L_k, ρ})

    Channels:
    1. Cavity emission: sqrt(κ_down) * a
    2. Cavity absorption: sqrt(κ_up) * a†
    3. Qubit 1 decay: sqrt(γ1) * σ-_1
    4. Qubit 2 decay: sqrt(γ1) * σ-_2
    5. Qubit 1 dephasing: sqrt(γφ/2) * σz_1
    6. Qubit 2 dephasing: sqrt(γφ/2) * σz_2
    """
    # Pre-compute jump operators
    L_cav_down = jnp.sqrt(rates.kappa_down) * ops.a
    L_cav_up = jnp.sqrt(rates.kappa_up) * ops.a_dag
    L_q1_decay = jnp.sqrt(rates.gamma1_q1) * ops.sm1
    L_q2_decay = jnp.sqrt(rates.gamma1_q2) * ops.sm2
    L_q1_deph = jnp.sqrt(rates.gamma_phi_q1 / 2) * ops.sz1
    L_q2_deph = jnp.sqrt(rates.gamma_phi_q2 / 2) * ops.sz2

    # Pre-compute L†L for each channel
    def lindblad_term(L: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """D[L]ρ = LρL† - 0.5{L†L, ρ}"""
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        return L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)

    def dissipator(rho: jnp.ndarray) -> jnp.ndarray:
        """Total dissipation L_D[ρ]"""
        D = jnp.zeros_like(rho)

        # Cavity channels
        D = D + lindblad_term(L_cav_down, rho)
        D = D + lindblad_term(L_cav_up, rho)

        # Qubit decay channels
        D = D + lindblad_term(L_q1_decay, rho)
        D = D + lindblad_term(L_q2_decay, rho)

        # Qubit dephasing channels
        D = D + lindblad_term(L_q1_deph, rho)
        D = D + lindblad_term(L_q2_deph, rho)

        return D

    return dissipator


def build_master_equation(
    ops: Operators,
    params: SystemParams,
    g_func: Callable[[float], float],
    delta_func: Callable[[float], float],
) -> Callable[[float, jnp.ndarray, None], jnp.ndarray]:
    """
    Build the full Lindblad master equation:

    dρ/dt = -i[H(t), ρ] + L_D[ρ]

    Args:
        ops: Pre-computed operators
        params: System parameters
        g_func: Coupling control function g(t)
        delta_func: Detuning control function Δ(t)

    Returns:
        RHS function for ODE solver: f(t, rho, args) -> drho/dt
    """
    # Pre-compute static operators
    V_jc = ops.V_jc
    sz_total = ops.sz1 + ops.sz2

    # Build dissipator
    rates = compute_dissipation_rates(params)
    dissipator = build_lindblad_superoperator(ops, rates)

    def dynamics(t: float, rho: jnp.ndarray, args: None) -> jnp.ndarray:
        # Time-dependent Hamiltonian
        g_t = g_func(t)
        delta_t = delta_func(t)
        H = 0.5 * delta_t * sz_total + g_t * V_jc

        # Commutator: -i[H, ρ]
        comm = -1j * (H @ rho - rho @ H)

        # Full Lindblad equation
        return comm + dissipator(rho)

    return dynamics


def build_master_equation_real(
    ops: Operators,
    params: SystemParams,
    g_func: Callable[[float], float],
    delta_func: Callable[[float], float],
) -> Tuple[Callable, int]:
    """
    Build master equation in real representation for ODE solvers.

    Density matrix is split into real and imaginary parts:
    y = [Re(ρ).flatten(), Im(ρ).flatten()]

    Returns:
        dynamics function and state dimension
    """
    dim = ops.dim
    state_size = 2 * dim * dim

    # Pre-compute
    V_jc = ops.V_jc
    sz_total = ops.sz1 + ops.sz2
    rates = compute_dissipation_rates(params)

    # Pre-compute jump operators and their products
    # We need to handle complex arithmetic manually for efficiency
    L_list = [
        jnp.sqrt(rates.kappa_down) * ops.a,
        jnp.sqrt(rates.kappa_up) * ops.a_dag,
        jnp.sqrt(rates.gamma1_q1) * ops.sm1,
        jnp.sqrt(rates.gamma1_q2) * ops.sm2,
        jnp.sqrt(rates.gamma_phi_q1 / 2) * ops.sz1,
        jnp.sqrt(rates.gamma_phi_q2 / 2) * ops.sz2,
    ]

    # Split operators into real/imag
    V_r, V_i = jnp.real(V_jc), jnp.imag(V_jc)
    sz_r = jnp.real(sz_total)  # sz is real

    L_data = [(jnp.real(L), jnp.imag(L), jnp.real(L.T), jnp.imag(L.T)) for L in L_list]
    LdL_data = [(jnp.real(L.conj().T @ L), jnp.imag(L.conj().T @ L)) for L in L_list]

    def dynamics(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        # Unpack state
        rho_r = y[: dim * dim].reshape((dim, dim))
        rho_i = y[dim * dim :].reshape((dim, dim))

        # Controls
        g_t = g_func(t)
        delta_t = delta_func(t)

        # Hamiltonian (real for JC, sz is real)
        Hr = 0.5 * delta_t * sz_r + g_t * V_r
        Hi = g_t * V_i

        # Commutator: -i[H, ρ] = -i(Hρ - ρH)
        # Let H = Hr + i*Hi, ρ = ρr + i*ρi
        # Hρ = (Hr*ρr - Hi*ρi) + i*(Hr*ρi + Hi*ρr)
        # ρH = (ρr*Hr - ρi*Hi) + i*(ρr*Hi + ρi*Hr)
        # [H,ρ] = Hρ - ρH
        # -i[H,ρ] real part = (Hr*ρi + Hi*ρr) - (ρr*Hi + ρi*Hr)
        # -i[H,ρ] imag part = -((Hr*ρr - Hi*ρi) - (ρr*Hr - ρi*Hi))

        Hrho_r = Hr @ rho_r - Hi @ rho_i
        Hrho_i = Hr @ rho_i + Hi @ rho_r
        rhoH_r = rho_r @ Hr - rho_i @ Hi
        rhoH_i = rho_r @ Hi + rho_i @ Hr

        comm_r = Hrho_i - rhoH_i
        comm_i = -(Hrho_r - rhoH_r)

        # Dissipation (accumulate)
        D_r = jnp.zeros_like(rho_r)
        D_i = jnp.zeros_like(rho_i)

        for (Lr, Li, Ldr, Ldi), (LdLr, LdLi) in zip(L_data, LdL_data):
            # L @ rho
            Lrho_r = Lr @ rho_r - Li @ rho_i
            Lrho_i = Lr @ rho_i + Li @ rho_r

            # (L @ rho) @ L†
            term1_r = Lrho_r @ Ldr - Lrho_i @ Ldi
            term1_i = Lrho_r @ Ldi + Lrho_i @ Ldr

            # L†L @ rho
            LdLrho_r = LdLr @ rho_r - LdLi @ rho_i
            LdLrho_i = LdLr @ rho_i + LdLi @ rho_r

            # rho @ L†L
            rhoLdL_r = rho_r @ LdLr - rho_i @ LdLi
            rhoLdL_i = rho_r @ LdLi + rho_i @ LdLr

            # D[L] = term1 - 0.5*(LdLrho + rhoLdL)
            D_r = D_r + term1_r - 0.5 * (LdLrho_r + rhoLdL_r)
            D_i = D_i + term1_i - 0.5 * (LdLrho_i + rhoLdL_i)

        # Total derivative
        drho_r = comm_r + D_r
        drho_i = comm_i + D_i

        return jnp.concatenate([drho_r.flatten(), drho_i.flatten()])

    return dynamics, state_size


def simulate(
    params: SystemParams,
    g_func: Callable[[float], float],
    delta_func: Callable[[float], float],
    rho_init: jnp.ndarray,
    t_final: float,
    dt0: float = 0.01,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[jnp.ndarray, float]:
    """
    Run full Lindblad simulation.

    Args:
        params: System parameters
        g_func: Coupling control g(t)
        delta_func: Detuning control Δ(t)
        rho_init: Initial density matrix
        t_final: Final time

    Returns:
        (final_rho, n_cav): Final state and cavity occupation
    """
    ops = build_operators(params)
    dynamics, state_size = build_master_equation_real(ops, params, g_func, delta_func)

    # Flatten initial state
    y0 = jnp.concatenate([jnp.real(rho_init).flatten(), jnp.imag(rho_init).flatten()])

    # Solve
    term = diffrax.ODETerm(dynamics)
    solver = diffrax.Kvaerno5()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=dt0,
        y0=y0,
        args=None,
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=50000,
    )

    # Reconstruct final state
    final_y = sol.ys[-1]
    dim = ops.dim
    rho_r = final_y[: dim * dim].reshape((dim, dim))
    rho_i = final_y[dim * dim :].reshape((dim, dim))
    rho_final = rho_r + 1j * rho_i

    # Compute cavity occupation
    n_cav = jnp.real(jnp.trace(ops.n_cav @ rho_final))

    return rho_final, float(n_cav)


if __name__ == "__main__":
    # Test simulation
    params = SystemParams(kappa=0.01, T_bath=1.0)
    ops = build_operators(params)

    # Initial state: thermal cavity + ground qubits
    from .initial_states import thermal_cavity_ground_qubits

    rho_init = thermal_cavity_ground_qubits(params)

    print(f"Initial state shape: {rho_init.shape}")
    n_init = jnp.real(jnp.trace(ops.n_cav @ rho_init))
    print(f"Initial <n>: {n_init:.4f}")

    # Constant coupling (baseline)
    g_const = lambda t: 0.5
    delta_const = lambda t: 0.0

    rho_final, n_final = simulate(params, g_const, delta_const, rho_init, t_final=10.0)
    print(f"Final <n> (g=0.5, Δ=0): {n_final:.4f}")
