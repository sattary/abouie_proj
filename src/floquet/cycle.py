"""
Floquet cycle framework for coherent cooling protocols.
Implements periodic control sequences that can beat the stochastic limit.

JIT-optimized using jax.lax.scan for performance.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple
from functools import partial
from src.physics import (
    SystemParams,
    Operators,
    build_operators,
    thermal_occupation,
)


class FloquetCycleParams(NamedTuple):
    """Parameters defining a Floquet cooling cycle."""

    T_cycle: float  # Total cycle period (ns)
    n_steps: int  # Number of discrete steps per cycle
    g_sequence: jnp.ndarray  # Coupling at each step (n_steps,)
    delta_sequence: jnp.ndarray  # Detuning at each step (n_steps,)

    @property
    def dt(self) -> float:
        """Time step duration."""
        return self.T_cycle / self.n_steps


def create_constant_cycle(
    T_cycle: float,
    n_steps: int,
    g_const: float,
    delta_const: float = 0.0,
) -> FloquetCycleParams:
    """Create a cycle with constant controls (baseline)."""
    return FloquetCycleParams(
        T_cycle=T_cycle,
        n_steps=n_steps,
        g_sequence=jnp.ones(n_steps) * g_const,
        delta_sequence=jnp.ones(n_steps) * delta_const,
    )


def create_bang_bang_cycle(
    T_cycle: float,
    n_steps: int,
    g_on: float,
    g_off: float = 0.0,
    delta_on: float = 0.0,
    delta_off: float = 0.0,
    duty_cycle: float = 0.5,
) -> FloquetCycleParams:
    """Create a bang-bang (on-off) cycle."""
    n_on = int(n_steps * duty_cycle)
    g_seq = jnp.concatenate(
        [
            jnp.ones(n_on) * g_on,
            jnp.ones(n_steps - n_on) * g_off,
        ]
    )
    delta_seq = jnp.concatenate(
        [
            jnp.ones(n_on) * delta_on,
            jnp.ones(n_steps - n_on) * delta_off,
        ]
    )
    return FloquetCycleParams(
        T_cycle=T_cycle,
        n_steps=n_steps,
        g_sequence=g_seq,
        delta_sequence=delta_seq,
    )


def create_ramp_cycle(
    T_cycle: float,
    n_steps: int,
    g_max: float,
    delta_max: float,
) -> FloquetCycleParams:
    """Create a linear ramp cycle (up then down)."""
    half = n_steps // 2
    g_seq = jnp.concatenate(
        [
            jnp.linspace(0, g_max, half),
            jnp.linspace(g_max, 0, n_steps - half),
        ]
    )
    delta_seq = jnp.concatenate(
        [
            jnp.linspace(0, delta_max, half),
            jnp.linspace(delta_max, 0, n_steps - half),
        ]
    )
    return FloquetCycleParams(
        T_cycle=T_cycle,
        n_steps=n_steps,
        g_sequence=g_seq,
        delta_sequence=delta_seq,
    )


class CycleStaticData(NamedTuple):
    """Pre-computed static data for JIT-compiled cycle."""

    V_jc: jnp.ndarray  # Interaction operator
    sz_total: jnp.ndarray  # Total sigma_z
    L_down: jnp.ndarray  # Cavity decay jump op
    L_up: jnp.ndarray  # Cavity heating jump op
    L_q1: jnp.ndarray  # Qubit 1 decay
    L_q2: jnp.ndarray  # Qubit 2 decay
    n_cav: jnp.ndarray  # Number operator
    dt: float  # Time step


def prepare_cycle_data(
    ops: Operators,
    params: SystemParams,
    dt: float,
) -> CycleStaticData:
    """Pre-compute all static operators for a cycle."""
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    kappa_down = params.kappa * (n_bar + 1)
    kappa_up = params.kappa * n_bar

    return CycleStaticData(
        V_jc=ops.V_jc,
        sz_total=ops.sz1 + ops.sz2,
        L_down=jnp.sqrt(kappa_down) * ops.a,
        L_up=jnp.sqrt(kappa_up) * ops.a_dag,
        L_q1=jnp.sqrt(params.gamma1) * ops.sm1,
        L_q2=jnp.sqrt(params.gamma1) * ops.sm2,
        n_cav=ops.n_cav,
        dt=dt,
    )


@partial(jax.jit, static_argnums=(2,))
def apply_single_step(
    rho: jnp.ndarray,
    controls: Tuple[float, float],  # (g, delta)
    static: CycleStaticData,
) -> jnp.ndarray:
    """Apply one Lindblad step (JIT-compiled)."""
    g, delta = controls
    dt = static.dt

    # Hamiltonian
    H = 0.5 * delta * static.sz_total + g * static.V_jc

    # Commutator: -i[H, rho]
    drho = -1j * (H @ rho - rho @ H)

    # Lindblad dissipation for each channel
    def lindblad_term(L, r):
        L_dag = L.conj().T
        return L @ r @ L_dag - 0.5 * (L_dag @ L @ r + r @ L_dag @ L)

    drho = drho + lindblad_term(static.L_down, rho)
    drho = drho + lindblad_term(static.L_up, rho)
    drho = drho + lindblad_term(static.L_q1, rho)
    drho = drho + lindblad_term(static.L_q2, rho)

    # Euler step with trace normalization
    rho_new = rho + dt * drho
    rho_new = rho_new / jnp.trace(rho_new)

    return rho_new


def apply_floquet_cycle_jit(
    rho_init: jnp.ndarray,
    g_seq: jnp.ndarray,
    delta_seq: jnp.ndarray,
    static: CycleStaticData,
) -> jnp.ndarray:
    """
    Apply one complete Floquet cycle using lax.scan.
    Fully JIT-compilable.
    """

    def scan_body(rho, controls):
        g, delta = controls
        rho_new = apply_single_step(rho, (g, delta), static)
        return rho_new, None

    # Stack controls for scan
    controls = jnp.stack([g_seq, delta_seq], axis=1)

    rho_final, _ = lax.scan(scan_body, rho_init, controls)
    return rho_final


# Create a JIT-compiled version of full cycle with RK4
@jax.jit
def _apply_cycle_core(
    rho, g_seq, delta_seq, V_jc, sz_total, L_down, L_up, L_q1, L_q2, dt
):
    """Core cycle function with RK4 integrator for stability."""

    def lindblad_rhs(r, g, delta):
        """Compute drho/dt for given state and controls."""
        H = 0.5 * delta * sz_total + g * V_jc
        drho = -1j * (H @ r - r @ H)

        for L in [L_down, L_up, L_q1, L_q2]:
            Ld = L.conj().T
            drho = drho + L @ r @ Ld - 0.5 * (Ld @ L @ r + r @ Ld @ L)

        return drho

    def rk4_step(rho, controls):
        """4th-order Runge-Kutta step."""
        g, delta = controls[0], controls[1]

        k1 = lindblad_rhs(rho, g, delta)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, g, delta)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, g, delta)
        k4 = lindblad_rhs(rho + dt * k3, g, delta)

        rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce Hermiticity and trace = 1
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new = rho_new / jnp.trace(rho_new)

        return rho_new, None

    controls = jnp.stack([g_seq, delta_seq], axis=1)
    rho_final, _ = lax.scan(rk4_step, rho, controls)
    return rho_final


def apply_floquet_cycle(
    ops: Operators,
    params: SystemParams,
    cycle: FloquetCycleParams,
    rho_init: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one complete Floquet cycle (wrapper for JIT version)."""
    static = prepare_cycle_data(ops, params, cycle.dt)

    return _apply_cycle_core(
        rho_init,
        cycle.g_sequence,
        cycle.delta_sequence,
        static.V_jc,
        static.sz_total,
        static.L_down,
        static.L_up,
        static.L_q1,
        static.L_q2,
        static.dt,
    )


def find_floquet_steady_state(
    ops: Operators,
    params: SystemParams,
    cycle: FloquetCycleParams,
    rho_init: jnp.ndarray,
    n_cycles_max: int = 500,
    tol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, float, int]:
    """
    Find the steady state under repeated Floquet cycles.
    Uses JIT-compiled cycle for performance.
    """
    static = prepare_cycle_data(ops, params, cycle.dt)

    # JIT-compile the cycle application
    @jax.jit
    def one_cycle(rho):
        return _apply_cycle_core(
            rho,
            cycle.g_sequence,
            cycle.delta_sequence,
            static.V_jc,
            static.sz_total,
            static.L_down,
            static.L_up,
            static.L_q1,
            static.L_q2,
            static.dt,
        )

    # Warm up JIT
    rho = one_cycle(rho_init)
    n_prev = float(jnp.real(jnp.trace(ops.n_cav @ rho)))

    for i in range(1, n_cycles_max):
        rho = one_cycle(rho)

        if jnp.any(jnp.isnan(rho)):
            if verbose:
                print(f"Warning: NaN at cycle {i}")
            return rho_init, float("nan"), i

        n_curr = float(jnp.real(jnp.trace(ops.n_cav @ rho)))

        if abs(n_curr - n_prev) < tol:
            return rho, n_curr, i + 1

        n_prev = n_curr

        if verbose and (i + 1) % 50 == 0:
            print(f"  Cycle {i + 1}: <n> = {n_curr:.4f}")

    return rho, n_curr, n_cycles_max


def compute_cycle_commutator(
    ops: Operators,
    cycle: FloquetCycleParams,
) -> float:
    """
    Compute the maximum Hamiltonian commutator within the cycle.
    Non-zero commutator is necessary for beating the stochastic limit.
    """
    sz_total = ops.sz1 + ops.sz2

    # Build all Hamiltonians
    H_list = [
        0.5 * cycle.delta_sequence[i] * sz_total + cycle.g_sequence[i] * ops.V_jc
        for i in range(cycle.n_steps)
    ]

    max_comm = 0.0
    for i, H_i in enumerate(H_list):
        for H_j in H_list[i + 1 :]:
            comm = H_i @ H_j - H_j @ H_i
            norm = float(jnp.sqrt(jnp.sum(jnp.abs(comm) ** 2)))
            max_comm = max(max_comm, norm)

    return max_comm


if __name__ == "__main__":
    from src.physics import thermal_cavity_ground_qubits
    from src.baseline import compute_stochastic_limit, StochasticParams
    import time

    print("=" * 60)
    print("FLOQUET CYCLE TEST (JIT-Optimized)")
    print("=" * 60)

    # System setup (smaller for faster test)
    params = SystemParams(
        kappa=0.05,  # Faster decay
        gamma1=0.01,  # Faster qubit decay
        T_bath=0.5,  # Warmer bath
        T_atom=0.05,
    )
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    n_init = float(jnp.real(jnp.trace(ops.n_cav @ rho_init)))
    print(f"\nInitial <n>: {n_init:.4f}")

    # Test constant cycle
    print("\n--- Constant Cycle (g=0.5) ---")
    cycle_const = create_constant_cycle(T_cycle=0.5, n_steps=10, g_const=0.5)

    t0 = time.time()
    rho_ss, n_ss, iters = find_floquet_steady_state(
        ops, params, cycle_const, rho_init, n_cycles_max=200
    )
    t1 = time.time()

    print(f"Steady-state <n>: {n_ss:.4f} ({iters} cycles, {t1 - t0:.2f}s)")
    print(f"Commutator norm: {compute_cycle_commutator(ops, cycle_const):.6f}")

    # Test bang-bang cycle
    print("\n--- Bang-Bang Cycle ---")
    cycle_bb = create_bang_bang_cycle(
        T_cycle=0.5,
        n_steps=10,
        g_on=1.0,
        g_off=0.0,
        delta_on=0.2,
        delta_off=-0.2,
    )

    t0 = time.time()
    rho_ss, n_ss, iters = find_floquet_steady_state(
        ops, params, cycle_bb, rho_init, n_cycles_max=200
    )
    t1 = time.time()

    print(f"Steady-state <n>: {n_ss:.4f} ({iters} cycles, {t1 - t0:.2f}s)")
    print(f"Commutator norm: {compute_cycle_commutator(ops, cycle_bb):.6f}")

    # Stochastic baseline
    stoch = StochasticParams(
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
    n_stoch, _ = compute_stochastic_limit(stoch, delta=0.0, two_atom=True)
    print(f"\nTarget (stochastic limit): n* = {n_stoch:.4f}")
