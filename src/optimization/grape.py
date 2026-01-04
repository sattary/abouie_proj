"""
GRAPE (Gradient Ascent Pulse Engineering) optimizer for Floquet cooling cycles.
Optimizes both coupling g(t) and detuning Δ(t) to minimize cavity occupation.
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax
from typing import Tuple, NamedTuple
from functools import partial

from src.physics import (
    SystemParams,
    Operators,
    build_operators,
    thermal_occupation,
    thermal_cavity_ground_qubits,
)
from src.floquet import (
    FloquetCycleParams,
    prepare_cycle_data,
    find_floquet_steady_state,
    compute_cycle_commutator,
)


class GRAPEConfig(NamedTuple):
    """Configuration for GRAPE optimization."""

    n_steps: int = 20  # Steps per cycle
    T_cycle: float = 0.5  # Cycle period (ns)
    n_cycles_eval: int = 100  # Cycles to run for steady-state
    learning_rate: float = 0.01
    n_iterations: int = 200
    g_max: float = 2.0  # Max coupling (GHz)
    delta_max: float = 0.5  # Max detuning (GHz)


def create_optimizable_cycle(
    n_steps: int,
    T_cycle: float,
    g_init: float = 0.5,
    delta_init: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create initial pulse parameters for optimization."""
    g_params = jnp.ones(n_steps) * g_init
    delta_params = jnp.ones(n_steps) * delta_init
    return g_params, delta_params


def params_to_cycle(
    g_params: jnp.ndarray,
    delta_params: jnp.ndarray,
    T_cycle: float,
    g_max: float,
    delta_max: float,
) -> FloquetCycleParams:
    """Convert raw parameters to bounded FloquetCycleParams."""
    # Apply tanh to bound parameters
    g_bounded = g_max * jnp.tanh(g_params)
    delta_bounded = delta_max * jnp.tanh(delta_params)

    return FloquetCycleParams(
        T_cycle=T_cycle,
        n_steps=len(g_params),
        g_sequence=g_bounded,
        delta_sequence=delta_bounded,
    )


# JIT-compiled steady-state evaluation for gradient computation
@partial(jax.jit, static_argnums=(4, 5, 6))
def compute_steady_state_n(
    g_params: jnp.ndarray,
    delta_params: jnp.ndarray,
    rho_init: jnp.ndarray,
    static_data: Tuple,  # (V_jc, sz_total, L_down, L_up, L_q1, L_q2, n_cav, dt)
    n_cycles: int,
    g_max: float,
    delta_max: float,
) -> float:
    """Compute steady-state cavity occupation (differentiable)."""
    V_jc, sz_total, L_down, L_up, L_q1, L_q2, n_cav, dt = static_data

    # Bound parameters
    g_seq = g_max * jnp.tanh(g_params)
    delta_seq = delta_max * jnp.tanh(delta_params)

    def lindblad_rhs(r, g, delta):
        H = 0.5 * delta * sz_total + g * V_jc
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

    def one_cycle(rho, _):
        controls = jnp.stack([g_seq, delta_seq], axis=1)
        rho_new, _ = lax.scan(rk4_step, rho, controls)
        return rho_new, None

    # Run n_cycles
    rho_final, _ = lax.scan(one_cycle, rho_init, None, length=n_cycles)

    # Return real part of <n>
    return jnp.real(jnp.trace(n_cav @ rho_final))


def run_grape_optimization(
    params: SystemParams,
    config: GRAPEConfig,
    verbose: bool = True,
) -> Tuple[FloquetCycleParams, list]:
    """
    Run GRAPE optimization to find optimal cooling cycle.

    Returns:
        (optimal_cycle, loss_history)
    """
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    # Prepare static data for JIT
    n_bar = thermal_occupation(params.omega_c, params.T_bath)
    kappa_down = params.kappa * (n_bar + 1)
    kappa_up = params.kappa * n_bar
    dt = config.T_cycle / config.n_steps

    static_data = (
        ops.V_jc,
        ops.sz1 + ops.sz2,
        jnp.sqrt(kappa_down) * ops.a,
        jnp.sqrt(kappa_up) * ops.a_dag,
        jnp.sqrt(params.gamma1) * ops.sm1,
        jnp.sqrt(params.gamma1) * ops.sm2,
        ops.n_cav,
        dt,
    )

    # Initialize parameters
    g_params, delta_params = create_optimizable_cycle(
        config.n_steps, config.T_cycle, g_init=0.3, delta_init=0.0
    )

    # Loss function (minimize cavity occupation)
    def loss_fn(params_tuple):
        g_p, delta_p = params_tuple
        return compute_steady_state_n(
            g_p,
            delta_p,
            rho_init,
            static_data,
            config.n_cycles_eval,
            config.g_max,
            config.delta_max,
        )

    # Optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init((g_params, delta_params))

    # JIT-compile update step
    @jax.jit
    def update_step(params_tuple, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params_tuple)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_tuple, updates)
        return new_params, opt_state, loss

    history = []
    params_tuple = (g_params, delta_params)

    if verbose:
        print("GRAPE Optimization")
        print("=" * 50)

    for i in range(config.n_iterations):
        params_tuple, opt_state, loss = update_step(params_tuple, opt_state)
        history.append(float(loss))

        if verbose and (i + 1) % 20 == 0:
            print(f"Iter {i + 1}: <n> = {loss:.4f}")

    # Create final cycle
    g_final, delta_final = params_tuple
    final_cycle = params_to_cycle(
        g_final, delta_final, config.T_cycle, config.g_max, config.delta_max
    )

    return final_cycle, history


def evaluate_cycle(
    params: SystemParams,
    cycle: FloquetCycleParams,
    n_cycles: int = 300,
) -> Tuple[float, float]:
    """Evaluate a cycle's performance."""
    ops = build_operators(params)
    rho_init = thermal_cavity_ground_qubits(params)

    rho_ss, n_ss, iters = find_floquet_steady_state(
        ops, params, cycle, rho_init, n_cycles_max=n_cycles, verbose=False
    )

    comm_norm = compute_cycle_commutator(ops, cycle)

    return n_ss, comm_norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from src.baseline import compute_stochastic_limit, StochasticParams

    print("=" * 60)
    print("GRAPE OPTIMIZATION FOR CAVITY COOLING")
    print("=" * 60)

    # System parameters
    params = SystemParams(
        kappa=0.05,
        gamma1=0.01,
        T_bath=0.5,
        T_atom=0.05,
    )

    # GRAPE config
    config = GRAPEConfig(
        n_steps=20,
        T_cycle=0.5,
        n_cycles_eval=50,  # Fewer cycles for faster gradient
        learning_rate=0.02,
        n_iterations=100,
        g_max=1.5,
        delta_max=0.3,
    )

    print(f"\nConfig: {config.n_steps} steps, T={config.T_cycle}ns")
    print(f"Optimizing for {config.n_iterations} iterations...")

    # Run optimization
    optimal_cycle, history = run_grape_optimization(params, config)

    # Evaluate final cycle
    n_final, comm_norm = evaluate_cycle(params, optimal_cycle)

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

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Initial <n>: {history[0]:.4f}")
    print(f"Final <n>:   {n_final:.4f}")
    print(f"Stochastic limit: {n_stoch:.4f}")
    print(f"Commutator norm: {comm_norm:.4f}")

    if n_final < n_stoch:
        print("\nSUCCESS: Beat the stochastic limit!")
        improvement = (n_stoch - n_final) / n_stoch * 100
        print(f"Improvement: {improvement:.1f}%")
    else:
        print("\nNot yet beating stochastic limit")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss history
    axes[0].plot(history, "b-", linewidth=1.5)
    axes[0].axhline(n_stoch, color="r", linestyle="--", label="Stochastic limit")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("<n>")
    axes[0].set_title("GRAPE Optimization")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Optimal g(t)
    t = np.linspace(0, config.T_cycle, config.n_steps)
    axes[1].step(t, optimal_cycle.g_sequence, "g-", linewidth=2, where="post")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("g(t) [GHz]")
    axes[1].set_title("Optimal Coupling Pulse")
    axes[1].grid(True, alpha=0.3)

    # Optimal delta(t)
    axes[2].step(t, optimal_cycle.delta_sequence, "purple", linewidth=2, where="post")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Δ(t) [GHz]")
    axes[2].set_title("Optimal Detuning Pulse")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grape_optimization.png", dpi=150)
    print("\nSaved 'grape_optimization.png'")
