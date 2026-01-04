"""
Gymnasium environment for RL-based Floquet cycle discovery.
Compatible with Stable-Baselines3 (SAC, PPO, etc.)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Dict, Any

from src.physics import (
    SystemParams,
    build_operators,
    thermal_occupation,
    thermal_cavity_ground_qubits,
)


class FloquetCoolingEnv(gym.Env):
    """
    RL environment for discovering optimal Floquet cooling cycles.

    The agent controls the coupling g(t) and detuning delta(t) at each step
    of a periodic cycle. The reward is based on minimizing cavity occupation.

    Observation space:
        - Current step in cycle (normalized)
        - Current cavity occupation estimate
        - Previous action (g, delta)

    Action space:
        - Continuous: [g, delta] bounded by max values
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_steps_per_cycle: int = 20,
        T_cycle: float = 0.5,
        n_cycles_per_episode: int = 100,
        g_max: float = 1.5,
        delta_max: float = 0.3,
        system_params: Optional[SystemParams] = None,
    ):
        super().__init__()

        self.n_steps = n_steps_per_cycle
        self.T_cycle = T_cycle
        self.n_cycles = n_cycles_per_episode
        self.g_max = g_max
        self.delta_max = delta_max

        # System parameters
        self.params = system_params or SystemParams(
            kappa=0.05, gamma1=0.01, T_bath=0.5, T_atom=0.05
        )

        # Pre-compute operators
        self.ops = build_operators(self.params)
        self._prepare_static_data()

        # Action space: [g, delta] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [step_fraction, n_estimate, prev_g, prev_delta]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # State
        self.current_step = 0
        self.current_cycle = 0
        self.rho = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.action_sequence = []

        # JIT-compile the step function
        self._compile_step_fn()

    def _prepare_static_data(self):
        """Pre-compute static operators for Lindblad evolution."""
        n_bar = thermal_occupation(self.params.omega_c, self.params.T_bath)
        kappa_down = self.params.kappa * (n_bar + 1)
        kappa_up = self.params.kappa * n_bar
        self.dt = self.T_cycle / self.n_steps

        self.static_data = {
            "V_jc": self.ops.V_jc,
            "sz_total": self.ops.sz1 + self.ops.sz2,
            "L_down": jnp.sqrt(kappa_down) * self.ops.a,
            "L_up": jnp.sqrt(kappa_up) * self.ops.a_dag,
            "L_q1": jnp.sqrt(self.params.gamma1) * self.ops.sm1,
            "L_q2": jnp.sqrt(self.params.gamma1) * self.ops.sm2,
            "n_cav": self.ops.n_cav,
            "dt": self.dt,
        }

    def _compile_step_fn(self):
        """JIT-compile the Lindblad step function."""
        sd = self.static_data

        @jax.jit
        def rk4_step(rho, g, delta):
            V_jc = sd["V_jc"]
            sz_total = sd["sz_total"]
            L_down, L_up = sd["L_down"], sd["L_up"]
            L_q1, L_q2 = sd["L_q1"], sd["L_q2"]
            dt = sd["dt"]

            def lindblad_rhs(r):
                H = 0.5 * delta * sz_total + g * V_jc
                drho = -1j * (H @ r - r @ H)
                for L in [L_down, L_up, L_q1, L_q2]:
                    Ld = L.conj().T
                    drho = drho + L @ r @ Ld - 0.5 * (Ld @ L @ r + r @ Ld @ L)
                return drho

            k1 = lindblad_rhs(rho)
            k2 = lindblad_rhs(rho + 0.5 * dt * k1)
            k3 = lindblad_rhs(rho + 0.5 * dt * k2)
            k4 = lindblad_rhs(rho + dt * k3)

            rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho_new = 0.5 * (rho_new + rho_new.conj().T)
            rho_new = rho_new / jnp.trace(rho_new)

            return rho_new

        self._rk4_step = rk4_step

    def _get_cavity_occupation(self) -> float:
        """Compute current cavity occupation."""
        return float(jnp.real(jnp.trace(self.static_data["n_cav"] @ self.rho)))

    def _get_obs(self) -> np.ndarray:
        """Construct observation."""
        step_frac = self.current_step / self.n_steps
        n_cav = self._get_cavity_occupation()
        return np.array(
            [
                step_frac,
                n_cav,
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_cycle = 0
        self.rho = thermal_cavity_ground_qubits(self.params)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.action_sequence = []

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: [g_normalized, delta_normalized] in [-1, 1]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Scale action to physical range
        g = float(action[0]) * self.g_max
        delta = float(action[1]) * self.delta_max

        # Store for observation
        self.prev_action = action.astype(np.float32)
        self.action_sequence.append((g, delta))

        # Apply one Lindblad step
        self.rho = self._rk4_step(self.rho, g, delta)

        # Update counters
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.current_step = 0
            self.current_cycle += 1

        # Compute reward (negative occupation = good)
        n_cav = self._get_cavity_occupation()
        reward = -n_cav  # Higher reward for lower occupation

        # Episode ends after n_cycles complete cycles
        terminated = self.current_cycle >= self.n_cycles
        truncated = False

        info = {
            "n_cav": n_cav,
            "cycle": self.current_cycle,
            "step": self.current_step,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_cycle_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the last complete cycle's parameters."""
        if len(self.action_sequence) < self.n_steps:
            return None, None

        last_cycle = self.action_sequence[-self.n_steps :]
        g_seq = np.array([a[0] for a in last_cycle])
        delta_seq = np.array([a[1] for a in last_cycle])

        return g_seq, delta_seq


def test_environment():
    """Test the RL environment with random actions."""
    env = FloquetCoolingEnv(n_steps_per_cycle=10, n_cycles_per_episode=20)

    print("Testing FloquetCoolingEnv")
    print("=" * 50)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    obs, _ = env.reset()
    print(f"\nInitial obs: {obs}")

    total_reward = 0
    n_total_steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_total_steps += 1

        if info["step"] == 0 and info["cycle"] > 0:
            print(f"Cycle {info['cycle']}: n_cav = {info['n_cav']:.4f}")

        if terminated or truncated:
            break

    print(f"\nTotal steps: {n_total_steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final n_cav: {info['n_cav']:.4f}")


if __name__ == "__main__":
    test_environment()
