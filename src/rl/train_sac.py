"""
SAC (Soft Actor-Critic) training for Floquet cycle discovery.
Uses Stable-Baselines3 for RL implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.env import FloquetCoolingEnv
from src.physics import SystemParams
from src.baseline import compute_stochastic_limit, StochasticParams


class CoolingCallback(BaseCallback):
    """Callback to track cooling performance during training."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_n_cav = float("inf")
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get latest info from environment
            infos = self.locals.get("infos", [])
            if infos:
                n_cav = infos[-1].get("n_cav", float("inf"))
                self.history.append((self.n_calls, n_cav))

                if n_cav < self.best_n_cav:
                    self.best_n_cav = n_cav

                if self.verbose:
                    print(
                        f"Step {self.n_calls}: n_cav = {n_cav:.4f} (best: {self.best_n_cav:.4f})"
                    )

        return True


def train_sac(
    total_timesteps: int = 50000,
    n_steps_per_cycle: int = 20,
    n_cycles_per_episode: int = 50,
    learning_rate: float = 3e-4,
    verbose: int = 1,
) -> tuple:
    """
    Train SAC agent to discover optimal Floquet cycles.

    Returns:
        (model, callback): Trained model and callback with history
    """
    # System parameters
    params = SystemParams(kappa=0.05, gamma1=0.01, T_bath=0.5, T_atom=0.05)

    # Create environment
    env = FloquetCoolingEnv(
        n_steps_per_cycle=n_steps_per_cycle,
        T_cycle=0.5,
        n_cycles_per_episode=n_cycles_per_episode,
        g_max=1.5,
        delta_max=0.3,
        system_params=params,
    )

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
    )

    # Training callback
    callback = CoolingCallback(check_freq=2000, verbose=verbose)

    print("=" * 60)
    print("SAC TRAINING FOR FLOQUET CYCLE DISCOVERY")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Cycle: {n_steps_per_cycle} steps x {n_cycles_per_episode} cycles")
    print()

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
    )

    return model, callback, env


def evaluate_trained_agent(model, env, n_eval_episodes: int = 5):
    """Evaluate trained agent and extract best cycle."""
    results = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append(
            {
                "reward": total_reward,
                "n_cav": info["n_cav"],
            }
        )
        print(
            f"Episode {ep + 1}: n_cav = {info['n_cav']:.4f}, reward = {total_reward:.2f}"
        )

    # Get cycle parameters from last episode
    g_seq, delta_seq = env.get_cycle_params()

    return results, g_seq, delta_seq


if __name__ == "__main__":
    # Quick training run
    model, callback, env = train_sac(
        total_timesteps=20000,  # Short for demo
        n_steps_per_cycle=10,
        n_cycles_per_episode=30,
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results, g_seq, delta_seq = evaluate_trained_agent(model, env, n_eval_episodes=3)

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

    best_n = min(r["n_cav"] for r in results)
    print(f"\nBest n_cav achieved: {best_n:.4f}")
    print(f"Stochastic limit: {n_stoch:.4f}")

    if best_n < n_stoch:
        improvement = (n_stoch - best_n) / n_stoch * 100
        print(f"Improvement: {improvement:.1f}%")

    # Plot results
    if callback.history:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Training curve
        steps, n_cavs = zip(*callback.history)
        axes[0].plot(steps, n_cavs, "b-", linewidth=1.5)
        axes[0].axhline(n_stoch, color="r", linestyle="--", label="Stochastic limit")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("<n>")
        axes[0].set_title("SAC Training Progress")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Discovered g(t)
        if g_seq is not None:
            t = np.linspace(0, 0.5, len(g_seq))
            axes[1].step(t, g_seq, "g-", linewidth=2, where="post")
            axes[1].set_xlabel("Time (ns)")
            axes[1].set_ylabel("g(t) [GHz]")
            axes[1].set_title("Discovered Coupling Pulse")
            axes[1].grid(True, alpha=0.3)

            axes[2].step(t, delta_seq, "purple", linewidth=2, where="post")
            axes[2].set_xlabel("Time (ns)")
            axes[2].set_ylabel("Δ(t) [GHz]")
            axes[2].set_title("Discovered Detuning Pulse")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("images/sac_training.png", dpi=150)
        print("\nSaved 'images/sac_training.png'")
