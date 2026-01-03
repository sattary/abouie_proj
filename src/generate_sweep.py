import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes, msgpack_restore
from generate_solution_v2 import PulseDiffuser, TIMESTEPS, BETAS, ALPHAS, ALPHAS_CUMPROD
from verify_solution import verify_pulse  # Import the simulator directly

import flax


# --- RE-DEFINE SAMPLING FOR BATCHING ---
def sample_batch(model, params, condition_values, shape=(1, 200)):
    # condition_values: list of floats
    batch_size = len(condition_values)
    key = jax.random.PRNGKey(42)

    # Create batch of noise
    x = jax.random.normal(key, (batch_size, 200))

    # Create batch of conditions
    cond_tensor = jnp.array(condition_values).reshape((batch_size, 1))

    for i in reversed(range(TIMESTEPS)):
        t = jnp.full((batch_size,), i, dtype=jnp.int32)
        predicted_noise = model.apply(params, x, t, cond_tensor)

        alpha = ALPHAS[i]
        alpha_hat = ALPHAS_CUMPROD[i]
        beta = BETAS[i]

        noise_z = (
            jax.random.normal(jax.random.fold_in(key, i), (batch_size, 200))
            if i > 0
            else 0.0
        )

        term1 = 1 / jnp.sqrt(alpha)
        term2 = (1 - alpha) / (jnp.sqrt(1 - alpha_hat))

        x = term1 * (x - term2 * predicted_noise) + jnp.sqrt(beta) * noise_z
    return x


def main():
    print("Loading Model...")
    try:
        mean_p, std_p = np.load("norm_stats.npy")
        print(f"Data Stats - Mean: {mean_p:.2f}, Std: {std_p:.2f}")
    except Exception:
        mean_p, std_p = 0.0, 1.0

    model = PulseDiffuser()
    dummy_x = jnp.ones((1, 200))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_c = jnp.ones((1, 1))
    variables = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_c)
    target_params = variables["params"]

    # Load Weights
    with open("model_params.msgpack", "rb") as f:
        file_bytes = f.read()

    # Robust Load
    try:
        params = from_bytes(target_params, file_bytes)
    except Exception:
        raw_state = msgpack_restore(file_bytes)
        params = from_bytes(
            target_params, flax.serialization.to_bytes(raw_state["params"])
        )

    # --- THE SWEEP ---
    # We try condition values from 1.0 (Good) to 2.0 (Amazing)
    # We avoid 3.0 because it caused the explosion.
    conditions = [1.0, 1.25, 1.5, 1.75, 2.0]
    print(f"\nGenerative Sweep: Testing conditions {conditions}...")

    generated_norm = sample_batch(model, {"params": params}, conditions)

    # Un-normalize
    generated_pulses = np.array(generated_norm) * std_p + mean_p

    best_score = 0.0
    best_pulse = None
    best_cond = 0.0

    print("\n--- PHYSICS VERIFICATION ---")
    for i, cond in enumerate(conditions):
        pulse = generated_pulses[i]

        # Safety Clamp: If pulse > 500 ueV, it's garbage.
        max_amp = np.max(np.abs(pulse))
        if max_amp > 500.0:
            print(f"Cond {cond}: REJECTED (Amplitude {max_amp:.0f} ueV is unphysical)")
            score = 0.0
        else:
            # Run Simulator
            score = verify_pulse(jnp.array(pulse))
            print(
                f"Cond {cond}: Amplitude {max_amp:.1f} ueV -> Cooling: {score * 100:.4f}%"
            )

        if score > best_score:
            best_score = score
            best_pulse = pulse
            best_cond = cond

    print("\n" + "=" * 30)
    if best_pulse is not None:
        print(f"WINNER: Condition {best_cond} achieved {best_score * 100:.4f}%")
        np.save("ai_pulse.npy", best_pulse)

        # Plot the winner
        t = np.linspace(0, 20, 200)
        plt.figure(figsize=(10, 6))
        plt.plot(
            t,
            best_pulse,
            color="purple",
            linewidth=3,
            label=f"AI Pulse ({best_score * 100:.2f}%)",
        )
        plt.title(f"The Optimized Cooling Protocol (Cond {best_cond})")
        plt.xlabel("Time (ns)")
        plt.ylabel("Detuning (ueV)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ai_solution_sweep.png")
        print("Saved 'ai_solution_sweep.png' and 'ai_pulse.npy'")
    else:
        print(
            "No valid pulse found. The model might need more training data or lower learning rate."
        )
    print("=" * 30)


if __name__ == "__main__":
    main()
