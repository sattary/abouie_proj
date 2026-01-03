import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes
from train_diffusion import PulseDiffuser, TIMESTEPS, BETAS, ALPHAS, ALPHAS_CUMPROD


# --- 1. REVERSE DIFFUSION (SAMPLING) ---
def sample(model, params, condition_value, shape=(1, 200)):
    # Start with pure random noise
    key = jax.random.PRNGKey(99)
    x = jax.random.normal(key, shape)

    # Condition: We want a score of "condition_value" (e.g., 2.0 std devs above mean)
    # Note: We trained on normalized scores. We need to feed a high normalized score.
    # Let's assume we want to be in the top 1% of the distribution.
    cond_tensor = jnp.full((shape[0], 1), condition_value)

    # Backward Loop: Denoise step-by-step
    for i in reversed(range(TIMESTEPS)):
        t = jnp.full((shape[0],), i, dtype=jnp.int32)

        # Predict Noise
        predicted_noise = model.apply(params, x, t, cond_tensor)

        # Math: x_{t-1} = (1/sqrt(alpha)) * (x_t - ... * predicted_noise) + sigma * z
        alpha = ALPHAS[i]
        alpha_hat = ALPHAS_CUMPROD[i]
        beta = BETAS[i]

        noise_z = jax.random.normal(jax.random.fold_in(key, i), shape) if i > 0 else 0.0

        term1 = 1 / jnp.sqrt(alpha)
        term2 = (1 - alpha) / (jnp.sqrt(1 - alpha_hat))

        x = term1 * (x - term2 * predicted_noise) + jnp.sqrt(beta) * noise_z

    return x


def main():
    print("Loading Model...")
    # Load Normalization Stats
    mean_p, std_p = np.load("norm_stats.npy")

    # Initialize Model Structure
    model = PulseDiffuser()
    dummy_x = jnp.ones((1, 200))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_c = jnp.ones((1, 1))
    variables = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_c)

    # Load Trained Weights
    with open("model_params.msgpack", "rb") as f:
        params = from_bytes(variables["params"], f.read())

    print("Generating the 'Dream Pulse'...")
    # Ask for a score that is 3 standard deviations above the average!
    # This forces the AI to extrapolate beyond the training data.
    target_score_norm = 3.0

    generated_pulse_norm = sample(model, {"params": params}, target_score_norm)

    # Un-normalize
    final_pulse = generated_pulse_norm * std_p + mean_p
    final_pulse = np.array(final_pulse[0])  # Take first batch item

    # Plot
    t = np.linspace(0, 20, 200)
    plt.figure(figsize=(10, 6))
    plt.plot(
        t,
        final_pulse,
        color="purple",
        linewidth=3,
        label="AI Generated (Target: High Cooling)",
    )
    plt.title("The AI-Designed Cooling Protocol")
    plt.xlabel("Time (ns)")
    plt.ylabel("Detuning (ueV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("ai_solution.png")

    # Save the raw numbers to test in the simulator
    np.save("ai_pulse.npy", final_pulse)
    print("\nSuccess! Saved plot to 'ai_solution.png'.")
    print("Run the simulator next to verify if this pulse actually works!")


if __name__ == "__main__":
    main()
