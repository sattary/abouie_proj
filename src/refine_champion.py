import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes, msgpack_restore
from generate_solution_v2 import PulseDiffuser, BETAS, ALPHAS, ALPHAS_CUMPROD
from verify_solution import verify_pulse
import flax


def refine_pulse(model, params, init_pulse, noise_level=50):
    """
    SDE-Edit: Starts with a real pulse, adds noise, and denoises it.
    noise_level: How many diffusion steps to go back (out of 200).
                 Small (20) = Polish. Large (100) = Redesign.
    """
    key = jax.random.PRNGKey(42)

    # 1. Forward Diffuse (Add Noise)
    t_start = noise_level
    noise = jax.random.normal(key, init_pulse.shape)

    # Get alpha_bar at t_start
    alpha_hat = ALPHAS_CUMPROD[t_start]

    # Noisy input: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
    x = jnp.sqrt(alpha_hat) * init_pulse + jnp.sqrt(1 - alpha_hat) * noise

    # 2. Reverse Diffuse (Denoise)
    # Condition: We ask for a slight improvement (1.2x average of top tier)
    cond_tensor = jnp.full((1, 1), 1.2)

    for i in reversed(range(t_start)):
        t_tensor = jnp.full((1,), i, dtype=jnp.int32)
        predicted_noise = model.apply(params, x, t_tensor, cond_tensor)

        alpha = ALPHAS[i]
        alpha_hat_prev = ALPHAS_CUMPROD[i]
        beta = BETAS[i]

        noise_z = (
            jax.random.normal(jax.random.fold_in(key, i), init_pulse.shape)
            if i > 0
            else 0.0
        )

        term1 = 1 / jnp.sqrt(alpha)
        term2 = (1 - alpha) / (jnp.sqrt(1 - alpha_hat_prev))

        x = term1 * (x - term2 * predicted_noise) + jnp.sqrt(beta) * noise_z

    return x


def main():
    print("Loading Data & Model...")
    # Load Stats
    try:
        mean_p, std_p = np.load("norm_stats.npy")
        pulses = np.load("dataset_pulses.npy")
        scores = np.load("dataset_scores.npy")
    except Exception:
        print("Error: Dataset files missing. Cannot refine.")
        exit()

    # Find the Champion (The 95% pulse)
    best_idx = np.argmax(scores)
    champion_pulse_raw = pulses[best_idx]
    champion_score = scores[best_idx]
    print(f"Loaded Champion Pulse. Score: {champion_score * 100:.2f}%")

    # Normalize Champion
    champion_norm = (champion_pulse_raw - mean_p) / std_p
    champion_jax = jnp.array([champion_norm])  # Add batch dim

    # Init Model
    model = PulseDiffuser()
    dummy_x = jnp.ones((1, 200))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_c = jnp.ones((1, 1))
    variables = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_c)
    target_params = variables["params"]

    # Load Weights
    with open("model_params.msgpack", "rb") as f:
        file_bytes = f.read()
    try:
        params = from_bytes(target_params, file_bytes)
    except Exception:
        raw_state = msgpack_restore(file_bytes)
        params = from_bytes(
            target_params, flax.serialization.to_bytes(raw_state["params"])
        )

    print("\n--- REFINEMENT SWEEP ---")
    # We try different levels of 'polishing'
    noise_levels = [10, 30, 50, 70]

    best_refined_score = 0.0
    best_refined_pulse = None

    for nl in noise_levels:
        print(f"Refining with noise level {nl}/200...")
        refined_norm = refine_pulse(
            model, {"params": params}, champion_jax, noise_level=nl
        )

        # Un-normalize
        refined_pulse = np.array(refined_norm[0]) * std_p + mean_p

        # Verify
        score = verify_pulse(jnp.array(refined_pulse))
        print(f"-> Result: {score * 100:.4f}%")

        if score > best_refined_score:
            best_refined_score = score
            best_refined_pulse = refined_pulse

    print("\n" + "=" * 30)
    print(f"Original Champion: {champion_score * 100:.4f}%")
    print(f"Refined Champion:  {best_refined_score * 100:.4f}%")

    if best_refined_score > champion_score:
        print("SUCCESS! AI improved the human baseline.")
        np.save("refined_pulse.npy", best_refined_pulse)

        plt.figure(figsize=(10, 6))
        t = np.linspace(0, 20, 200)
        plt.plot(t, champion_pulse_raw, "--", color="gray", label="Original", alpha=0.5)
        plt.plot(t, best_refined_pulse, color="purple", linewidth=2, label="AI Refined")
        plt.title(
            f"AI Refinement: {champion_score * 100:.1f}% -> {best_refined_score * 100:.1f}%"
        )
        plt.legend()
        plt.savefig("refinement_result.png")
    else:
        print("The original was already optimal (or AI drifted).")
    print("=" * 30)


if __name__ == "__main__":
    main()
