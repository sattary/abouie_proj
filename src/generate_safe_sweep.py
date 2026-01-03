import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes, msgpack_restore
from generate_solution_v2 import PulseDiffuser, TIMESTEPS, BETAS, ALPHAS, ALPHAS_CUMPROD
from verify_solution import verify_pulse

import flax


def sample_batch_safe(model, params, condition_values, shape=(1, 200)):
    batch_size = len(condition_values)
    key = jax.random.PRNGKey(999)  # New seed

    # Start with noise
    x = jax.random.normal(key, (batch_size, 200))
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

        # --- THE FIX: CLAMPING ---
        # Force the latent space to stay within 3 standard deviations.
        # This prevents the exponential explosion to 60,000.
        x = jnp.clip(x, -3.0, 3.0)

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

    with open("model_params.msgpack", "rb") as f:
        file_bytes = f.read()

    try:
        params = from_bytes(target_params, file_bytes)
    except Exception:
        raw_state = msgpack_restore(file_bytes)
        params = from_bytes(
            target_params, flax.serialization.to_bytes(raw_state["params"])
        )

    # Sweeping safe conditions
    conditions = [0.0, 0.5, 0.8, 1.0, 1.2]
    print(f"\nGenerative Sweep: Testing conditions {conditions}...")

    generated_norm = sample_batch_safe(model, {"params": params}, conditions)

    # Un-normalize
    generated_pulses = np.array(generated_norm) * std_p + mean_p

    best_score = 0.0
    best_pulse = None
    best_cond = 0.0

    print("\n--- PHYSICS VERIFICATION ---")
    for i, cond in enumerate(conditions):
        pulse = generated_pulses[i]
        max_amp = np.max(np.abs(pulse))

        # Check if clamping worked
        if max_amp > 400.0:
            print(f"Cond {cond}: STILL UNPHYSICAL (Amp {max_amp:.0f})")
        else:
            score = verify_pulse(jnp.array(pulse))
            print(f"Cond {cond}: Amp {max_amp:.1f} ueV -> Cooling: {score * 100:.4f}%")

            if score > best_score:
                best_score = score
                best_pulse = pulse
                best_cond = cond

    print("\n" + "=" * 30)
    if best_pulse is not None:
        print(f"WINNER: Condition {best_cond} achieved {best_score * 100:.4f}%")

        # Compare to Baseline
        try:
            baseline = np.max(np.load("dataset_scores.npy"))
            print(f"Baseline to beat: {baseline * 100:.2f}%")
            if best_score > baseline:
                print("RESULT: NEW STATE OF THE ART FOUND!")
        except Exception:
            pass

        np.save("ai_pulse_final.npy", best_pulse)

        # Plot
        t = np.linspace(0, 20, 200)
        plt.figure(figsize=(10, 6))
        plt.plot(
            t,
            best_pulse,
            color="purple",
            linewidth=3,
            label=f"AI Pulse ({best_score * 100:.2f}%)",
        )
        plt.title("Final AI Optimized Protocol")
        plt.xlabel("Time (ns)")
        plt.ylabel("Detuning (ueV)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("final_result.png")
        print("Saved 'final_result.png'")
    else:
        print("Optimization failed. Try training for more epochs.")
    print("=" * 30)


if __name__ == "__main__":
    main()
