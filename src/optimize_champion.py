import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from verify_solution import DQDSimulator  # Reuse physics

# --- CONFIGURATION ---
LEARNING_RATE = 0.5  # Step size for optimization
STEPS = 200  # How many refinement steps


def optimize_pulse(init_pulse):
    # Convert to JAX array
    pulse_params = jnp.array(init_pulse)

    # Define Loss Function (Negative Probability because we minimize)
    def loss_fn(params):
        # We need to recreate the physics call here to be traceable by JAX
        # (verify_pulse is traceable if we keep it pure JAX inside)

        # 1. Define Interpolated Pulse
        time_points = jnp.linspace(0, 20, 200)

        def pulse_shape(t):
            return jnp.interp(t, time_points, params)

        # 2. Solver
        import diffrax
        from diffrax import PIDController

        y0 = jnp.zeros(10).at[2].set(1.0)
        term = diffrax.ODETerm(DQDSimulator.dynamics_equation)
        solver = diffrax.Kvaerno5()
        stepsize_controller = PIDController(
            rtol=1e-5, atol=1e-5
        )  # Slightly looser for speed

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=20.0,
            dt0=0.01,
            y0=y0,
            args=pulse_shape,
            stepsize_controller=stepsize_controller,
            max_steps=40000,
        )

        final_y = sol.ys[-1]
        prob = final_y[1] ** 2 + final_y[6] ** 2
        return -prob  # Minimize negative prob

    # Optimizer Setup
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(pulse_params)

    # Value and Grad function
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    print(f"Starting Optimization (LR={LEARNING_RATE})...")

    history = []
    current_params = pulse_params

    for i in tqdm(range(STEPS)):
        loss, grads = value_and_grad_fn(current_params)

        # Safety Clip Gradients (prevent explosion)
        grads = jnp.clip(grads, -100.0, 100.0)

        updates, opt_state = optimizer.update(grads, opt_state)
        current_params = optax.apply_updates(current_params, updates)

        # Clamp amplitude to stay physical
        current_params = jnp.clip(current_params, -400.0, 400.0)

        score = -loss
        history.append(score)

        if i % 20 == 0:
            tqdm.write(f"Step {i}: Score {score * 100:.4f}%")

    return current_params, history


def main():
    print("Loading Baseline Data...")
    try:
        pulses = np.load("dataset_pulses.npy")
        scores = np.load("dataset_scores.npy")
    except Exception:
        print("Error: dataset files not found.")
        return

    # Find Champion
    best_idx = np.argmax(scores)
    champion_pulse = pulses[best_idx]
    start_score = scores[best_idx]

    print(f"Initial Champion Score: {start_score * 100:.4f}%")
    print("Applying JAX Differentiable Physics Optimization...")

    optimized_pulse, history = optimize_pulse(champion_pulse)

    final_score = history[-1]

    print("\n" + "=" * 30)
    print(f"Start: {start_score * 100:.4f}%")
    print(f"End:   {final_score * 100:.4f}%")
    improvement = (final_score - start_score) * 100
    print(f"Improvement: +{improvement:.2f}%")

    if final_score > 0.99:
        print("CRITICAL SUCCESS: NEAR-PERFECT COOLING ACHIEVED!")
    elif final_score > start_score:
        print("SUCCESS: Optimization verified.")
    else:
        print("Warning: No improvement found.")
    print("=" * 30)

    # Save Results
    np.save("optimized_pulse.npy", optimized_pulse)

    # Plot
    plt.figure(figsize=(10, 6))
    t = np.linspace(0, 20, 200)
    plt.plot(
        t,
        champion_pulse,
        "--",
        color="gray",
        label=f"Original ({start_score * 100:.1f}%)",
        alpha=0.6,
    )
    plt.plot(
        t,
        optimized_pulse,
        color="green",
        linewidth=2,
        label=f"Optimized ({final_score * 100:.1f}%)",
    )
    plt.title("Differentiable Physics Optimization")
    plt.xlabel("Time (ns)")
    plt.ylabel("Detuning (ueV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("optimization_result.png")
    print("Saved 'optimization_result.png' and 'optimized_pulse.npy'")


if __name__ == "__main__":
    main()
