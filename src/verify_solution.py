import jax.numpy as jnp
import diffrax
from diffrax import PIDController
import numpy as np
import matplotlib.pyplot as plt

# --- 1. RE-ESTABLISH PHYSICS ENGINE ---
HBAR = 0.6582
TC, EZ, ESO, UC = 2.0, 10.0, 0.5, 2000.0


class DQDSimulator:
    @staticmethod
    def get_hamiltonian_terms(epsilon):
        E = jnp.zeros(5)
        E = E.at[0].set(-EZ)
        E = E.at[1].set(-epsilon)
        E = E.at[2].set(0.0)
        E = E.at[3].set(epsilon + UC)
        E = E.at[4].set(EZ)
        H_real = jnp.diag(E)
        H_real = H_real.at[1, 2].set(TC)
        H_real = H_real.at[2, 1].set(TC)
        H_imag = jnp.zeros((5, 5))
        H_imag = H_imag.at[0, 2].set(ESO)
        H_imag = H_imag.at[2, 0].set(-ESO)
        H_imag = H_imag.at[4, 2].set(ESO)
        H_imag = H_imag.at[2, 4].set(-ESO)
        return H_real, H_imag

    @staticmethod
    def dynamics_equation(t, y, args):
        # Args is now the INTERPOLATED pulse function
        pulse_func = args
        epsilon = pulse_func(t)

        u, v = y[:5], y[5:]
        H_real, H_imag = DQDSimulator.get_hamiltonian_terms(epsilon)

        H_r_u = jnp.dot(H_real, u)
        H_r_v = jnp.dot(H_real, v)
        H_i_u = jnp.dot(H_imag, u)
        H_i_v = jnp.dot(H_imag, v)

        du = (H_i_u + H_r_v) / HBAR
        dv = (H_i_v - H_r_u) / HBAR
        return jnp.concatenate([du, dv])


def verify_pulse(pulse_array):
    # Create an interpolation function for the discrete AI pulse
    time_points = jnp.linspace(0, 20, 200)

    # Linear interpolation of the AI's output
    def pulse_shape(t):
        return jnp.interp(t, time_points, pulse_array)

    y0 = jnp.zeros(10).at[2].set(1.0)

    term = diffrax.ODETerm(DQDSimulator.dynamics_equation)
    solver = diffrax.Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)  # High precision

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=20.0,
        dt0=0.01,
        y0=y0,
        args=pulse_shape,
        stepsize_controller=stepsize_controller,
        max_steps=50000,
    )

    final_y = sol.ys[-1]
    return final_y[1] ** 2 + final_y[6] ** 2


# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- TRUTH TEST ---")

    # Load AI Pulse
    try:
        ai_pulse = np.load("ai_pulse.npy")
        print("Loaded 'ai_pulse.npy'.")
    except Exception:
        print(
            "Error: Could not find 'ai_pulse.npy'. Run generate_solution_v2.py first!"
        )
        exit()

    # Load Baseline (Random Search)
    try:
        random_scores = np.load("dataset_scores.npy")
        best_random = np.max(random_scores)
        print(f"Baseline (Best Random Pulse): {best_random * 100:.2f}%")
    except Exception:
        best_random = 0.0
        print("Warning: Could not load dataset scores for comparison.")

    # RUN SIMULATION
    print("Simulating AI Pulse physics...")
    ai_score = verify_pulse(jnp.array(ai_pulse))

    print("\n" + "=" * 30)
    print(f"AI PERFORMANCE: {ai_score * 100:.4f}%")
    print("=" * 30)

    if ai_score > best_random:
        print("RESULT: SUCCESS! The AI beat the Brute Force method.")
        print(f"Improvement: +{(ai_score - best_random) * 100:.2f}%")
    else:
        print("RESULT: The AI pulse is valid, but didn't beat the best random seed.")
        print("Try generating again with a higher 'target_score_norm'.")

    # Plot Comparison
    plt.figure(figsize=(10, 5))
    t = np.linspace(0, 20, 200)
    plt.plot(t, ai_pulse, color="purple", label=f"AI Generated ({ai_score * 100:.2f}%)")
    plt.axhline(0, color="black", alpha=0.3)
    plt.title("The AI-Designed Cooling Protocol")
    plt.xlabel("Time (ns)")
    plt.ylabel("Detuning (ueV)")
    plt.legend()
    plt.savefig("verification_result.png")
    print("Saved 'verification_result.png'")
