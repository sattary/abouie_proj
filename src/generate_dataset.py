import jax
import jax.numpy as jnp
import diffrax
from diffrax import PIDController
import numpy as np
from tqdm import tqdm  # Progress bar

# --- RE-USE PHYSICS (Copying core logic for standalone safety) ---
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
        # Args is now a TUPLE: (coeffs, freqs) representing the pulse
        coeffs, freqs = args

        # Reconstruct pulse epsilon(t) from Fourier series
        # epsilon(t) = sum( A_n * sin(w_n * t) )
        epsilon = jnp.sum(coeffs * jnp.sin(freqs * t))

        u, v = y[:5], y[5:]
        H_real, H_imag = DQDSimulator.get_hamiltonian_terms(epsilon)

        # Matrix Math
        H_r_u = jnp.dot(H_real, u)
        H_r_v = jnp.dot(H_real, v)
        H_i_u = jnp.dot(H_imag, u)
        H_i_v = jnp.dot(H_imag, v)

        du = (H_i_u + H_r_v) / HBAR
        dv = (H_i_v - H_r_u) / HBAR
        return jnp.concatenate([du, dv])


# --- BATCH SIMULATOR ---
def simulate_single_pulse(coeffs, freqs):
    # Initial State: |S(1,1)>
    y0 = jnp.zeros(10).at[2].set(1.0)

    term = diffrax.ODETerm(DQDSimulator.dynamics_equation)
    solver = diffrax.Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    # Pack args
    args = (coeffs, freqs)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=20.0,
        dt0=0.01,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
        max_steps=40000,
    )

    final_y = sol.ys[-1]
    # Return Probability of Ground State
    return final_y[1] ** 2 + final_y[6] ** 2


# THE MAGIC: Vectorize the simulator across the batch dimension
# in_axes=(0, 0) means "split the first argument (coeffs) and second (freqs) across rows"
batch_simulate = jax.vmap(simulate_single_pulse, in_axes=(0, 0))


# --- GENERATOR ---
def generate_and_save_data(num_samples=5000, batch_size=1000):
    print(f"Generating {num_samples} samples on GPU...")

    # Fourier Parameter Settings
    num_components = 5  # Complexity of the pulse

    all_pulses = []
    all_scores = []

    # Discretized time for saving the pulse shapes (for the Diffusion Model)
    time_axis = jnp.linspace(0, 20, 200)

    num_batches = num_samples // batch_size

    for _ in tqdm(range(num_batches)):
        # 1. Generate Random Fourier Coefficients (JAX Keys are functional)
        key = jax.random.PRNGKey(np.random.randint(0, 100000))
        k1, k2 = jax.random.split(key)

        # Random Amplitudes: Uniform between -100 and 100 ueV
        coeffs_batch = jax.random.uniform(
            k1, (batch_size, num_components), minval=-100, maxval=100
        )

        # Random Frequencies: 0.1 to 2.0 GHz
        freqs_batch = jax.random.uniform(
            k2, (batch_size, num_components), minval=0.1, maxval=2.0
        )

        # 2. Run Simulation on GPU
        scores_batch = batch_simulate(coeffs_batch, freqs_batch)

        # 3. Reconstruct the actual pulse shapes (time-series) for saving
        # Shape: (Batch, TimeSteps)
        # We need this because the Diffusion Model learns "Images of curves", not coefficients
        def get_shape(c, f):
            return jnp.sum(
                c[:, None] * jnp.sin(f[:, None] * time_axis[None, :]), axis=0
            )

        # Vectorize the shape reconstruction
        batch_shapes = jax.vmap(get_shape)(coeffs_batch, freqs_batch)

        # Move to CPU for saving
        all_pulses.append(np.array(batch_shapes))
        all_scores.append(np.array(scores_batch))

    # Concatenate
    X = np.concatenate(all_pulses, axis=0)  # Inputs: Pulse Shapes
    Y = np.concatenate(all_scores, axis=0)  # Labels: Cooling Prob

    # Save
    np.save("dataset_pulses.npy", X)
    np.save("dataset_scores.npy", Y)
    print(f"\nSaved 'dataset_pulses.npy' shape: {X.shape}")
    print(f"Saved 'dataset_scores.npy' shape: {Y.shape}")
    print(f"Best cooling found in batch: {np.max(Y):.4f}")


if __name__ == "__main__":
    generate_and_save_data()
