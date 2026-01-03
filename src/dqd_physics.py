import jax
import jax.numpy as jnp
import diffrax
from diffrax import PIDController

# --- PHYSICAL CONSTANTS (From Table 1 or Text) ---
# Units: ueV (energy), ns (time)
HBAR = 0.6582  # Planck constant
TC = 2.0  # Tunneling coupling (S11 <-> S02)
EZ = 10.0  # Zeeman Energy (Splitting of Triplets)
ESO = 0.5  # Spin-Orbit Coupling (mixes S and T)
UC = 2000.0  # Charging Energy (Hubbard U) - CAUSES STIFFNESS

# Basis Map: [T-, S02, S11, S20, T+]
# Indices:    0    1    2    3    4


class DQDSimulator:
    @staticmethod
    def get_hamiltonian_terms(epsilon):
        """
        Returns the Real and Imaginary parts of the Hamiltonian matrix H.
        H_real is symmetric, H_imag is anti-symmetric.
        """
        # --- 1. Diagonals (Real Only) ---
        # Energies relative to S(1,1)
        E = jnp.zeros(5)
        E = E.at[0].set(-EZ)  # T-
        E = E.at[1].set(-epsilon)  # S(0,2)
        E = E.at[2].set(0.0)  # S(1,1)
        E = E.at[3].set(epsilon + UC)  # S(2,0) (High Energy)
        E = E.at[4].set(EZ)  # T+

        H_real = jnp.diag(E)

        # --- 2. Tunneling (Real Coupling) ---
        # Connects S(0,2) [1] <-> S(1,1) [2]
        H_real = H_real.at[1, 2].set(TC)
        H_real = H_real.at[2, 1].set(TC)

        # --- 3. Spin-Orbit (Complex/Imaginary Coupling) ---
        # Connects S(1,1) [2] <-> T+ [4] and T- [0]
        # H_so = i * ESO
        H_imag = jnp.zeros((5, 5))

        # Coupling to T-
        H_imag = H_imag.at[0, 2].set(ESO)  # i * ESO
        H_imag = H_imag.at[2, 0].set(-ESO)  # -i * ESO (Hermitian conjugate)

        # Coupling to T+
        H_imag = H_imag.at[4, 2].set(ESO)
        H_imag = H_imag.at[2, 4].set(-ESO)

        return H_real, H_imag

    @staticmethod
    def dynamics_equation(t, y, args):
        """
        Schrodinger Eq: d(psi)/dt = -i/hbar * H * psi

        We split psi into Real (u) and Imag (v) parts:
        psi = u + i*v
        d(u)/dt = (H_imag * u + H_real * v) / hbar
        d(v)/dt = (H_imag * v - H_real * u) / hbar
        """
        pulse_func = args
        epsilon = pulse_func(t)

        # Unpack state vector (size 10) -> (5 real, 5 imag)
        u = y[:5]  # Real part
        v = y[5:]  # Imag part

        H_real, H_imag = DQDSimulator.get_hamiltonian_terms(epsilon)

        # Compute derivatives (Matrix-Vector multiplication)
        # 1. Term: H_real * u
        H_r_u = jnp.dot(H_real, u)
        # 2. Term: H_real * v
        H_r_v = jnp.dot(H_real, v)
        # 3. Term: H_imag * u
        H_i_u = jnp.dot(H_imag, u)
        # 4. Term: H_imag * v
        H_i_v = jnp.dot(H_imag, v)

        # Schrodinger split:
        # du/dt = (H_imag*u + H_real*v) / hbar
        du = (H_i_u + H_r_v) / HBAR

        # dv/dt = (H_imag*v - H_real*u) / hbar
        dv = (H_i_v - H_r_u) / HBAR

        return jnp.concatenate([du, dv])


def simulate_pulse(pulse_amplitude):
    """
    Runs one cooling cycle.
    Returns: Probability of being in Ground State |S(0,2)> (Index 1).
    """

    # Define Pulse: Simple Sine wave for now
    def pulse_shape(t):
        return pulse_amplitude * jnp.sin(t * 0.5)

    # Initial State: Start in |S(1,1)> (Index 2)
    # y = [Real... , Imag...]
    y0 = jnp.zeros(10)
    y0 = y0.at[2].set(1.0)  # Real part of Index 2 is 1.0

    # --- SOLVER SETUP FOR STIFF PHYSICS ---
    term = diffrax.ODETerm(DQDSimulator.dynamics_equation)

    # Use Kvaerno5 (Implicit solver) for stiff equations
    solver = diffrax.Kvaerno5()

    # Adaptive step size controller (PID)
    # This prevents the solver from blowing up when frequencies are high
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)

    # Run for 20ns
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

    # Calculate Probability of Target State |S(0,2)> (Index 1)
    # Prob = Real[1]^2 + Imag[1]^2
    prob_ground = final_y[1] ** 2 + final_y[6] ** 2

    return prob_ground


if __name__ == "__main__":
    print(f"JAX Devices: {jax.devices()}")

    # 1. Run Forward Simulation
    print("Running simulation...")
    ground_state_prob = simulate_pulse(50.0)
    print(f"Final Ground State Probability: {ground_state_prob:.6f}")

    # 2. Run Backward (Gradient)
    print("Computing gradient...")
    grad_fn = jax.value_and_grad(simulate_pulse)
    value, grad = grad_fn(50.0)

    print(f"Gradient (dProb/dAmp): {grad:.8f}")

    if not jnp.isnan(value):
        print("\nSUCCESS: Simulation is stable and differentiable.")
        print("NEXT STEP: We can now optimize this pulse using AI.")
    else:
        print("\nFAIL: Still getting NaN. Check constants.")
