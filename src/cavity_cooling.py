import diffrax
import jax.numpy as jnp
from diffrax import PIDController

# --- 1. CONFIGURATION ---
# Based on Table 2 of the paper [cite: 355]
HBAR = 1.0  # Normalized units
WC = 5.0 * 2 * jnp.pi  # Cavity Freq (5 GHz)
WA = 5.0 * 2 * jnp.pi  # Atom Freq (Resonant)
G_MAX = 0.5 * 2 * jnp.pi  # Max Coupling (0.5 GHz)
KAPPA = 0.01  # Cavity Decay (Simulating the bath)

# Hilbert Space Sizes
N_CAV = 5  # Fock states (0 to 4) - sufficient for low temp
N_QUBIT = 2
DIM = N_CAV * N_QUBIT * N_QUBIT  # Total dimension = 5 * 2 * 2 = 20


class CavityCoolingSim:
    @staticmethod
    def get_operators():
        # 1. Cavity Operators (Annilihation 'a')
        # Matrix form of 'a' in Fock basis
        a_data = jnp.sqrt(jnp.arange(1, N_CAV))
        a_mat = jnp.diag(a_data, 1)

        # 2. Qubit Operators (Sigma_minus)
        sm_mat = jnp.array([[0.0, 1.0], [0.0, 0.0]])  # |g><e|
        id_q = jnp.eye(2)
        id_c = jnp.eye(N_CAV)

        # 3. Tensor Products to build full 20x20 operators
        # A = a (x) I (x) I
        A = jnp.kron(a_mat, jnp.kron(id_q, id_q))

        # SM1 = I (x) sm (x) I
        SM1 = jnp.kron(id_c, jnp.kron(sm_mat, id_q))

        # SM2 = I (x) I (x) sm
        SM2 = jnp.kron(id_c, jnp.kron(id_q, sm_mat))

        return A, SM1, SM2

    @staticmethod
    def dynamics(t, rho, args):
        # args = (pulse_func, A, SM1, SM2, A_dag, SM1_dag, SM2_dag)
        pulse_func, A, SM1, SM2, Ad, SM1d, SM2d = args

        # Get control value g(t)
        g_val = pulse_func(t)

        # --- HAMILTONIAN (Interaction Picture roughly) ---
        # H_int = g(t) * (a^dag * (sm1 + sm2) + a * (sm1^dag + sm2^dag))
        # Note: We assume resonance (WC=WA), so rotating terms cancel.

        collective_sm = SM1 + SM2
        collective_smd = SM1d + SM2d

        H = g_val * (jnp.matmul(Ad, collective_sm) + jnp.matmul(A, collective_smd))

        # Von Neumann: -i[H, rho]
        comm = -1j * (jnp.matmul(H, rho) - jnp.matmul(rho, H))

        # Dissipation (Cavity Decay to Bath)
        # L = sqrt(kappa) * a
        # D[L] = L rho L^dag - 0.5 {L^dag L, rho}
        L = jnp.sqrt(KAPPA) * A
        Ld = jnp.sqrt(KAPPA) * Ad

        jump = jnp.matmul(L, jnp.matmul(rho, Ld))
        anticomm = 0.5 * jnp.matmul(Ld, jnp.matmul(L, rho)) + 0.5 * jnp.matmul(
            rho, jnp.matmul(Ld, L)
        )
        dissipation = jump - anticomm

        return comm + dissipation


def run_simulation(pulse_amp):
    # Precompute operators
    A, SM1, SM2 = CavityCoolingSim.get_operators()
    Ad, SM1d, SM2d = A.T.conj(), SM1.T.conj(), SM2.T.conj()
    args = (lambda t: pulse_amp, A, SM1, SM2, Ad, SM1d, SM2d)

    # Initial State:
    # Cavity = Hot (Thermal state n=1)
    # Qubits = Cold (Ground state |gg>)

    # 1. Cavity Thermal State
    p_cav = jnp.exp(-0.5 * jnp.arange(N_CAV))  # Dummy thermal distribution
    p_cav = p_cav / jnp.sum(p_cav)
    rho_c = jnp.diag(p_cav)

    # 2. Qubits Ground State
    rho_q = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # |g><g|

    # 3. Full State
    rho_init = jnp.kron(rho_c, jnp.kron(rho_q, rho_q))
    rho_init = jnp.array(rho_init, dtype=jnp.complex64)

    # Solver
    term = diffrax.ODETerm(CavityCoolingSim.dynamics)
    solver = diffrax.Kvaerno5()

    # Run for interaction time tau = 50 ns [cite: 355]
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=0.2,
        dt0=0.001,
        y0=rho_init,
        args=args,
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-6),
        max_steps=5000,
    )

    final_rho = sol.ys[-1]

    # Calculate Final Cavity Occupation <a^dag a>
    # Trace out qubits
    # This is a bit tricky in tensor form, doing simplified trace:
    # We just want Tr( (a^dag a x I x I) * rho )

    NumberOp = jnp.matmul(Ad, A)
    n_mean = jnp.trace(jnp.matmul(NumberOp, final_rho))

    return jnp.real(n_mean)


if __name__ == "__main__":
    print("Testing Cavity Cooling Physics...")
    # Test with g = 0 (No cooling)
    n_hot = run_simulation(0.0)
    print(f"Initial Phonons (No Coupling): {n_hot:.4f}")

    # Test with g = 0.5 (Strong Coupling)
    n_cooled = run_simulation(0.5)
    print(f"Final Phonons (Coupling ON):   {n_cooled:.4f}")

    if n_cooled < n_hot:
        print("SUCCESS: Cooling physics is active.")
