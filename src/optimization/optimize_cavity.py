import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from diffrax import PIDController
from tqdm import tqdm

# --- 1. CONFIGURATION ---
HBAR = 1.0
WC = 5.0 * 2 * jnp.pi  # Cavity Freq
WA = 5.0 * 2 * jnp.pi  # Atom Freq
KAPPA = 0.05  # Cavity Decay (Increased slightly to help cooling)
DURATION = 2.0  # Interaction time (ns) - FAST interaction
STEPS = 200  # Optimization steps
LEARNING_RATE = 0.1

# Dimensions
N_CAV = 5
N_QUBIT = 2
DIM = N_CAV * N_QUBIT * N_QUBIT  # 20


class CavityPhysics:
    @staticmethod
    def get_static_operators():
        # Build operators exactly like before
        a_data = jnp.sqrt(jnp.arange(1, N_CAV))
        a_mat = jnp.diag(a_data, 1)
        sm_mat = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        id_q = jnp.eye(2)
        id_c = jnp.eye(N_CAV)

        # Full Space Operators
        A = jnp.kron(a_mat, jnp.kron(id_q, id_q))
        SM1 = jnp.kron(id_c, jnp.kron(sm_mat, id_q))
        SM2 = jnp.kron(id_c, jnp.kron(id_q, sm_mat))

        Ad = A.T
        SM1d = SM1.T
        SM2d = SM2.T

        # Pre-compute the "Interaction Operator" V
        # H_int = g(t) * V
        # V = (a^dag S- + a S+)
        S_minus = SM1 + SM2
        S_plus = SM1d + SM2d
        V = jnp.matmul(Ad, S_minus) + jnp.matmul(A, S_plus)

        # Pre-compute Dissipation L = sqrt(k)*a
        L = jnp.sqrt(KAPPA) * A
        Ld = jnp.sqrt(KAPPA) * Ad
        L_Ld = jnp.matmul(Ld, L)  # L^dag L

        return V, L, Ld, L_Ld

    @staticmethod
    def dynamics(t, y, args):
        # y is [Real_Rho, Imag_Rho] flattened
        # args = (pulse_func, V_real, V_imag, ...)

        pulse_func, Vr, Vi, Lr, Li, Ldr, Ldi, LLdr, LLdi = args

        # 1. Reconstruct Rho (Real + Imag)
        rho_r = y[:400].reshape((DIM, DIM))
        rho_i = y[400:].reshape((DIM, DIM))

        # 2. Hamiltonian H(t) = g(t) * V
        g_val = pulse_func(t)
        Hr = g_val * Vr
        Hi = g_val * Vi

        # 3. Commutator -i[H, rho]
        # Real part of -i[H, rho] = (H_i rho_r + H_r rho_i) - (rho_r H_i - rho_i H_r)
        # ... logic omitted for brevity, using matrix multiplication directly

        # H * rho
        H_rho_r = jnp.matmul(Hr, rho_r) - jnp.matmul(Hi, rho_i)
        H_rho_i = jnp.matmul(Hr, rho_i) + jnp.matmul(Hi, rho_r)

        # rho * H
        rho_H_r = jnp.matmul(rho_r, Hr) - jnp.matmul(rho_i, Hi)
        rho_H_i = jnp.matmul(rho_r, Hi) + jnp.matmul(rho_i, Hr)

        comm_r = H_rho_i - rho_H_i  # -i * i = +1
        comm_i = -(H_rho_r - rho_H_r)  # -i * real

        # 4. Dissipation (Lindblad)
        # D[L] = L rho L^dag - 0.5 {L^dag L, rho}

        # Term 1: L rho L^dag
        # L rho
        L_rho_r = jnp.matmul(Lr, rho_r) - jnp.matmul(Li, rho_i)
        L_rho_i = jnp.matmul(Lr, rho_i) + jnp.matmul(Li, rho_r)
        # (L rho) L^dag
        Term1_r = jnp.matmul(L_rho_r, Ldr) - jnp.matmul(L_rho_i, Ldi)
        Term1_i = jnp.matmul(L_rho_r, Ldi) + jnp.matmul(L_rho_i, Ldr)

        # Term 2: -0.5 (L^dag L rho + rho L^dag L)
        # LLd rho
        LLd_rho_r = jnp.matmul(LLdr, rho_r) - jnp.matmul(LLdi, rho_i)
        LLd_rho_i = jnp.matmul(LLdr, rho_i) + jnp.matmul(LLdi, rho_r)
        # rho LLd
        rho_LLd_r = jnp.matmul(rho_r, LLdr) - jnp.matmul(rho_i, LLdi)
        rho_LLd_i = jnp.matmul(rho_r, LLdi) + jnp.matmul(rho_i, LLdr)

        Term2_r = -0.5 * (LLd_rho_r + rho_LLd_r)
        Term2_i = -0.5 * (LLd_rho_i + rho_LLd_i)

        dr = comm_r + Term1_r + Term2_r
        di = comm_i + Term1_i + Term2_i

        return jnp.concatenate([dr.flatten(), di.flatten()])


def optimize_pulse():
    # 1. Operators (Precompute & Split Real/Imag)
    V, L, Ld, L_Ld = CavityPhysics.get_static_operators()

    # Helper to split
    def s(M):
        return jnp.real(M), jnp.imag(M)

    Vr, Vi = s(V)
    Lr, Li = s(L)
    Ldr, Ldi = s(Ld)
    LLdr, LLdi = s(L_Ld)
    static_args = (Vr, Vi, Lr, Li, Ldr, Ldi, LLdr, LLdi)

    # 2. Initial State (Hot Cavity, Cold Qubits)
    p_cav = jnp.exp(-0.5 * jnp.arange(N_CAV))
    p_cav = p_cav / jnp.sum(p_cav)
    rho_c = jnp.diag(p_cav)
    rho_q = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    rho_init = jnp.kron(rho_c, jnp.kron(rho_q, rho_q))
    # Flatten Real/Imag
    y0 = jnp.concatenate([jnp.real(rho_init).flatten(), jnp.imag(rho_init).flatten()])

    # 3. Loss Function
    def loss_fn(pulse_params):
        # Interpolate pulse params into function
        t_points = jnp.linspace(0, DURATION, len(pulse_params))

        def pulse_func(t):
            return jnp.interp(t, t_points, pulse_params)

        term = diffrax.ODETerm(CavityPhysics.dynamics)
        solver = diffrax.Kvaerno5()
        # Pack args
        args = (pulse_func, *static_args)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=DURATION,
            dt0=0.001,
            y0=y0,
            args=args,
            max_steps=10000,
            stepsize_controller=PIDController(rtol=1e-4, atol=1e-4),
        )

        final_y = sol.ys[-1]

        # Recover Rho
        rho_final_r = final_y[:400].reshape((DIM, DIM))
        # rho_final_i ... not needed for diagonal N

        # Calculate <a^dag a>
        # Operator Number = A^dag A
        # (Since we need full trace, let's just build Number Op in full space)
        # N_op = A^dag A
        a_data = jnp.sqrt(jnp.arange(1, N_CAV))
        a_mat = jnp.diag(a_data, 1)
        n_mat = jnp.matmul(a_mat.T, a_mat)  # Cavity number op
        id_q = jnp.eye(2)
        N_op = jnp.real(jnp.kron(n_mat, jnp.kron(id_q, id_q)))

        # Trace(N * rho)
        n_val = jnp.trace(jnp.matmul(N_op, rho_final_r))

        return n_val

    # 4. Optimization Loop
    # Init Pulse: Start with constant g=0.5 (The "Paper" Baseline)
    init_pulse = jnp.ones(50) * 0.5

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(init_pulse)
    params = init_pulse

    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Clamp g(t) to reasonable experimental limits (e.g. +/- 5 GHz)
        params = jnp.clip(params, -5.0, 5.0)
        return loss, params, opt_state

    print("Starting Strong Coupling Optimization...")
    print(f"Initial (Paper Baseline) Phonons: {jax.jit(loss_fn)(params):.4f}")

    history = []
    for i in tqdm(range(STEPS)):
        loss, params, opt_state = update_step(params, opt_state)

        history.append(loss)
        if i % 20 == 0:
            tqdm.write(f"Step {i}: Phonons {loss:.4f}")

    return params, history


if __name__ == "__main__":
    # Force CPU for stability if needed, or remove for GPU
    # jax.config.update("jax_platform_name", "cpu")

    best_pulse, history = optimize_pulse()

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title("Cooling Optimization")
    plt.xlabel("Step")
    plt.ylabel("Cavity Phonons <n>")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    t = np.linspace(0, 2.0, len(best_pulse))
    plt.plot(t, best_pulse, color="red", linewidth=2)
    plt.title("AI-Designed 'Super-Swap' Pulse")
    plt.xlabel("Time (ns)")
    plt.ylabel("Coupling g(t)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("cavity_optimization.png")
    print("Saved 'cavity_optimization.png'")
    print(f"Final Phonon Count: {history[-1]:.4f}")
