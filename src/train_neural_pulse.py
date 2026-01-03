import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffrax import PIDController
from flax import linen as nn
from tqdm import tqdm

# --- 1. CONFIGURATION ---
HBAR = 1.0
WC = 5.0 * 2 * jnp.pi
WA = 5.0 * 2 * jnp.pi
KAPPA = 0.05
DURATION = 10.0  # FIXED: Increased from 2.0 to 10.0 (Enough time for swaps)
STEPS = 500  # More training steps
LEARNING_RATE = 0.005  # Slower, deeper learning

# Dimensions
N_CAV = 5
N_QUBIT = 2
DIM = N_CAV * N_QUBIT * N_QUBIT  # 20


# --- 2. THE NEURAL PULSE GENERATOR (True AI) ---
class PulseNet(nn.Module):
    @nn.compact
    def __call__(self, t):
        # Input: Time t (scalar)
        # We assume t is normalized 0..1 inside, so we divide by DURATION roughly
        inputs = jnp.array([t / DURATION])

        # Small MLP
        x = nn.Dense(32)(inputs)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)  # Output: Coupling strength g

        # Physics Constraint: Tanh to keep g in range [-5, 5]
        return 5.0 * nn.tanh(x)[0]


# --- 3. PHYSICS ENGINE (Same as before) ---
class CavityPhysics:
    @staticmethod
    def get_static_operators():
        a_data = jnp.sqrt(jnp.arange(1, N_CAV))
        a_mat = jnp.diag(a_data, 1)
        sm_mat = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        id_q = jnp.eye(2)
        id_c = jnp.eye(N_CAV)

        A = jnp.kron(a_mat, jnp.kron(id_q, id_q))
        SM1 = jnp.kron(id_c, jnp.kron(sm_mat, id_q))
        SM2 = jnp.kron(id_c, jnp.kron(id_q, sm_mat))

        Ad = A.T
        S_minus = SM1 + SM2
        S_plus = SM1.T + SM2.T
        V = jnp.matmul(Ad, S_minus) + jnp.matmul(A, S_plus)

        L = jnp.sqrt(KAPPA) * A
        Ld = jnp.sqrt(KAPPA) * Ad
        L_Ld = jnp.matmul(Ld, L)

        return V, L, Ld, L_Ld

    @staticmethod
    def dynamics(t, y, args):
        model_apply, params, Vr, Vi, Lr, Li, Ldr, Ldi, LLdr, LLdi = args

        rho_r = y[:400].reshape((DIM, DIM))
        rho_i = y[400:].reshape((DIM, DIM))

        # AI GENERATED CONTROL
        g_val = model_apply(params, t)

        Hr = g_val * Vr
        Hi = g_val * Vi

        # Commutator
        H_rho_r = jnp.matmul(Hr, rho_r) - jnp.matmul(Hi, rho_i)
        H_rho_i = jnp.matmul(Hr, rho_i) + jnp.matmul(Hi, rho_r)
        rho_H_r = jnp.matmul(rho_r, Hr) - jnp.matmul(rho_i, Hi)
        rho_H_i = jnp.matmul(rho_r, Hi) + jnp.matmul(rho_i, Hr)

        comm_r = H_rho_i - rho_H_i
        comm_i = -(H_rho_r - rho_H_r)

        # Dissipation
        L_rho_r = jnp.matmul(Lr, rho_r) - jnp.matmul(Li, rho_i)
        L_rho_i = jnp.matmul(Lr, rho_i) + jnp.matmul(Li, rho_r)

        Term1_r = jnp.matmul(L_rho_r, Ldr) - jnp.matmul(L_rho_i, Ldi)
        Term1_i = jnp.matmul(L_rho_r, Ldi) + jnp.matmul(L_rho_i, Ldr)

        LLd_rho_r = jnp.matmul(LLdr, rho_r) - jnp.matmul(LLdi, rho_i)
        LLd_rho_i = jnp.matmul(LLdr, rho_i) + jnp.matmul(LLdi, rho_r)

        rho_LLd_r = jnp.matmul(rho_r, LLdr) - jnp.matmul(rho_i, LLdi)
        rho_LLd_i = jnp.matmul(rho_r, LLdi) + jnp.matmul(rho_i, LLdr)

        Term2_r = -0.5 * (LLd_rho_r + rho_LLd_r)
        Term2_i = -0.5 * (LLd_rho_i + rho_LLd_i)

        dr = comm_r + Term1_r + Term2_r
        di = comm_i + Term1_i + Term2_i

        return jnp.concatenate([dr.flatten(), di.flatten()])


def train_neural_pulse():
    # 1. Init Operators
    V, L, Ld, L_Ld = CavityPhysics.get_static_operators()

    def s(M):
        return jnp.real(M), jnp.imag(M)

    Vr, Vi = s(V)
    Lr, Li = s(L)
    Ldr, Ldi = s(Ld)
    LLdr, LLdi = s(L_Ld)
    static_ops = (Vr, Vi, Lr, Li, Ldr, Ldi, LLdr, LLdi)

    # 2. Init State
    p_cav = jnp.exp(-0.5 * jnp.arange(N_CAV))
    p_cav = p_cav / jnp.sum(p_cav)
    rho_c = jnp.diag(p_cav)
    rho_q = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    rho_init = jnp.kron(rho_c, jnp.kron(rho_q, rho_q))
    y0 = jnp.concatenate([jnp.real(rho_init).flatten(), jnp.imag(rho_init).flatten()])

    # 3. Init Neural Network
    model = PulseNet()
    key = jax.random.PRNGKey(0)
    # Init params with a dummy input
    init_params = model.init(key, 0.0)

    # 4. Loss Function
    def loss_fn(params):
        term = diffrax.ODETerm(CavityPhysics.dynamics)
        solver = diffrax.Kvaerno5()
        # Pass model.apply and params to the solver
        args = (model.apply, params, *static_ops)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=DURATION,
            dt0=0.01,
            y0=y0,
            args=args,
            max_steps=10000,
            stepsize_controller=PIDController(rtol=1e-4, atol=1e-4),
        )

        final_y = sol.ys[-1]
        rho_final_r = final_y[:400].reshape((DIM, DIM))

        # Calculate Number of Phonons
        a_data = jnp.sqrt(jnp.arange(1, N_CAV))
        a_mat = jnp.diag(a_data, 1)
        n_mat = jnp.matmul(a_mat.T, a_mat)
        id_q = jnp.eye(2)
        N_op = jnp.real(jnp.kron(n_mat, jnp.kron(id_q, id_q)))

        return jnp.trace(jnp.matmul(N_op, rho_final_r))

    # 5. Training Loop
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(init_params)
    params = init_params

    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    print("Training Neural Pulse (AI Control)...")
    print("Goal: Beat the Paper Baseline (0.24)")

    history = []

    for i in tqdm(range(STEPS)):
        loss, params, opt_state = update_step(params, opt_state)

        history.append(loss)

        if i % 20 == 0:
            tqdm.write(f"Epoch {i}: Phonons {loss:.4f}")

    return params, model, history


if __name__ == "__main__":
    best_params, model, history = train_neural_pulse()

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.axhline(0.24, color="red", linestyle="--", label="Paper Baseline")
    plt.title("Neural Training")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    t_vals = jnp.linspace(0, DURATION, 200)
    # Reconstruct the learned pulse
    g_vals = [model.apply(best_params, t) for t in t_vals]
    plt.plot(t_vals, g_vals, color="purple", linewidth=2)
    plt.title("AI-Generated Control Pulse")
    plt.xlabel("Time (ns)")
    plt.ylabel("Coupling g(t)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("neural_pulse_result.png")
    print(f"Final Result: {history[-1]:.4f} Phonons")
    print("Saved 'neural_pulse_result.png'")
