import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes, msgpack_restore
import flax


# --- 1. DEFINE MODEL INLINE (Ensures consistency) ---
class TimeEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        half_dim = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return nn.Dense(self.dim)(embedding)


class PulseDiffuser(nn.Module):
    @nn.compact
    def __call__(self, x, t, condition, training=True):
        x = x[..., None]
        t_emb = TimeEmbedding(64)(t)
        c_emb = nn.Dense(64)(condition)
        emb = t_emb + c_emb
        emb = nn.swish(emb)[:, None, :]

        # Encoder
        h1 = nn.Conv(32, kernel_size=(5,))(x)
        h1 = nn.swish(h1 + nn.Dense(32)(emb))

        h2 = nn.Conv(64, kernel_size=(5,), strides=(2,))(h1)
        h2 = nn.swish(h2 + nn.Dense(64)(emb))

        h3 = nn.Conv(128, kernel_size=(5,), strides=(2,))(h2)
        h3 = nn.swish(h3 + nn.Dense(128)(emb))

        # Bottleneck
        h_mid = nn.Conv(256, kernel_size=(3,))(h3)
        h_mid = nn.swish(h_mid)

        # Decoder
        u1 = nn.ConvTranspose(128, kernel_size=(5,), strides=(2,))(h_mid)
        u1 = u1[:, : h2.shape[1], :]
        u1 = jnp.concatenate([u1, h2], axis=-1)
        u1 = nn.Conv(128, kernel_size=(3,))(u1)
        u1 = nn.swish(u1 + nn.Dense(128)(emb))

        u2 = nn.ConvTranspose(64, kernel_size=(5,), strides=(2,))(u1)
        u2 = u2[:, : h1.shape[1], :]
        u2 = jnp.concatenate([u2, h1], axis=-1)
        u2 = nn.Conv(64, kernel_size=(3,))(u2)
        u2 = nn.swish(u2 + nn.Dense(64)(emb))

        out = nn.Conv(1, kernel_size=(3,))(u2)
        return out.squeeze(-1)


# --- 2. CONSTANTS ---
TIMESTEPS = 200


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


BETAS = cosine_beta_schedule(TIMESTEPS)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = jnp.cumprod(ALPHAS, axis=0)


# --- 3. SAFE LOAD & SAMPLING ---
def sample(model, params, condition_value, shape=(1, 200)):
    key = jax.random.PRNGKey(123)  # New seed
    x = jax.random.normal(key, shape)
    cond_tensor = jnp.full((shape[0], 1), condition_value)

    for i in reversed(range(TIMESTEPS)):
        t = jnp.full((shape[0],), i, dtype=jnp.int32)
        predicted_noise = model.apply(params, x, t, cond_tensor)

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
    try:
        mean_p, std_p = np.load("norm_stats.npy")
    except Exception:
        print("Warning: norm_stats.npy not found, using default.")
        mean_p, std_p = 0.0, 1.0

    model = PulseDiffuser()
    dummy_x = jnp.ones((1, 200))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_c = jnp.ones((1, 1))

    # Initialize target structure
    variables = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_c)
    target_params = variables["params"]

    # --- ROBUST LOADING ---
    with open("model_params.msgpack", "rb") as f:
        file_bytes = f.read()

    try:
        # Try direct load
        params = from_bytes(target_params, file_bytes)
    except ValueError:
        print("Direct load failed. Attempting to unwrap dictionary...")
        # Inspect structure
        raw_state = msgpack_restore(file_bytes)
        if "params" in raw_state:
            # If the file was saved as {'params': ...} but we need the inner content
            params = from_bytes(
                target_params, flax.serialization.to_bytes(raw_state["params"])
            )
        else:
            print("CRITICAL: File structure unknown. Keys found:", raw_state.keys())
            raise

    print("Generating the 'Dream Pulse'...")
    # Target: 3.0 standard deviations above mean (High Performance)
    generated_pulse_norm = sample(model, {"params": params}, 3.0)

    final_pulse = generated_pulse_norm * std_p + mean_p
    final_pulse = np.array(final_pulse[0])

    # Plot
    t = np.linspace(0, 20, 200)
    plt.figure(figsize=(10, 6))
    plt.plot(t, final_pulse, color="purple", linewidth=3, label="AI Generated")
    plt.title("The AI-Designed Cooling Protocol")
    plt.grid(True, alpha=0.3)
    plt.savefig("ai_solution.png")

    np.save("ai_pulse.npy", final_pulse)
    print("\nSuccess! Saved 'ai_solution.png' and 'ai_pulse.npy'.")
    print("Run verify_solution.py next!")


if __name__ == "__main__":
    main()
