import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- 1. THE NEURAL NETWORK (1D U-Net) ---
class TimeEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        # Sinusoidal embedding for diffusion time steps
        half_dim = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return nn.Dense(self.dim)(embedding)


class PulseDiffuser(nn.Module):
    @nn.compact
    def __call__(self, x, t, condition, training=True):
        # x: Pulse shape [Batch, 200]
        # t: Time step [Batch]
        # condition: Cooling Score [Batch, 1]

        # 1. Expand inputs to match dimensions
        # Treat 1D signal as image with 1 channel: [Batch, 200, 1]
        x = x[..., None]

        # 2. Embeddings
        t_emb = TimeEmbedding(64)(t)
        c_emb = nn.Dense(64)(condition)
        emb = t_emb + c_emb
        emb = nn.swish(emb)[:, None, :]  # Broadcast to time dim

        # 3. Downsampling (Encoder)
        h1 = nn.Conv(32, kernel_size=(5,))(x)
        h1 = nn.swish(h1 + nn.Dense(32)(emb))  # Add time info

        h2 = nn.Conv(64, kernel_size=(5,), strides=(2,))(h1)  # Downsample
        h2 = nn.swish(h2 + nn.Dense(64)(emb))

        h3 = nn.Conv(128, kernel_size=(5,), strides=(2,))(h2)  # Downsample
        h3 = nn.swish(h3 + nn.Dense(128)(emb))

        # 4. Bottleneck
        h_mid = nn.Conv(256, kernel_size=(3,))(h3)
        h_mid = nn.swish(h_mid)

        # 5. Upsampling (Decoder)
        u1 = nn.ConvTranspose(128, kernel_size=(5,), strides=(2,))(h_mid)
        # Pad/Trim to match h2 shape if needed (simple hack for now: slice)
        u1 = u1[:, : h2.shape[1], :]
        u1 = jnp.concatenate([u1, h2], axis=-1)  # Skip Connection
        u1 = nn.Conv(128, kernel_size=(3,))(u1)
        u1 = nn.swish(u1 + nn.Dense(128)(emb))

        u2 = nn.ConvTranspose(64, kernel_size=(5,), strides=(2,))(u1)
        u2 = u2[:, : h1.shape[1], :]
        u2 = jnp.concatenate([u2, h1], axis=-1)  # Skip Connection
        u2 = nn.Conv(64, kernel_size=(3,))(u2)
        u2 = nn.swish(u2 + nn.Dense(64)(emb))

        # 6. Output
        out = nn.Conv(1, kernel_size=(3,))(u2)  # Back to 1 channel
        return out.squeeze(-1)  # [Batch, 200]


# --- 2. DIFFUSION LOGIC ---
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


# Constants
TIMESTEPS = 200
BETAS = cosine_beta_schedule(TIMESTEPS)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = jnp.cumprod(ALPHAS, axis=0)


# --- 3. TRAINING LOOP ---
@jax.jit
def train_step(state, batch_x, batch_cond, key):
    # Sample random noise and timesteps
    noise = jax.random.normal(key, batch_x.shape)
    t = jax.random.randint(key, (batch_x.shape[0],), 0, TIMESTEPS)

    # Forward Diffusion (Add Noise)
    # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
    alpha_hat = ALPHAS_CUMPROD[t][:, None]
    noisy_x = jnp.sqrt(alpha_hat) * batch_x + jnp.sqrt(1 - alpha_hat) * noise

    def loss_fn(params):
        predicted_noise = state.apply_fn(params, noisy_x, t, batch_cond)
        return jnp.mean((noise - predicted_noise) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    print("Loading Dataset...")
    # Load data
    pulses = np.load("dataset_pulses.npy")
    scores = np.load("dataset_scores.npy")

    # Normalize Pulses (Critical for Neural Nets)
    mean_p = np.mean(pulses)
    std_p = np.std(pulses)
    pulses = (pulses - mean_p) / std_p

    # Filter only "Good" pulses for training (Top 20%)
    # We want the AI to learn ONLY success, not failure.
    threshold = np.percentile(scores, 80)
    good_indices = scores > threshold
    train_x = pulses[good_indices]
    train_c = scores[good_indices][:, None]  # Condition

    print(f"Training on {len(train_x)} 'Elite' pulses (Score > {threshold:.2f})")

    # Initialize Model
    key = jax.random.PRNGKey(42)
    model = PulseDiffuser()
    dummy_x = jnp.ones((1, 200))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_c = jnp.ones((1, 1))

    params = model.init(key, dummy_x, dummy_t, dummy_c)
    tx = optax.adam(learning_rate=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Train
    loss_history = []
    batch_size = 64
    epochs = 500  # Fast training

    for epoch in tqdm(range(epochs)):
        # Random batch
        idx = np.random.choice(len(train_x), batch_size)
        batch_x = jnp.array(train_x[idx])
        batch_c = jnp.array(train_c[idx])

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, batch_x, batch_c, subkey)
        loss_history.append(loss)

    # Save Model Weights (Simulated saving)
    print("\nTraining Complete!")
    plt.plot(loss_history)
    plt.title("Diffusion Model Training Loss")
    plt.savefig("training_loss.png")

    # Save statistics for later un-normalization
    np.save("norm_stats.npy", np.array([mean_p, std_p]))

    # --- RETURN STATE FOR INFERENCE ---
    return state


if __name__ == "__main__":
    trained_state = main()
    # Serialize params for the next step (Phase 4)
    from flax.serialization import to_bytes

    with open("model_params.msgpack", "wb") as f:
        f.write(to_bytes(trained_state.params))
    print("Model saved to 'model_params.msgpack'")
