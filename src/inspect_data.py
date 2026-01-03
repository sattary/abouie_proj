import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
pulses = np.load("dataset_pulses.npy")  # Shape (5000, 200)
scores = np.load("dataset_scores.npy")  # Shape (5000,)

# 2. Find Best and Worst indices
best_idx = np.argmax(scores)
worst_idx = np.argmin(scores)

print(f"Best Pulse Index: {best_idx}")
print(f"Best Score (Prob): {scores[best_idx]:.4f}")
print(f"Worst Score (Prob): {scores[worst_idx]:.4f}")

# 3. Time axis (must match the generation step)
t = np.linspace(0, 20, 200)

# 4. Plot
plt.figure(figsize=(12, 6))

# Plot Best
plt.subplot(1, 2, 1)
plt.plot(t, pulses[best_idx], color="green", linewidth=2)
plt.title(f"The 'Champion' Pulse\nCooling: {scores[best_idx] * 100:.2f}%")
plt.xlabel("Time (ns)")
plt.ylabel("Detuning epsilon (ueV)")
plt.grid(True, alpha=0.3)

# Plot Worst
plt.subplot(1, 2, 2)
plt.plot(t, pulses[worst_idx], color="red", linewidth=2)
plt.title(f"A Failed Pulse\nCooling: {scores[worst_idx] * 100:.2f}%")
plt.xlabel("Time (ns)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pulse_comparison.png")
print("\nSaved plot to 'pulse_comparison.png'. Check it now!")
plt.show()
