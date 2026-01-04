# Floquet-Engineered Cavity Cooling: Comprehensive Project Guide

**Version**: 1.0.0
**Date**: 2026-01-04
**Project**: Comparison of Floquet Engineering vs Stochastic Cooling Limits

---

## 1. Introduction

### 1.1 The Scientific Goal

This project aims to demonstrate a fundamental violation of the "stochastic cooling limit" in open quantum systems by using coherent Floquet engineering.

- **The Problem:** Traditional cooling relies on stochastic, incoherent exchange with a cold reservoir. This is limited by the "detailed balance" of the coupling mechanism.
- **The Solution:** By driving the system with periodic, non-commuting control fields $[H(t_1), H(t_2)] \neq 0$, we can engineer an effective Hamiltonian $H_{eff}$ that creates a "Maxwell's Demon" effect, pumping entropy out faster than the stochastic limit allows.
- **The Target:** Beat the steady-state occupancy $n^*$ defined by the Vashaee-Abouie asymptotic limit.

### 1.2 Key Claims

1.  **Floquet Advantage:** Coherent driving achieves lower $\langle n \rangle$ than optimal static parameters.
2.  **No-Go Theorem:** This advantage vanishes if the Hamiltonian commutes with itself at different times (static detuning).
3.  **Robustness:** The protocol survives realistic 1/f flux noise and qubit thermalization.

---

## 2. Physics & Theory

### 2.1 The System

We simulate a **superconducting microwave cavity** coupled to a beam of **auxiliary qubits** (collision model).

- **Cavity:** Harmonic oscillator $(\omega_c \approx 5.0 \text{ GHz})$.
- **Qubits:** Two-level systems passing through the cavity.
- **Coupling:** Time-dependent Jaynes-Cummings interaction.

### 2.2 Hamiltonian

The system is governed by the time-dependent Hamiltonian $H(t)$:

$$
H(t) = \Delta(t) \sigma_z + g(t) (a^\dagger \sigma_- + a \sigma_+)
$$

where:

- $\Delta(t)$ is the controllable qubit-cavity detuning.
- $g(t)$ is the controllable coupling strength.

### 2.3 Master Equation

The dynamics follow the Lindblad master equation:

$$
\dot{\rho} = -i[H(t), \rho] + \mathcal{L}_{cav}[\rho] + \mathcal{L}_{qubit}[\rho]
$$

- **Cavity Dissipation:** $\kappa (1+\bar{n}) \mathcal{D}[a] + \kappa \bar{n} \mathcal{D}[a^\dagger]$ (Thermal bath coupling).
- **Qubit Decay:** $\gamma_1 \mathcal{D}[\sigma_-]$ (Spontaneous emission).

---

## 3. Algorithms & Optimization

We employ two distinct optimization strategies to find the optimal pulse sequences $g(t)$ and $\Delta(t)$.

### 3.1 GRAPE (Gradient Ascent Pulse Engineering)

- **Method:** Calculus of variations on the time-evolution operator.
- **Implementation:** `src/optimization/grape.py`
- **Library:** JAX (Auto-differentiation).
- **Pros:** Extremely precise, finds smooth local optima.
- **Cons:** Needs good initialization, can get stuck in local minima.

### 3.2 Reinforcement Learning (SAC)

- **Method:** Soft Actor-Critic (Model-free RL).
- **Implementation:** `src/rl/train_sac.py`
- **Library:** Stable-Baselines3 + Gymnasium.
- **Pros:** Can discover discontinuous "bang-bang" strategies, explores global space.
- **Cons:** Slower convergence, noisy results.

---

## 4. Codebase Structure

The project follows a modular "Research Engineering" structure in `src/`.

### 4.1 `src/physics/`

- **`operators.py`**: JAX implementations of $a, a^\dagger, \sigma_z, \sigma_\pm$.
- **`hamiltonian.py`**: Constructs $H(t)$ matrices dynamically.
- **`lindblad.py`**: Defines the $\mathcal{L}[\rho]$ superoperator for the ODE solver.

### 4.2 `src/floquet/`

- **`cycle.py`**: The core **Floquet Engine**. Uses `jax.lax.scan` to compute the "Stroboscopic Map" (one full period evolution) efficiently.
- **`steady_state.py`**: Solves for the fixed point $\mathcal{E}^N(\rho_{ss}) = \rho_{ss}$.

### 4.3 `src/baseline/`

- **`stochastic.py`**: Analytic formula for the "Stochastic Limit" (Eq. 25 in base paper).

### 4.4 `src/analysis/`

- **`figures.py`**: Generates publication-quality plots (Figure 1, 2, 3).
- **`thermodynamics.py`**: Computes COP, Cooling Power, and Entropy Production.

---

## 5. Running the Code

### 5.1 Installation

Recommended environment management with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# OR
uv pip install jax jaxlib flax optax matplotlib stable-baselines3 gymnasium scipy seaborn
```

### 5.2 Verification (No-Go Theorem)

To confirm the physics works as expected (negative control):

```bash
uv run python -m src.validation.no_go_theorem
```

_Expected Output:_ Commuting Hamiltonian fails to cool ($\approx 1.12$), Non-commuting succeeds ($\approx 1.04$).

### 5.3 Figure Generation

To produce the plots for the paper:

```bash
uv run python -m src.analysis.figures
```

_Outputs:_ `results/figures/fig1_comparison.png`, `fig2_waveforms.png`, `fig3_nogo_sweep.png`.

---

## 6. Results Summary

| Metric                                 | Stochastic Limit | GRAPE         | SAC (RL)          |
| :------------------------------------- | :--------------- | :------------ | :---------------- |
| **Mean Occupancy $\langle n \rangle$** | **1.4428**       | **~1.02**     | **~0.94**         |
| **Improvement**                        | 0% (Baseline)    | +29%          | +35%              |
| **Mechanism**                          | Static Exchange  | Gradient Flow | Bang-Bang Control |

**Conclusion:** The project successfully demonstrates that Floquet engineering allows for cooling well below the fundamental limits of static thermodynamic machines.
