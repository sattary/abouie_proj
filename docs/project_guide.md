# Floquet-Engineered Cavity Cooling: Comprehensive Project Guide

**Version**: 2.0.0
**Date**: 2026-01-04
**Author**: Reza Sattary
**Supervisor**: Dr. Abouie

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Physics & Theory](#2-physics--theory)
3. [System Parameters](#3-system-parameters)
4. [Algorithms & Optimization](#4-algorithms--optimization)
5. [Codebase Structure](#5-codebase-structure)
6. [Running the Code](#6-running-the-code)
7. [Validation Tiers](#7-validation-tiers)
8. [Presentation Figures](#8-presentation-figures)
9. [Results Summary](#9-results-summary)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 The Scientific Goal

This project demonstrates a **fundamental violation of the stochastic cooling limit** in open quantum systems using coherent Floquet engineering.

- **The Problem:** Traditional cooling relies on stochastic, incoherent exchange with a cold reservoir. This is fundamentally limited by "detailed balance" — the system cannot cool below a temperature set by the coupling mechanism.

- **The Solution:** By driving the system with periodic, **non-commuting** control fields where $[H(t_1), H(t_2)] \neq 0$, we engineer an effective Hamiltonian that acts as a "Maxwell's Demon", pumping entropy out faster than any static protocol.

- **The Target:** Beat the steady-state cavity occupation $n^*$ defined by the Vashaee-Abouie asymptotic limit (Eq. 25 in the original paper).

### 1.2 Key Scientific Claims

| Claim                 | Description                                                                        | Verification      |
| --------------------- | ---------------------------------------------------------------------------------- | ----------------- |
| **Floquet Advantage** | Coherent driving achieves lower $\langle n \rangle$ than optimal static parameters | Figures 1, 9      |
| **No-Go Theorem**     | Advantage vanishes if $[H(t_1), H(t_2)] = 0$ (static detuning)                     | Figures 3, 10     |
| **Robustness**        | Protocol survives 1/f flux noise and qubit thermalization                          | Figure 11         |
| **RWA Validity**      | Rotating-Wave Approximation holds for $g/\omega_c < 0.1$                           | Tier 2 validation |

---

## 2. Physics & Theory

### 2.1 The Physical System

We simulate a **superconducting microwave cavity** coupled to a beam of **cold auxiliary qubits** (collision model).

```
                    ┌─────────────────┐
   Thermal Bath ~~~~│     CAVITY      │<---- Cold Qubits (T_atom << T_bath)
   (T_bath)         │   (mode a)      │
                    └─────────────────┘
                            ↑
                    Control: g(t), Δ(t)
```

- **Cavity:** Harmonic oscillator at frequency $\omega_c \approx 5.0$ GHz
- **Qubits:** Two-level systems at frequency $\omega_a$ passing through the cavity
- **Coupling:** Time-dependent Jaynes-Cummings interaction

### 2.2 The Hamiltonian

The system is governed by the time-dependent Hamiltonian $H(t)$:

$$
H(t) = \frac{\Delta(t)}{2} (\sigma_z^{(1)} + \sigma_z^{(2)}) + g(t) (a^\dagger \sigma_-^{(1)} + a \sigma_+^{(1)} + a^\dagger \sigma_-^{(2)} + a \sigma_+^{(2)})
$$

where:

- $\Delta(t) = \omega_a - \omega_c$ is the controllable qubit-cavity detuning
- $g(t)$ is the controllable coupling strength
- Superscripts $(1), (2)$ denote the two auxiliary qubits

### 2.3 The Lindblad Master Equation

Open-system dynamics follow the Lindblad master equation:

$$
\dot{\rho} = -i[H(t), \rho] + \mathcal{L}_{cav}[\rho] + \mathcal{L}_{qubit}[\rho]
$$

**Dissipators:**

| Channel           | Form                                    | Physical Origin       |
| ----------------- | --------------------------------------- | --------------------- |
| Cavity emission   | $\kappa (1+\bar{n}) \mathcal{D}[a]$     | Thermal bath coupling |
| Cavity absorption | $\kappa \bar{n} \mathcal{D}[a^\dagger]$ | Thermal bath coupling |
| Qubit 1 decay     | $\gamma_1 \mathcal{D}[\sigma_-^{(1)}]$  | Spontaneous emission  |
| Qubit 2 decay     | $\gamma_1 \mathcal{D}[\sigma_-^{(2)}]$  | Spontaneous emission  |

where $\bar{n} = 1/(e^{\omega_c/T_{bath}} - 1)$ is the Bose-Einstein thermal occupation.

### 2.4 The Stochastic Limit

The **Vashaee-Abouie limit** gives the minimum achievable cavity occupation for static (non-Floquet) protocols:

$$
n^* = \frac{\kappa \bar{n} + \Gamma_{ex} \bar{n}_{atom}}{\kappa + \Gamma_{ex}}
$$

where $\Gamma_{ex}$ is the exchange rate and $\bar{n}_{atom}$ is the qubit thermal occupation.

---

## 3. System Parameters

### 3.1 Default Parameter Values

| Parameter         | Symbol     | Default Value | Units | Description            |
| ----------------- | ---------- | ------------- | ----- | ---------------------- |
| Cavity frequency  | $\omega_c$ | 5.0           | GHz   | Microwave resonator    |
| Qubit frequency   | $\omega_a$ | 5.0           | GHz   | Auxiliary qubit        |
| Cavity decay rate | $\kappa$   | 0.05          | GHz   | Bath coupling strength |
| Qubit decay rate  | $\gamma_1$ | 0.01          | GHz   | Spontaneous emission   |
| Bath temperature  | $T_{bath}$ | 0.5           | K     | Hot reservoir          |
| Atom temperature  | $T_{atom}$ | 0.01-0.05     | K     | Cold auxiliary qubits  |
| Hilbert space dim | $n_{fock}$ | 5             | -     | Cavity Fock states     |

### 3.2 Control Parameter Ranges

| Parameter         | Symbol      | Typical Range | Notes               |
| ----------------- | ----------- | ------------- | ------------------- |
| Coupling strength | $g(t)$      | 0.0 - 1.5     | GHz, time-dependent |
| Detuning          | $\Delta(t)$ | -0.5 - 0.5    | GHz, time-dependent |
| Cycle period      | $T_{cycle}$ | 0.5 - 1.0     | ns                  |
| Steps per cycle   | $n_{steps}$ | 10 - 50       | Discretization      |

### 3.3 Physical Validity Constraints

- **RWA validity:** $g/\omega_c < 0.1$ (checked in Tier 2)
- **Markov approximation:** $\kappa \ll \omega_c$
- **Weak coupling:** $g \ll \omega_c$

---

## 4. Algorithms & Optimization

### 4.1 GRAPE (Gradient Ascent Pulse Engineering)

**Method:** Variational optimization using automatic differentiation.

**Implementation:** `src/optimization/grape.py`

**Key Components:**

- Cost function: $\mathcal{L} = \langle n \rangle_{ss}$ (steady-state occupation)
- Optimizer: Adam (via `optax`)
- Gradients: Computed via `jax.grad` through the entire Floquet cycle

**Hyperparameters:**

| Parameter       | Typical Value |
| --------------- | ------------- |
| `n_iterations`  | 100-2000      |
| `learning_rate` | 0.01-0.02     |
| `n_cycles_eval` | 50-100        |

**Pros:** Precise, smooth solutions, fast convergence
**Cons:** Local minima, requires good initialization

### 4.2 Reinforcement Learning (SAC)

**Method:** Soft Actor-Critic (model-free RL)

**Implementation:** `src/rl/train_sac.py`, `src/rl/env.py`

**Key Components:**

- Environment: `FloquetCoolingEnv` (Gymnasium-compatible)
- Agent: SAC from Stable-Baselines3
- Reward: $R = -\langle n \rangle$ (negative occupation)

**Hyperparameters:**

| Parameter              | Typical Value    |
| ---------------------- | ---------------- |
| `total_timesteps`      | 50,000 - 300,000 |
| `n_steps_per_cycle`    | 20               |
| `n_cycles_per_episode` | 50               |

**Pros:** Explores global space, discovers bang-bang strategies
**Cons:** Slower, noisier results

---

## 5. Codebase Structure

```
src/
├── physics/           # Core quantum mechanics
│   ├── operators.py   # a, a†, σ±, σz operators (JAX)
│   ├── hamiltonian.py # Time-dependent H(t) construction
│   ├── lindblad.py    # Lindblad superoperator L[ρ]
│   └── initial_states.py # Thermal states, ground states
│
├── floquet/           # Floquet cycle engine
│   ├── cycle.py       # JIT-compiled cycle via lax.scan
│   └── __init__.py    # Exports FloquetCycleParams, find_floquet_steady_state
│
├── baseline/          # Stochastic limit reference
│   └── stochastic.py  # Analytic n* formula (Eq. 25)
│
├── optimization/      # GRAPE optimizer
│   └── grape.py       # GRAPEConfig, run_grape_optimization
│
├── rl/                # Reinforcement learning
│   ├── env.py         # FloquetCoolingEnv (Gymnasium)
│   └── train_sac.py   # SAC training script
│
├── validation/        # Physics validation
│   ├── tier2.py       # RWA vs full TDSE comparison
│   ├── tier3.py       # Noise robustness (1/f flux, TLS)
│   └── no_go_theorem.py # Commuting vs non-commuting test
│
├── analysis/          # Results and figures
│   ├── figures.py     # PRL publication figures (3)
│   ├── presentation_figures.py # Comprehensive figures (12)
│   └── thermodynamics.py # COP, cooling power, entropy
│
└── utils/             # Utilities
    └── io.py          # Save/load GRAPE and SAC results
```

---

## 6. Running the Code

### 6.1 Installation

```bash
# Clone repository
git clone https://github.com/sattary/abouie_proj.git
cd abouie_proj

# Create environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install jax jaxlib flax optax matplotlib stable-baselines3 gymnasium scipy seaborn
```

### 6.2 Quick Verification

```bash
# Run No-Go theorem verification
uv run python -m src.validation.no_go_theorem

# Expected output:
# Commuting case: n ~ 1.12 (cannot beat stochastic limit)
# Non-commuting case: n ~ 1.04 (beats it!)
```

### 6.3 Generate PRL Figures

```bash
uv run python -m src.analysis.figures
# Outputs: results/figures/fig1_comparison.png, fig2_waveforms.png, fig3_nogo_sweep.png
```

### 6.4 Generate Presentation Figures

```bash
uv run python -m src.analysis.presentation_figures
# Outputs: results/presentation/fig01-12*.png (12 comprehensive figures)
```

### 6.5 GPU Training (Google Colab)

Open `notebooks/colab_gpu.ipynb` in Google Colab:

1. Enable GPU runtime: Runtime → Change runtime type → GPU
2. Run all cells to train GRAPE and SAC on GPU
3. Download results from the `results/` directory

---

## 7. Validation Tiers

### 7.1 Tier 1: Basic Physics

**Location:** `tests/test_physics.py`

- Operator algebra: $[a, a^\dagger] = 1$
- State normalization: $\text{Tr}(\rho) = 1$
- Hermiticity: $\rho = \rho^\dagger$
- Thermal occupation formula

### 7.2 Tier 2: RWA Validity

**Location:** `src/validation/tier2.py`

Compares Rotating-Wave Approximation (RWA) Hamiltonian to full time-dependent Schrödinger equation (TDSE) with counter-rotating terms.

**Key Result:** RWA breaks down for $g/\omega_c > 0.1$ (~53% discrepancy)

### 7.3 Tier 3: Noise Robustness

**Location:** `src/validation/tier3.py`

Tests robustness to realistic noise sources:

- **1/f flux noise:** Detuning fluctuations
- **TLS defects:** Random two-level system coupling
- **Reset errors:** Imperfect qubit initialization

**Key Result:** Protocol survives up to 5% flux noise amplitude

---

## 8. Presentation Figures

12 comprehensive figures for explaining the project (in `results/presentation/`):

| #   | Filename                          | Content                     |
| --- | --------------------------------- | --------------------------- |
| 1   | `fig01_system_schematic.png`      | Physical setup cartoon      |
| 2   | `fig02_thermal_occupation.png`    | $\bar{n}$ vs temperature    |
| 3   | `fig03_hamiltonian_structure.png` | Jaynes-Cummings matrix      |
| 4   | `fig04_lindblad_dynamics.png`     | Approach to steady state    |
| 5   | `fig05_stochastic_limit.png`      | $n^*$ vs parameters         |
| 6   | `fig06_floquet_concept.png`       | Periodic control concept    |
| 7   | `fig07_grape_convergence.png`     | GRAPE learning curve        |
| 8   | `fig08_optimal_pulses.png`        | Optimal $g(t)$, $\Delta(t)$ |
| 9   | `fig09_comparison_bar.png`        | Method comparison           |
| 10  | `fig10_nogo_theorem.png`          | Commuting vs non-commuting  |
| 11  | `fig11_noise_robustness.png`      | Flux noise sweep            |
| 12  | `fig12_summary_table.png`         | Results summary table       |

---

## 9. Results Summary

### 9.1 Main Results

| Method               | $\langle n \rangle$ | Improvement   | Strategy                  |
| -------------------- | ------------------- | ------------- | ------------------------- |
| **Stochastic Limit** | 1.44                | 0% (baseline) | Static parameters         |
| **GRAPE**            | ~1.02               | **29%**       | Gradient-optimized pulses |
| **SAC (RL)**         | ~0.94               | **35%**       | Policy-learned bang-bang  |

### 9.2 Physical Interpretation

1. **Why does Floquet work?** The non-commuting dynamics create an effective "ratchet" that pumps energy preferentially in one direction.

2. **Why does commuting fail?** Static control preserves detailed balance — the system reaches the same equilibrium regardless of protocol.

3. **What limits performance?** Cavity decay rate $\kappa$ and qubit thermalization $\gamma_1$ provide lower bounds.

---

## 10. References

1. **Vashaee-Abouie Paper:** Original stochastic cooling limit derivation

   - Equation 25 defines the asymptotic limit $n^*$

2. **GRAPE Algorithm:**

   - Khaneja et al., "Optimal control of coupled spin dynamics" (2005)

3. **Soft Actor-Critic:**

   - Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)

4. **Floquet Engineering:**

   - Eckardt, "Colloquium: Atomic quantum gases in periodically driven optical lattices" (2017)

5. **Lindblad Master Equation:**
   - Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)

---

**Conclusion:** This project successfully demonstrates that Floquet engineering enables cooling well below the fundamental thermodynamic limits of static machines. The code provides a complete framework for exploring coherent control in open quantum systems.
