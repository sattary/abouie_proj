Circumventing the Coarse-Grained Limit:
Floquet-Engineered Cooling Cycles for
Superconducting Cavities

Research Proposal based on Vashaee & Abouie (2025)

January 3, 2026

Abstract

Passive quantum refrigeration using stochastic streams of correlated ancillas is con-
strained by a fundamental thermodynamic bound derived from the coarse-grained detailed
balance of the interaction [Vashaee & Abouie, arXiv:2512.06996]. We propose to circum-
vent this “stochastic limit” by replacing the random Poissonian stream with numerically
discovered, coherent Floquet cooling cycles. By transitioning to a structured, peri-
odic interaction protocol, we exploit coherent interference effects—specifically non-vanishing
commutators in the Floquet-Magnus expansion—to selectively suppress heating transitions
while enhancing cooling pathways. We outline a rigorous three-tier validation method using
Deep Reinforcement Learning (DRL) benchmarked against Gradient Ascent Pulse Engineer-
ing (GRAPE), incorporating realistic noise models (TLS defects, 1/f noise, finite T1/T2).
We aim to demonstrate that these “digital” cycles can achieve steady-state temperatures sig-
nificantly lower than the analytic limits of the stochastic model while maintaining a favorable
coefficient of performance (COP) within the 1 K stage energy budget.

1 Motivation: The Stochastic Bottleneck

The Vashaee-Abouie model establishes a framework for cooling a microwave cavity using a stream
of correlated qubit pairs. However, the cooling limit is bounded by the coarse-grained Lind-
blad generator derived from the random Poissonian arrival of pairs. The effective steady-state
generator is given by [1]:

Lstreamρ ∝ Rϕ2 (cid:104)

2 D[a]ρ + r(2)
r(2)

(cid:105)
1 D[a†]ρ

(1)

) to heating (r(2)
where the ratio of cooling (r(2)
) rates is fixed by the thermal and correlated
1
2
state of the ancilla pair [1]. This stochastic approach effectively “washes out” phase information,
treating the interaction as a purely incoherent thermodynamic resource.

The Opportunity: In modern circuit QED (cQED), interactions are triggered by high-
precision clocks. We propose to replace the stochastic stream with a periodic (Floquet) unitary
sequence Ucycle(t). By optimizing the timing, detuning ∆(t), and coupling g(t), we can engineer
an effective Hamiltonian where heating matrix elements interfere destructively.

2 Theoretical Framework: Floquet-Magnus Expansion

To demonstrate the physical mechanism for circumventing the stochastic limit, we consider
a periodic protocol of duration Tcycle. The evolution is governed by the stroboscopic map
. Using the Floquet-Magnus expansion, the effective generator
Ecycle = T exp

L(t)dt

(cid:17)

(cid:16)(cid:82) Tcycle
0

1

Lef f approximates to:

Lef f ≈

1
Tcycle

(cid:90) Tcycle

0

L(t)dt +

1
2Tcycle

(cid:90) Tcycle

(cid:90) t1

0

0

[L(t1), L(t2)] dt2 dt1 + . . .

(2)

In the stochastic limit, the commutator terms average to zero. In a coherent cycle, we engineer
[L(t1), L(t2)] ̸= 0 to renormalize the transition rates.

Hypothesis 1. A periodic protocol beats the stochastic limit if and only if the interaction Hamil-
tonian is non-commuting at different times within the cycle ([Hint(t1), Hint(t2)] ̸= 0), allowing
for a coherent violation of the effective detailed balance.

3 Methodology: Three-Tier Simulation & Optimization

We employ a “Physics-Informed AI” approach, benchmarking Soft Actor-Critic (SAC) against
standard optimal control (GRAPE). Validation occurs in three tiers of increasing rigor.

3.1 Tier 1: Fast Stroboscopic Map (Discovery)

• Physics: Discrete Kraus maps representing instantaneous unitary kicks + free evolution.

• Purpose: Rapid exploration of cycle period T and number of steps N .

• Benchmark: RL results compared against random search and gradient-descent baselines.

3.2 Tier 2: Full Time-Dependent Unitary (Physics Validation)

• Physics: Integration of the time-dependent Schrödinger equation (TDSE).

• Critical Check: Includes counter-rotating terms and non-secular effects ignored in
the RWA limit of the base paper [1]. This ensures the cooling advantage is physical and
not an artifact of RWA breakdown.

• Controls: Continuous modulation of coupling g(t) and detuning ∆(t).

3.3 Tier 3: The “Dirty” Model (Robustness)

• Physics: Full open-system dynamics with realistic noise defined in Table 2 of [1].

• Noise Sources: Finite qubit T1/T2; TLS spectral defects (detuning holes); Reset errors

(1 − 3% residual population).

• Goal: Prove robustness against ±10% parameter drift.

4 Thermodynamic Viability: The Energy Budget

To address thermodynamic concerns, we explicitly model the Coefficient of Performance
(COP): η = ˙Qcool/Pinput.

• Cooling Power:

˙Qcool = ℏωcavΓef f

↓

⟨n⟩.

• Input Power: Pinput = Pctrl + Preset.

• Constraint: Pinput ≪ P1K_stage ≈ 100–500 mW [1].

We will demonstrate that the control pulses (integrated Ω2(t)) add negligible heat load compared
to the 1 K stage capacity.

2

5 Proposed Results

1. Circumventing the Limit: A comparison plot of Tcav vs. Detuning ∆.

• Baseline: Analytic Poisson limit (Eq. 37 in [1]).
• Floquet: AI-discovered protocol. (Must show TF loquet < TStochastic).

2. The Recipe: Explicit waveform diagrams {g(t), ∆(t)} for experimental AWG implemen-

tation.

3. Thermodynamic Cost: A plot of Entropy Efficiency (η) vs. Cooling Rate, proving the

protocol is thermodynamically sound.

Extension: New Ideas & Implementation Candidates

Based on recent critiques, we identify the following high-value extensions that are feasible within
the current simulation framework:

1. The “No-Go” Theorem Verification: We will numerically verify that if the cycle Hamil-
tonian commutes with itself at all times ([H(t), H(t′)] = 0), the RL agent cannot beat the
stochastic limit. This provides a negative control case that strengthens the physics claim.

2. Adaptive Detuning (∆(t)): Instead of fixed detuning, the control scheme will allow dy-
namic modulation of the qubit frequency. This creates “interference bands” in the frequency
domain, akin to Floquet topological insulators, potentially blocking phonon absorption
channels entirely.

3. The Entropy-per-Joule Metric: We will introduce a specific figure of merit: ηS =
∆Scav/Wpulse, quantifying the bits of entropy removed per Joule of control work. This
elevates the work from “engineering” to “quantum thermodynamics.”

4. Hybrid Semi-Feedback Loop: Implementation of a “weak-measurement” step at the end
of every N cycles. This allows the protocol to switch between two distinct Floquet cycles
based on a coarse readout, bridging the gap between open-loop control and full feedback
cooling.

References

[1] D. Vashaee and J. Abouie, “Quantum Correlation Assisted Cooling of Microwave Cavities

Below the Ambient Temperature,” arXiv:2512.06996 (2025).

3

