Quantum Correlation-Assisted Cooling of Microwave Cavities Below the Ambient Temperature

Daryoosh Vashaeea,b* and Jahanfar Abouiec
a Department of Materials Science & Engineering, North Carolina State University, Raleigh, NC 27606,
USA
b Department of Electrical & Computer Engineering, North Carolina State University, Raleigh, NC 27606,
USA
c Department of Physics, Institute for Advanced Studies in Basic Sciences, Zanjan 45137-66731, Iran

Abstract

 We develop a theoretical framework for cooling a microwave cavity mode using a Poisson stream of
internally  correlated  pairs  of  two-level  systems  and  analyze  its  performance  under  realistic  dissipation.
Starting from a Lindblad model of a phonon-tethered cavity interacting with sequentially injected atom pairs,
we derive closed-form expressions for the steady-state cavity occupation and effective temperature. Two
coupling geometries are examined: a one-atom configuration, where only one member of each pair interacts
with the cavity, and a two-atom configuration, where both atoms couple collectively. The single-atom model
enables cooling below the  phonon  bath  but  not  below the reservoir temperature, whereas the two-atom
scheme  exhibits  enhanced  refrigeration  -  pair  correlations  modify  the  cavity's  upward  and  downward
transition rates so that the steady-state temperature can fall well below that of the reservoir for weak phonon
damping.  We  map  the  parameter  space  including  detuning,  coupling  strength,  damping,  and  intra-pair
exchange, identifying cooling valleys near resonance and the crossover between reservoir- and phonon-
dominated  regimes.  The  two-atom  configuration  thus  realizes  a  genuine  quantum-enhanced  cooling
mechanism absent in the single-atom case. We further outline an experimental implementation using two
superconducting qubits repeatedly prepared, coupled, and reset inside a 3D cavity. Realistic reset and flux-
tuning  protocols  support  MHz-rate  interaction  cycles,  enabling  engineered  reservoirs  to  impose  cavity
temperatures of 50-120 mK even when the cryostat is at ~1 K, offering a pathway to autonomous, on-chip
refrigeration of microwave modes in scalable quantum hardware.

Keywords: Quantum refrigeration, Circuit QED, Superconducting qubits, Collision models, Quantum

thermodynamics

Introduction

The  rapid  growth  of  quantum  information  science  has  placed  increasing  demands  on  cryogenic
infrastructure for hosting superconducting qubits, microwave resonators, semiconductor spin qubits, and
associated control electronics. A central systems-level challenge is the drastic decrease in available cooling
power  as  one  descends  from  kelvin  to  millikelvin  temperatures  in  a  dilution  refrigerator.  For  example,
commercial systems typically provide on the order of 10-20 μW at the base (≈  20 mK) plate, while sub-
milliwatt levels are available around 100 mK, and hundreds of milliwatts can be supplied near 1 K.1,2,3,4,5,6,7
This  multi-order-of-magnitude  disparity forces nearly  all quantum devices  onto the coldest stage, where
thermal budgets, wiring heat loads, and limited physical space become primary obstacles to scaling.8,9

A promising route to alleviating this bottleneck is to relocate as much classical and passive hardware
as  possible  to  higher-temperature  stages  (1-4  K),  while  using  engineered  quantum  reservoirs  to  pull
selected  degrees  of  freedom,  specific  cavity  modes,  qubits,  or  mesoscopic  devices,  to  effective
temperatures  far  below  the  ambient  of  their  hosting  stage.  This  idea  is  firmly  rooted  in  quantum
thermodynamics and open quantum systems: collision models, micromaser physics, and quantum thermal
machines all show how repeated interactions with small ancilla systems can impose a controlled detailed
balance  on a  target mode  and drive it toward  a  nontrivial steady state.10,11,12,13,14 In  particular, few-body

5
2
0
2
c
e
D
2
1

]
h
p
-
t
n
a
u
q
[

6
9
9
6
0
.
2
1
5
2
:
v
i
X
r
a

* dvashae@ncsu.edu

1

quantum  refrigerators  and  absorption  machines  have  been  shown  theoretically  to  cool  a  target  system
using only finite-size working media.15,16,17

A particularly relevant example in this context is the “photo-Carnot” engine of Dillenschneider and Lutz,
where  a stream of correlated two-level atoms  interacts with  a single cavity  mode and shifts its  effective
temperature  through  correlation-modified  transition  rates.10  In  that  model,  the  cavity  acts  as  a  harmonic
working medium, while the atomic pairs form a generalized thermal reservoir whose level populations and
coherences determine the net heat flow. Under appropriate conditions, the cavity can attain an effective
temperature  lower  than  that  associated  with  the  atomic  populations  alone,  illustrating  how  quantum
correlations can be harnessed as a thermodynamic resource. Closely related micromaser models, in which
individual  atoms  or  atom  pairs  pass  repeatedly  through  a  cavity,  have  been  extensively  developed  and
provide a microscopic foundation for such ancilla-based refrigerators.14,18,19

However,  the  original  photo-Carnot  formulation  and  related  micromaser  models  typically  neglect
coupling  to  a  realistic  phonon  bath  and  assume  an  idealized  atomic  beam,  which  complicates  direct
implementation in microwave hardware. At the same time, circuit quantum electrodynamics (cQED) has
matured  into  a  versatile  platform  for  engineering  light–matter  interactions  between  high-Q  microwave
cavities and superconducting qubits.8,9,20,21,22 In cQED, it is straightforward to realize strong or ultrastrong
light–matter coupling, to implement tunable inter-qubit exchange, and to exploit well-developed techniques
for state preparation, readout, and dissipation engineering. These capabilities suggest that the role of the
atomic beam in the photo-Carnot setup can be played by a small register of superconducting qubits that
are repeatedly prepared, interact with the cavity, and are then reset.

In this work we develop and analyze such an architecture. We replace the idealized atomic stream of
Ref. [10] with a pair of superconducting qubits that serve as a resettable, correlated reservoir. The qubits
repeatedly interact with a high-Q microwave cavity mode and are then actively reset using experimentally
demonstrated  protocols  such  as  Purcell-enhanced  dissipation,  measurement-and-feedback  reset,  or
autonomous  reservoir  engineering.23,24,25,26,27,28,29,30  The  cavity  is  additionally  coupled  to  a  phonon  (or
thermal) bath at temperature 𝑇bath, which we take to lie around 1 K to reflect realistic high-power stages in
modern cryogenic systems.1,2,3,4,5,6  Within this framework, the  qubit  pair acts as  a sacrificial low-entropy
working medium, while the cavity is the object to be cooled; the resulting dynamics are naturally described
by a collision model with tunable arrival rate, interaction time, detuning, and qubit–qubit correlations.12

We  first  extend  the  energetics-of-correlations  model  of  Dillenschneider  and  Lutz  [10]  by  explicitly
including cavity-bath coupling and by distinguishing two interaction scenarios: (i) one-subsystem coupling,
in which only one  atom of the pair couples to the cavity, and (ii) two-subsystem coupling, in which both
atoms interact with the cavity field. Using standard  input–output and Lindblad techniques,11,12 we derive
closed-form birth-death equations for the cavity photon number that incorporate both the engineered two-
qubit  reservoir  and  the  phonon  bath,  and  we  obtain  analytic  expressions  for  the  steady-state  cavity
occupation number and effective temperature. We show how the detailed-balance ratios governing upward
and downward transitions depend on the pair’s level populations and, when both atoms couple, their mutual
coherence; on the strength and duration of each atom–cavity interaction; on atom-cavity detuning; on the
arrival rate of pairs; and on the cavity’s damping to its phonon bath, thereby generalizing the photo-Carnot
temperature shift to a realistic, phonon-tethered setting.

We  then  evaluate  the  resulting  analytic  steady-state  expressions  over  experimentally  relevant
parameter ranges, focusing on regimes compatible with existing cQED implementations. In particular, we
explore how the cavity temperature 𝑇cav depends on the atom-cavity coupling 𝑔, the inter-qubit exchange
coupling 𝜆, the atoms and cavity frequencies detuning Δ, the arrival rate 𝑅, and the cavity-bath damping
rate 𝜅. We identify broad regions where the engineered reservoir dominates over the 1 K bath and drives
the  cavity  to  effective  temperatures  in  the  50  –  20    mK  range,  even  though  the  qubit  reservoir  itself  is
maintained at an effective temperature 𝑇atom ≈ 30 – 80 mK by fast reset. In the two-subsystem case, we
find  that  correlations  induced  by  intra-atoms  interaction  can  yield  additional  cooling  relative  to  the  one-
subsystem scenario, particularly near resonance, in line with the qualitative predictions of Ref. [10] but now
in a fully open-system framework.

Finally, we present a detailed experimental mapping of the proposed refrigerator onto contemporary
cQED hardware. We discuss how two transmons or flux-type qubits coupled to a 3D superconducting cavity

2

can  implement  the  required  prepare-interact-reset  cycles  at  MHz  rates,  making  use  of  established
techniques for rapid frequency tuning,28,47 Purcell-engineered and parametric reset,26,27,28,29,31 and driven-
dissipative state stabilization.25,32,33,34 We then outline how the cooled cavity mode can in turn be used to
refrigerate a target system, such as a gate-defined double quantum dot coupled to the cavity field35,36 or a
secondary cavity mode used as a bosonic memory.34,37,38,39 In this way, the two-qubit refrigerator operating
at a 1 K stage can create a local sub–100-mK environment for selected degrees of freedom without relying
on the limited cooling power of the dilution refrigerator’s base plate.

The  main  contributions  of  this  paper  are  thus:  (i)  a  theoretical  extension  of  the  energetics-of-
correlations model [10] to include realistic phonon baths and both one- and two-subsystem cavity couplings;
(ii) analytic derivation and systematic evaluation of the resulting steady-state cavity temperatures across
experimentally relevant parameter regimes; and (iii) a concrete experimental blueprint for realizing a two-
qubit photo-Carnot refrigerator in cQED, capable of cooling a cavity mode and, via that mode, a mesoscopic
device at a 1 K stage. Together, these results illustrate how quantum correlations  and engineered reset
protocols can be harnessed to mitigate cryogenic bottlenecks in scalable quantum architectures.

To streamline the presentation in the sections that follow, Table 1 collects all symbols, operators, and

rates used in the analysis of the one-atom and two-atom interaction models.

Table 1: Symbols and operators used in the manuscript.

Symbol / Operator
Fundamental constants and
thermodynamic quantities
ℏ
𝑘B

𝑇atom

𝑇bath

𝑇cav

𝛽

𝑛̅1

Frequencies, detuning,
Hamiltonians
𝜔1
𝜔
𝜆
Δ

𝐻

𝐻c

𝐻int

Operators for cavity and atoms
𝑎, 𝑎†

±
±, 𝜎𝐵
𝜎𝐴

𝑆±

𝜌

Definition

Reduced Planck constant.
Boltzmann constant.
Temperature used to prepare each correlated two-atom pair in the
stream.
Ambient phonon-bath temperature (substrate temperature).
Effective  cavity  temperature  defined  from  Bose  inversion  of  the
steady-state photon number (𝑛∗)
Inverse temperature, 𝛽 = 1/(𝑘B𝑇atom)
Thermal Bose occupation of the phonon bath at 𝜔1:
𝑛ˉ 1 = 1/(exp [ℏ𝜔1/(𝑘B𝑇bath)] − 1).

Cavity angular frequency (single microwave mode).
Single-atom level spacing in the correlated pair.
Intra-pair exchange coupling between the two atoms.
Atom–cavity detuning: (Δ = 𝜔 − 𝜔1).
Two-atom pair Hamiltonian:
+𝜎𝐵
𝑧) + ℏλ(𝜎𝐴
𝑧 + 𝜎𝐵
𝐻 = ℏω(𝜎𝐴
Free cavity Hamiltonian: (𝐻𝑐 = ℏ𝜔1𝑎†𝑎).
(𝐻int = ℏ𝑔(𝑎𝜎+ + 𝑎†𝜎−)),  with
Jaynes–Cummings
appropriate generalization for two atoms. In general the couplings
may differ (𝑔𝐴, 𝑔𝐵), but here we assume 𝑔𝐴 = 𝑔𝐵 ≡ 𝑔.

interaction:

− + 𝜎𝐴

−𝜎𝐵

+).

Cavity annihilation and creation operators.
Ladder  operators,  respectively,  for  atom  A  and  atom  B  in  a
correlated pair.

Collective two-atom ladder operators: (𝑆± = 𝜎𝐴
Density operator for cavity+atomic pair.

± + 𝜎𝐵

±).

3

𝒟[𝐿]

Lindblad dissipator: (𝒟[𝐿]𝜌 = 𝐿𝜌𝐿† −

{𝐿†𝐿, 𝜌}).

1

2

Tr[⋅]
⟨⋅⟩
Cavity photon numbers and
steady-state quantities
𝑛 ≡ ⟨𝑎†𝑎⟩
𝑛∗
Thermal weights and stream
coefficients

Trace operation.
Expectation value with respect to (𝜌).

Cavity photon number.
Steady-state cavity photon number.

𝑍

𝜌𝑒, 𝜌𝑔

𝜌𝑑, 𝜌𝑛𝑑

𝑟1, 𝑟2

(2), 𝑟2
𝑟1

(2)

Collision / stream parameters
𝑔
𝜏
𝜙 ≡ 𝑔𝜏
2 ≡ 𝜒 𝜙2

𝜙2

𝜒 ∈ [1,2]

𝑅
Dissipative rates and cavity
cooling parameters
𝜅

𝜅(±)

Γ↓, 𝐽↑

(2), 𝐽↑
Γ↓

(2)

𝐴↑, 𝐴↓
Collision maps and Liouvillians

Partition function for the correlated pair:
𝑍 = 2[cosh (𝛽ℏ𝜔) + cosh (𝛽ℏ𝜆)].
Thermal weights of the single-excitation manifolds:
𝜌𝑒 = 𝑒−𝛽ℏ𝜔/𝑍,   𝜌𝑔 = 𝑒𝛽ℏ𝜔/𝑍
Diagonal and non-diagonal (coherent) pair weights:
𝜌𝑑 = cosh (𝛽ℏ𝜆)/𝑍,   𝜌𝑛𝑑 = −sinh (𝛽ℏ𝜆)/𝑍
One-subsystem  stream  coefficients  (only  one  atom  couples  to
cavity photon): 𝑟1 = 𝜌𝑒 + 𝜌𝑑, 𝑟2 = 𝜌𝑔 + 𝜌𝑑. Satisfy 𝑟1 + 𝑟2 = 1
Two-subsystem  stream  coefficients  (both  atoms  couple  to  cavity
photon): 𝑟1
Satisfy, 𝑟1

(2) = 𝜌𝑒 + 𝜌𝑑 + 𝜌𝑛𝑑, 𝑟2
(2) = 1 + 2𝜌𝑛𝑑.
(2) + 𝑟2

(2) = 𝜌𝑔 + 𝜌𝑑 + 𝜌𝑛𝑑.

Jaynes–Cummings coupling strength between cavity and atom(s).
Interaction (passage) time per collision.
Per-collision interaction angle (assumed (𝜑 ≪ 1).
Effective two-atom coupling strength per collision.
Coherent  enhancement  factor  for  two-atom  coupling;  (𝜒 = 2)  for
fully symmetric coupling.
Stream flux (Poisson arrival rate of correlated pairs).

Cavity-phonon energy-damping rate.
Thermal Lindblad rates:
 𝜅(−) = 𝜅(𝑛ˉ 1 + 1), 𝜅(+) = 𝜅𝑛ˉ 1
One-subsystem cavity “down/up” coefficients appearing in
 𝑛̇ = −Γ↓𝑛 + 𝐽↑:    Γ↓ = 𝜅 + 𝑅(𝑟2 − 𝑟1)𝜙2, 𝐽↑ = 𝜅𝑛ˉ 1 + 𝑅𝑟1𝜙2
Two-subsystem analogs with 𝑟1,2
(2) − 𝑟1
 Γ↓
“Birth-death” up/down rates for 𝑛1:
(2)𝜙2
 𝐴↑ = 𝜅𝑛ˉ 1 + 𝑅𝑟1

(2)and 𝜙2
2:
(2) = 𝜅𝑛ˉ 1 + 𝑅𝑟1

2, 𝐴↓ = 𝜅(𝑛ˉ 1 + 1) + 𝑅𝑟2

(2) = 𝜅 + 𝑅(𝑟2

(2))𝜙2

(2)𝜙2
2

(2)𝜙2
2

2;  𝐽↑

𝑈

Φ(𝜌)

ℒstream

ℒbath
Notation conventions

Single-collision unitary:
 𝑈 = exp [−(𝑖/ℏ)𝐻int𝜏] ≃ 𝕀 − (𝑖/ℏ)𝐻int𝜏 −
Single-collision map: Φ(𝜌) = Trpair[𝑈(𝜌 ⊗ 𝜌pair)𝑈†]
Coarse-grained stream Liouvillian (one-subsystem):
 𝑅𝜙2[𝑟2𝒟[𝑎] + 𝑟1𝒟[𝑎†]]
Thermal phonon Liouvillian on cavity: 𝜅(−)𝒟[𝑎] + 𝜅(+)𝒟[𝑎†]

2 𝜏2 + ⋯

2ℏ2 𝐻int

1

4

( · )*

Denotes steady-state value (e.g., 𝑛∗).

1.  Correlated atom-pair stream: only one atom coupled to a phonon-tethered cavity

1.1  System and parameters

We  consider  a  single  bosonic  cavity  mode  𝑎  of  frequency  𝜔1,  described  by  the  Hamiltonian  𝐻c =
ℏ𝜔1 𝑎†𝑎. The cavity exchanges energy with an ambient phonon bath at temperature 𝑇bath, which we model
phenomenologically  as  a  thermal  Lindblad  contact  resonant  at  𝜔1  with  overall  coupling  rate  𝜅.  The
corresponding Bose-Einstein occupation at the cavity frequency is

𝑛ˉ 1 =

1
exp (ℏ𝜔1/𝑘B𝑇bath) − 1

.

(1)

The cavity is also repeatedly and weakly “kicked” by a stream of correlated two two-level atoms that
act as an engineered atomic reservoir. Each incoming atom pair is prepared in the thermal state associated
with a XY  Heisenberg Hamiltonian, characterized by a  local splitting  ℏ𝜔 and an exchange coupling  ℏ𝜆.
Writing 𝛽 ≡ 1/(𝑘B𝑇atom), the state weights in the product basis {|𝑒𝑒⟩, |𝑒𝑔⟩, |𝑔𝑒⟩, |𝑔𝑔⟩} are:10,18

𝜌𝑒 =

𝑒−𝛽ℏ𝜔
𝑍

, 𝜌𝑔 =

𝑒+𝛽ℏ𝜔
𝑍

, 𝜌𝑑 =

cosh (𝛽ℏ𝜆)
𝑍

, 𝜌𝑛𝑑 = −

sinh (𝛽ℏ𝜆)
𝑍

,

(2)

with the partition function

𝑍 = 2[cosh (𝛽ℏ𝜔) + cosh (𝛽ℏ𝜆)].

(3)

During a short passage of duration 𝜏, only atom 𝐴 of each atom pair couples (near-resonantly) to the

cavity through a Jaynes-Cummings interaction,

𝐻int = ℏ𝑔(𝑎 𝜎𝐴

+ + 𝑎†𝜎𝐴

−),

(4)

where 𝜎𝐴

± act on atom A. We work in the weak, short-collision (small-angle) limit:

𝜙 ≡ 𝑔𝜏 ≪ 1.

(5)

The arrivals of independently prepared atom pairs are assumed to follow a Poisson process with rate

𝑅, ensuring memoryless, uncorrelated collision events on the coarse-grained timescale.

Because  only  atom  A  couples,  the  local  excited  state  and  ground  state  populations  relevant  to  the
cavity dynamics are the marginals of atom A. Evaluating the reduced state of A from Eqs. (2)-(3) yields the
effective stream coefficients

𝑟1 = 𝜌𝑒 + 𝜌𝑑,

𝑟2 = 𝜌𝑔 + 𝜌𝑑,

𝑟1 + 𝑟2 = 1

(6)

Explicitly,

𝑟2 − 𝑟1 =

𝑒𝛽ℏ𝜔 − 𝑒−𝛽ℏ𝜔
𝑍

=

2 sinh(𝛽ℏ𝜔)
𝑍

,

(7)

so that correlations (finite 𝜆 increasing 𝑍) reduce the effective asymmetry 𝑟2 − 𝑟1 at fixed 𝜔, 𝑇atom.

1.2  Detuning and the spectral-overlap filter

We now allow a finite detuning between atom (𝜔) and the cavity (𝜔1),

Δ ≡ 𝜔 − 𝜔1.

(8)

5

In a single short collision of duration 𝜏, the atom-cavity energy exchange (𝛿𝐸) arises from the overlap of
two  time/frequency  windows:  (i)  the  cavity  field  correlation,  which  decays  as  〈𝑎(𝑡)𝑎†(𝑠)〉env ∝
exp[−(𝜅/2)|𝑡 − 𝑠|]  due  to  its  ambient  loss  channel,  and  (ii)  the  finite-time  envelope  of  the  atom-cavity
interaction, which is nonzero only for 0 ≤ 𝑡, 𝑠 ≤ 𝜏 and therefore has a Fourier width set by 1/𝜏. This can be
seen  directly  by  evaluating  the  second-order  term  in  the  single-collision  map  (𝛿𝐸 ∝ 𝑔2Re{𝐼(∆)}),  which
contains the kernel:11,12,19

𝜏

𝐼(Δ) = ∫ 𝑑𝑡

0

𝜏
∫ 𝑒−(
0

𝜅
2

) ∣𝑡−𝑠∣ 𝑒𝑖Δ(𝑡−𝑠) 𝑑𝑠

𝜏
   =   2 Re ∫ (𝜏 − 𝑢) 𝑒−(
0

𝜅
2

−𝑖Δ)𝑢 𝑑𝑢,

    (9)

The integral has the closed form:

𝐼(Δ) = 2 Re {

𝜏
𝛼

(1 − 𝑒−𝛼𝜏) −

1
𝛼2 (1 − 𝑒−𝛼𝜏(1 + 𝛼𝜏))} ,

𝛼 ≡

𝜅
2

− 𝑖Δ.    (10)

Two distinct limits are instructive. When 𝜅𝜏 ≫ 1 (the collision is long compared to the cavity correlation

time 2/𝜅; equivalently, the overlap is cavity-limited), the exponential terms vanish and:

𝐼(Δ) ≈ 2 [

𝜏 (𝜅/2)
(𝜅/2)2 + Δ2 −

(𝜅/2)2 − Δ2
((𝜅/2)2 + Δ2)2]     (11)

For 𝜅𝜏 ≫ 1 the 𝜏-independent piece is negligible compared to the 𝜏 term, so the leading contribution is:

𝐼(Δ) ≈

𝜅𝜏

(𝜅/2)2 + Δ2     (12)

i.e.,  a Lorentzian  in  Δ with half width at  half maximum  (HWHM)  𝜅/2. This regime corresponds to a

relatively low quality factor 𝑄 = 𝜔1/𝜅.

Conversely, when 𝜅𝜏 ≪ 1 (very short collision), the cavity remains nearly constant over the window,

and:

sin 2(Δ𝜏/2)
Δ
(
2
whose main lobe has a characteristic width of order 1/𝜏.

𝐼(Δ) ≈  2 ∫ (𝜏 − 𝑢)cos (Δ𝑢) 𝑑𝑢

)2

=

0

𝜏

= 𝜏2 sinc2 (

Δ𝜏
2

)     (13)

Normalizing  by  𝐼(∆= 0)  to  define  the  spectral  weight  actually  entering  the  collision  rate,  𝐿(Δ) ≡
𝐼(Δ)/𝐼(0),  the  exact  lineshape  interpolates  smoothly  between  a  Lorentzian  (cavity-limited)  and  a  sinc2
(time-window-limited) profile. In the coarse-grained (Markov) generator we retain only the slowly varying
secular component; a standard and accurate interpolation is to replace the bare per-collision strength 𝜙2 =
(𝑔𝜏)2 by a Lorentzian detuning filter 𝐿(Δ) whose width is the sum of the two limiting half-widths, namely the
cavity  homogeneous  width  and  the  finite-time  Fourier  width.  With  this  replacement  𝜙2 → 𝜙2𝐿(Δ),  the
detuning simply rescales both the source and slope terms generated by the stream.

Specifically, in a short, weak collision of duration 𝜏, the stream-cavity energy exchange is governed by
the overlap of (i) the cavity’s Lorentzian line, whose spectral full width at half maximum (FWHM) equals its
energy-decay  rate  𝜅  (since  the  field  correlation  decays  as  exp[−(𝜅/2)|𝑡 − 𝑠|]),  and  (ii)  the  finite-time
“window” of the interaction, whose spectral envelope has a main-lobe width of order 1/𝜏 (from the Fourier
transform  of  a  length-𝜏  time  gate).  Evaluating  the  second-order  kernel  𝐼(Δ)  yields  a  closed  form  that
interpolates between a cavity-limited Lorentzian of width 𝜅 when 𝜅𝜏 ≫ 1 (Eq. 12) and a time-window-limited
sinc²  profile  of  width  ∼ 1/𝜏  when  𝜅𝜏 ≪ 1  (Eq.  13).  In  the  coarse-grained  generator  we  represent  this
interpolation by a single Lorentzian “overlap” factor,13

𝐿(Δ) =

1
⁄
1 + (2Δ 𝛤over

)2                (14)

6

with an effective overlap linewidth taken as the sum of the two limiting widths, 𝛤over ≡ 𝜅  +  1/𝜏. This is
essentially a detuning filter which replaces the bare 𝜙2. The choice  Γover = 𝜅 + 1/𝜏 captures the correct
limiting behaviors: for 𝜏 → ∞ (or 𝜅 → ∞), one recovers 𝐿(Δ) = 1/[1 + (2Δ/𝜅)2], while for 𝜅 → 0, one obtains
the expected time-window scaling 𝐿(Δ) ≈ 1/[1 + (2Δ𝜏)2]. Small order-unity prefactors that depend on the
precise definition of “width” (FWHM vs. HWHM, rectangular vs. smooth window) can be absorbed into Γover
without altering the linear structure of the master equation.

A compact derivation proceeds by expanding the single-collision unitary

𝑈 = exp [−

𝑖
ℏ

𝐻int𝜏] ≃ 𝕀 −

𝑖
ℏ

𝐻int𝜏 −

1
2ℏ2 𝐻int

2 𝜏2 + 𝑂(𝜙3),

(15)

and evaluating the atom-traced map Φ(𝜌) = Tratom[𝑈(𝜌 ⊗ 𝜌atom)𝑈†]. The second-order terms involve
oscillatory  phase  factors  𝑒±𝑖Δ𝑡  from  the  off-resonant  JC  coupling  and  the  cavity  field  correlation
⟨𝑎(𝑡)𝑎†(0)⟩ ∝ 𝑒−𝜅𝑡/2 set by the ambient bath. Integrals of the form:

𝜏
∫ 𝑑𝑡
0

𝜏

∫ 𝑒−(𝜅/2)∣𝑡−𝑠∣ 𝑒±𝑖Δ(𝑡−s)𝑑s
0

(16)

yield, after coarse-graining, a Lorentzian detuning profile with sum width equal to the cavity width plus
the finite-time Fourier width 1/𝜏. This gives Eq. (14). In the continuous-wave limit 𝜏 → ∞ one recovers the
familiar cavity-limited filter 𝐿(Δ) = 1/[1 + (2Δ/𝜅)2]; conversely, for very short collisions 1/𝜏 ≫ 𝜅  the overlap
is dominated by the Fourier width, suppressing off-resonant exchange.

Operationally, the detuning simply rescales the per-collision strength:

𝜙2 ⟶ 𝜙2𝐿(Δ) in all stream-induced terms.

(17)

1.3  Collision model and effective Lindblad generator

With Eq. (17), expanding the one-passage unitary (Eq. (15)) to 𝒪(𝜙²) and tracing over the pair, yields

the detuning-dressed one-passage map:

Φ(𝜌) ≃ 𝜌 + 𝜙2𝐿(Δ)[𝑟2𝒟[𝑎]𝜌 + 𝑟1𝒟[𝑎†]𝜌],

(18)

where 𝒟[𝐿]𝜌 = 𝐿𝜌𝐿† −

generator is:14

1

2

{𝐿†𝐿, 𝜌}. For a Poisson train of collisions with rate 𝑅, the coarse-grained stream

ℒstream𝜌 = 𝑅𝜙2𝐿(Δ)[𝑟2𝒟[𝑎]𝜌 + 𝑟1𝒟[𝑎†]𝜌].

(19)

In parallel, the ambient phonon contact at 𝑇bath contributes:

ℒbath𝜌 = 𝜅(−)𝒟[𝑎]𝜌 + 𝜅(+)𝒟[𝑎†]𝜌,

𝜅(−) = 𝜅(𝑛ˉ 1 + 1),       𝜅(+) = 𝜅𝑛ˉ 1,

(20)

which satisfies detailed balance at 𝑇bath. The reduced master equation is therefore:

𝜌̇ = ℒbath𝜌 + ℒstream𝜌.

(21)

1.4  Photon-number dynamics and cavity effective temperature

Let 𝑛 = ⟨𝑎†𝑎⟩. Using the standard identities Tr[𝑎†𝑎 𝒟[𝑎]𝜌] = −𝑛 and Tr[𝑎†𝑎 𝒟[𝑎†]𝜌] = 𝑛 + 1, Eq. (21)

closes on 𝑛 as:

7

𝑑𝑛
𝑑𝑡

= −𝜅 (𝑛 − 𝑛ˉ 1) + 𝑅𝑟1 𝜙2𝐿(Δ) − 𝑅(𝑟2 − 𝑟1) 𝜙2𝐿(Δ)𝑛.

(22)

It is convenient to define the detuning-dressed damping and injection,19

Γ↓(Δ) ≡ 𝜅 + 𝑅(𝑟2 − 𝑟1)𝜙2𝐿(Δ),      𝐽↑(Δ) ≡ 𝜅𝑛ˉ 1 + 𝑅𝑟1𝜙2𝐿(Δ),

(23)

so that

𝑑𝑛
𝑑𝑡

= −Γ↓(Δ) 𝑛 + 𝐽↑(Δ).

(24)

The solution for any initial 𝑛(0) is:

𝑛(𝑡) = 𝑛∗(Δ) + [𝑛(0) − 𝑛∗(Δ)]𝑒−Γ↓(Δ)𝑡,

𝑛∗(Δ) =

𝐽↑(Δ)
Γ↓(Δ)

=

𝜅𝑛ˉ 1 + 𝑅𝑟1𝜙2𝐿(Δ)
𝜅 + 𝑅(𝑟2 − 𝑟1)𝜙2𝐿(Δ)

.

(25)

By modeling the steady state using a Bose-Einstein distribution at frequency 𝜔1, we obtain the cavity’s

effective temperature:

𝑛∗(Δ) =

1
𝑒ℏ𝜔1/(𝑘B𝑇cav(Δ)) − 1

⇒ 𝑇cav(Δ) =

ℏ𝜔1
𝑘Bln(1 + 1/𝑛∗(Δ))

.

(26)

1.5  Consistency checks and parameter trends

1.  Resonant limit. For Δ = 0, 𝐿(0) = 1 and Eqs. (22)-(26) reduce to the resonant expressions:

𝑛(𝑡) = 𝑛∗ + [𝑛(0) − 𝑛∗]𝑒−Γ↓𝑡, 𝑛∗ =

𝜅𝑛ˉ 1+𝑅𝑟1𝜙2
𝜅+𝑅(𝑟2−𝑟1)𝜙2 ,      (27)

and

2.  No stream. If 𝑅𝜙2 ⟶ 0, then 𝑛∗(Δ) → 𝑛ˉ 1 and the cavity thermalizes to 𝑇bath.
3.  No phonons. If 𝜅 ⟶ 0, then

𝑇cav =

ℏ𝜔1
𝑘𝐵 ln(1+1/𝑛∗)

 .       (28)

𝑟1
𝑟2 − 𝑟1
independent of Δ, because the cavity line collapses to a delta and the only width in Γover is 1/𝜏; the common
factor 𝐿(Δ) cancels between numerator and denominator in Eq. (25). On resonance this reproduces the
one-subsystem stream relation 𝑛∗/(𝑛∗ + 1) = 𝑟1/𝑟2 and hence

𝑛∗(Δ)

𝜅⟶0
→

(29)

,

𝑒−ℏ𝜔1/𝑘B𝑇cav =

𝑟1
𝑟2

=

𝑒−𝛽ℏ𝜔+cosh (𝛽ℏ𝜆)
𝑒𝛽ℏ𝜔+cosh (𝛽ℏ𝜆)

 ,       (30)

with 𝛽 = 1/(𝑘B𝑇atom).

4.  Large detuning or short dwell. For |Δ| ≫ Γover, 𝐿(Δ) ≪ 1 and the stream’s influence is quenched;
𝑛∗(Δ)  approaches  𝑛ˉ 1.  For  very  short  collisions  1/𝜏  dominates  Γover,  so  𝐿(Δ) ≈ 1/[1 + (2Δ𝜏)2].
There is a natural trade-off: the pump scale grows as 𝜙2 = 𝑔2𝜏2 but the detuning penalty decreases
with larger 𝜏 (narrower Fourier width). For fixed 𝑔, Δ, 𝜅 this produces an optimal 𝜏 (hence optimal
𝜙) that maximizes the net stream leverage.

5.  Role of atom-atom interaction 𝜆. The detuning filter does not modify the bias 𝑟2 − 𝑟1, which remains
set by (𝜔, 𝑇atom, 𝜆) via Eqs. (2)-(7). Increasing |𝜆| increases 𝑍 and thereby weakens the stream’s
net cooling/heating leverage at any fixed Δ through a smaller 𝑟2 − 𝑟1.

1.6  Physical interpretation

The  one-subsystem  stream  provides  an  incoherent  pump  at  rate  𝑅𝑟1𝜙2𝐿(Δ)  (via  𝒟[𝑎†])  and  an
additional  loss  channel  at  rate  𝑅𝑟2𝜙2𝐿(Δ)  (via  𝒟[𝑎]).  Their  difference  𝑅(𝑟2 − 𝑟1)𝜙2𝐿(Δ)  augments  the

8

cavity’s damping if 𝑟2 > 𝑟1(cooling tendency) or acts as net gain if 𝑟1 > 𝑟2 (heating tendency). The ambient
phonon contact fixes the  baseline  occupation  𝑛ˉ 1 and contributes additive damping  𝜅, which  guarantees
stability  even  for  a  slightly  inverted  stream  provided  𝜅 > 𝑅(𝑟1 − 𝑟2)𝜙2𝐿(Δ).  The  detuning  factor  𝐿(Δ)
transparently encodes the spectral-overlap physics: cavity losses and finite interaction time both broaden
the effective window for exchange (through Γover), and only the fraction of the stream strength falling inside
that window contributes to pumping and damping.

1.7  Validity of the detuned description

The derivation assumes (i) weak, short collisions 𝜙 = 𝑔𝜏 ≪ 1; (ii) Poissonian arrivals with 𝑅𝜏 ≪ 1 and
finite  𝑅𝜙2  enabling  Markov  coarse-graining;  (iii)  near-resonant  coupling  and  the  rotating-wave
approximation (RWA) |𝜔 − 𝜔1| ≪ 𝜔1; and (iv) a Born approximation neglecting cavity-stream correlations
between collisions. Within this regime, Eqs. (22)-(26) provide a controlled detuned generalization of the
resonant  one-subsystem  result.  We  note  that  the  choice  Γover = 𝜅 + 1/𝜏  collects  the  two  relevant
broadening  mechanisms  (cavity  damping  and  finite-time  Fourier  width)  in  the  simplest  additive  way.
Alternative conventions may differ by order-unity prefactors (e.g., whether one uses FWHM or HWHM for
1/𝜏); such refinements can be absorbed into a calibrated Γover without altering the linear structure of the
theory.

2.  Correlated atom-pair stream: both atoms coupled a phonon-tethered cavity

We return to the collision model of a stream of correlated atom pairs impinging on a single-mode cavity,
now allowing both atoms of each pair to couple to the field during their short passage. The cavity remains
in contact with an ambient phonon bath at temperature 𝑇bath, described exactly as in Section 1 (with bath
occupation 𝑛̄ 1 defined in Eq. (1)). The reservoir of correlated pairs is the same as in Section 1: the state
weights 𝜌𝑒, 𝜌𝑔, 𝜌𝑑, and 𝜌𝑛𝑑 and the partition function 𝑍 are those already introduced in Eqs. (2) and (3).

Each pair interacts with the cavity for a short time  𝜏. For a single atom coupling with strength 𝑔, we
defined the one-atom small angle 𝜙 = 𝑔𝜏 ≪ 1. When both atoms couple over the same interval, the net
second-order kick on the cavity is enhanced. We parametrize this by an effective per-collision strength

2 ≡ 𝜒𝜙2, 1 ≤ 𝜒 ≤ 2,

𝜙2

(31)

where 𝜒 captures geometry and phase: 𝜒 ≃ 1 for effectively independent or dephased couplings, and
𝜒 = 2 for equal, in-phase, simultaneous couplings that add coherently. Pairs arrive as a Poisson process
with flux 𝑅 (pairs/s).

In order to define the two-subsystem stream coefficients, let 𝑆± ≡ 𝜎𝐴

± denote the collective pair
ladders. The second-order collision map depends on the pair correlators ⟨𝑆+𝑆−⟩ and ⟨𝑆−𝑆+⟩, which evaluate
to:10

± + 𝜎𝐵

(2) = 𝜌𝑒 + 𝜌𝑑 + 𝜌𝑛𝑑,
𝑟1

(2) = 𝜌𝑔 + 𝜌𝑑 + 𝜌𝑛𝑑.
𝑟2

(32)

Two useful identities follow immediately:

(2) − 𝑟1
𝑟2

(2) = 𝜌𝑔 − 𝜌𝑒 =

2 sinh(𝛽ℏ𝜔)
𝑍

> 0 (𝛽ℏ𝜔 > 0),

(2) + 𝑟2
𝑟1

(2) = 1 + 2𝜌𝑛𝑑.

(33)

As  in  Section  1,  short,  weak  collisions  only  “see”  the  spectral  overlap;  we  therefore  use  the  same

detuning filter 𝐿(𝛥) defined in Section 1 and simply replace 𝜙2² → 𝜙2²𝐿(𝛥) in stream-induced terms.

The one-passage map becomes:

Φ(𝜌) ≃ 𝜌 + 𝜙2

2𝐿(Δ)[𝑟2

(2)𝒟[𝑎1]𝜌 + 𝑟1

(2)𝒟[𝑎1

†]𝜌].

(34)

For a Poisson train of collisions at rate 𝑅,

9

ℒstream𝜌 = 𝑅𝜙2

2𝐿(Δ)[𝑟2

(2)𝒟[𝑎1]𝜌 + 𝑟1

(2)𝒟[𝑎1

†]𝜌].

(35)

The cavity’s ambient phonon contact remains exactly as in Section 1; denoting it by ℒbath, the reduced

master equation is:

𝜌̇ = ℒbath𝜌 + ℒstream𝜌.

(36)

2.1  Effective cavity temperature and limiting cases (two-subsystem)

Parametrizing  the  steady  state  by  a  Bose-Einstein  distribution  function  at  𝜔1  the  cavity’s  effective

temperature is given by,

𝑇cav(Δ) =

ℏ𝜔1
𝑘𝐵 ln (1+1/𝑛∗(Δ))

,

(37)

where 𝑛∗(Δ) is obtained from Eq. (25) by replacing 𝑟1, 𝑟2, and 𝜙 respectively with 𝑟1

(2), 𝑟2

(2), and 𝜙2.

2.2  Limiting cases and trends (two-subsystem)

(i) Resonant limit. For Δ = 0, 𝐿(0) = 1, and 𝑛∗(Δ) and 𝑇cav(Δ) reduce to the resonant expressions:

𝑛∗ =

𝜅𝑛ˉ 1+𝑅𝑟1
(2)

(2)

2
𝜙2
(2)

𝜅+𝑅(𝑟2

−𝑟1

)𝜙2

2 and 𝑇cav =

ℏ𝜔1
𝑘𝐵 ln(1+1/𝑛∗)

 .

(ii) No phonons: if 𝜅 → 0, the common factor 𝐿(Δ) cancels,

𝑛∗(Δ)

𝜅⟶0
→

(2)
𝑟1
(2)
−𝑟1
𝑟2

(2) ,

𝑛∗
𝑛∗+1

=

(2)
𝑟1
(2) ,
𝑟2

(38)

i.e., the two-subsystem stream result independent of detuning.

(iii) No stream: if 𝑅𝜙2

2 → 0 (or 𝐿(Δ) → 0 at large |Δ|), 𝑛∗(Δ) → 𝑛̄ 1 (ambient thermalization).

(iv) Trade-off with 𝜏 at fixed 𝑔, 𝛥, 𝜅: the stream leverage scales as 𝜙2

2 ∝ 𝜏² but is penalized by 𝐿(Δ) with

𝛤over = 𝜅 + 1/𝜏; an optimal 𝜏 therefore maximizes the net cooling/heating leverage.

(vi) Role of 𝜌𝑛𝑑 and 𝜒: 𝜌𝑛𝑑 shifts both 𝑟1

(2) equally (Eq. 33), biasing the detailed balance colder
(2) and 𝑟2
2 = 𝜒𝜙².  The
when  𝜌𝑛𝑑 < 0  (e.g.,  𝜆 > 0),  while  𝜒  sets  the  overall  weight  of  the  stream  terms  through  𝜙2
detuning factor 𝐿(Δ) multiplies both pump and slope, transparently encoding spectral mismatch during the
finite interaction.

Validity. The derivation assumes weak, short collisions (𝜙 = 𝑔𝜏 ≪ 1); Poissonian arrivals with 𝑅𝜏 ≪ 1
2 enabling Markov coarse-graining; near-resonant coupling and RWA (|𝜔 − 𝜔1| ≪ 𝜔1); and a
and finite 𝑅𝜙2
Born approximation neglecting cavity-stream correlations between collisions. Within this regime, Eq. (37)
provides a controlled two-subsystem.

2.3  Physical role of 𝝆𝒏𝒅 and 𝝌 (two-subsystem case)

Relative to the one-subsystem configuration, the two-subsystem collision map admits the inter-atomic
(2) (see Eq. (32)).
(2) and 𝑟2
coherence 𝜌𝑛𝑑 directly into both the upward and downward stream coefficients 𝑟1
(2), it does not modify the net damping bias
(2) and 𝑟2
Because 𝜌𝑛𝑑 appears with the same sign in both  𝑟1
(2)  (which  is  fixed  solely  by  the  local  population  asymmetry  𝜌𝑔 − 𝜌𝑒  and  remains  equal  to
(2) − 𝑟1
𝑟2
2sinh(𝛽ℏ𝜔)/𝑍, cf. Eq. (33)), but it does rescale the absolute magnitude of the stream-induced pump and
loss terms. In other words, 𝜌𝑛𝑑 controls how strongly the stream “engages” the cavity without changing the
direction  (cooling  versus  heating)  set  by  the  population  imbalance.  When  𝜌𝑛𝑑 < 0  (e.g.,  𝜆 > 0,
(2)  are  lowered
antiferromagnetic  interaction  between  atoms  in  the  thermal  pair  state),  both  𝑟1
relative to the one-subsystem case at the same temperature 𝑇atom, biasing the effective detailed balance

(2)  and  𝑟2

10

colder than 𝑇cav for a given collision strength. This cold bias competes with the phonon tether that steadily
drags 𝑛 toward the ambient occupation 𝑛̄ 1 set by 𝑇bath.

The geometry/phase factor 𝜒 in 𝜑2

2   =  𝜒𝜑2 (Eq. (31)) quantifies how the second-order “kicks” from the
two simultaneous atom-cavity couplings add. For equal-magnitude, in-phase couplings (atoms traversing
an antinode with identical phases), 𝜒 approaches 2; for geometry or timing that causes partial cancellation
(2)𝒟[𝑎†]]
or independent addition, 1 ≤ 𝜒 < 2. Since the stream Liouvillian enters as 𝑅𝜑2
(Eq. (35)), 𝜒 sets the overall weight of the stream terms multiplicatively with the detuning filter 𝐿(Δ). At finite
detuning  Δ ≠ 0,  the  spectral-overlap  factor  𝐿(Δ)  defined  in  Section  1  multiplies  both  the  upward  and
(2) either; instead, 𝐿(Δ) reduces
downward channels equally, so detuning does not alter the bias  𝑟2
the effective engagement of the stream by the common factor 0 ≤ 𝐿(Δ) ≤ 1. Consequently, the net damping
and injection that appear in the photon-number equation are both dressed by the same  𝜒𝐿(Δ) multiplier
(2).  This
through  𝜑2
separation of roles is useful experimentally: 𝜒 is engineering-controlled (mode shape, trajectory, phase),
(2) is thermodynamics-
𝐿(Δ) is spectroscopy-controlled (detuning and the overlap width 𝛤over), and 𝑟2
controlled (𝜔, 𝑇atom, 𝜆 through 𝑍).

2 𝐿(Δ),  while  the  relative  bias  between  them  is  still  controlled  entirely  by  𝑟2

(2)𝒟[𝑎] + 𝑟1

2𝐿(Δ)[𝑟2

(2) − 𝑟1

(2) − 𝑟1

(2) − 𝑟1

Detuning also introduces a practical trade-off with the dwell time 𝜏. Increasing 𝜏 increases 𝜑2 = 𝑔2𝜏2
2 = 𝜒𝑔2𝜏2, but it simultaneously narrows the time-window contribution to the overlap width,
and therefore 𝜑2
𝛤over = 𝜅 + 1/𝜏 (Section 1), which increases 𝐿(Δ) for any fixed Δ because the Lorentzian denominator 1 +
(2Δ/𝛤over)2  shrinks when 𝛤over grows. Thus, for fixed 𝑔, Δ, and 𝜅 there is an optimal 𝜏 that maximizes the
combined leverage𝜒𝜑2𝐿(Δ). In the regime where 𝜅𝜏 ≫ 1, 𝛤over ≃ 𝜅 and 𝐿(Δ) is governed by the cavity line;
in the opposite regime 𝜅𝜏 ≪ 1, 𝛤over ≃ 1/𝜏 and 𝐿(Δ) ∼ 1/[1  +   (2Δ𝜏)2]. In both regimes, 𝜌𝑛𝑑 and 𝜒 act only
2), while 𝐿(Δ) acts as a spectroscopic gate that
as amplitude multipliers (through 𝑟1
uniformly attenuates stream-induced pump and loss.

(2) and through 𝜑2

(2), 𝑟2

Finally, stability considerations in the detuned two-subsystem model are transparent in the grouped
(2) − 𝑟1
(2) > 0 for 𝛽ℏ𝜔 > 0 (Eq.
2𝐿(Δ). Because 𝑟2
coefficients: the net damping is 𝛤↓
2(Δ) remains strictly positive
(33)), detuning cannot invert the stream; it only scales its strength. Hence  𝛤↓
for any Δ, with the phonon contact guaranteeing stability and setting a baseline relaxation even when 𝜒 is
large and 𝐿(Δ) ≃ 1.

2(Δ) = 𝜅 + 𝑅 (𝑟2

(2) − 𝑟1

(2))𝜑2

2.4  Evaluation workflow for the analytic steady-state (two-subsystem case)

Given the pair thermodynamics (𝑇atom, 𝜆, 𝜔), first compute the state weights 𝜌𝑒, 𝜌𝑔, 𝜌𝑑, and 𝜌𝑛𝑑 and the
partition function 𝑍 as in Section 1. From these, form the two-subsystem stream coefficients via Eq. (32):
(2) = 𝜌𝑔 + 𝜌𝑑 + 𝜌𝑛𝑑.  Next,  specify  the  cavity  frequency  𝜔1  and  the  ambient
(2) = 𝜌𝑒 + 𝜌𝑑 + 𝜌𝑛𝑑  and  𝑟2
𝑟1
) −  1].  Choose  the  collision
temperature  𝑇bath  to  obtain  the  bath  occupation  𝑛̄ 1 = 1/[exp(ℏ𝜔1 𝑘B𝑇bath
2 =  𝜒𝜑2
parameters 𝑔 and 𝜏, which set the one-atom small angle 𝜑 = 𝑔𝜏 and the two-atom enhancement 𝜑2
via  Eq.  (31),  with  𝜒  determined  by  geometry  and  phase.  Fix  the  stream  flux  𝑅  (pairs/s)  and  the  cavity-
phonon rate 𝜅.

⁄

To  introduce  the  spectroscopic  parameters  for  detuning,  pick  the  single-atom  transition  𝜔atom  (the
same 𝜔 used in the pair thermodynamics assuming that the atom that couples has the local gap ℏ𝜔) and
set the detuning 𝛥 . With 𝜏 and 𝜅 specified, evaluate the effective overlap width 𝛤over = 𝜅  +  1/𝜏 (defined in
Section  1)  and  the  detuning  filter  𝐿(Δ) = 1/[1 + (2Δ/𝛤over)2 ].  At  this  point,  all  ingredients  entering  the
photon-number dynamics are known.

the
(2) − 𝑟1

Equations:
and
Compute
(2)(Δ) = 𝜅 + 𝑅 (𝑟2
2𝐿(Δ).  Insert  these  into  Eq.  (24)  to  obtain
𝛤↓
the  linear  birth-death  evolution  for  𝑛(𝑡);  the  steady  state  is  given  in  closed  form  as  𝑛∗(Δ) =
(2) − 𝑟1
. From 𝑛∗(Δ), recover the effective cavity temperature via
[𝜅𝑛ˉ 1 + 𝑅𝑟1
Eq. (37): 𝑇cav = ℏ𝜔1/[𝑘𝐵ln(1 + 1/𝑛1

detuning-dressed
2𝐿(Δ)  and  𝐽↑

damping
(2)(Δ) = 𝜅𝑛ˉ 1 + 𝑅 𝑟1

2𝐿(Δ)] [𝜅 + 𝑅 (𝑟2
⁄

(2))𝜙2
∗)].

injection

(2))𝜑2

2𝐿(Δ)]

(2)𝜙2

(2)𝜙2

from

11

This  pipeline  makes  explicit  how  each  physical  knob  enters:  (i)  thermodynamics  (𝑇atom, 𝜆, 𝜔)   →
(2) through  𝑍;  (ii)  geometry  and  timing  (𝜒, 𝜏)  and  coupling  strength  𝑔 → 𝜑2
(2), 𝑟2
2;  (iii)  spectroscopy
 𝑟1
(Δ, 𝛤over)   →  𝐿(Δ);  and  (iv)  environment  (𝜅, 𝑇bath)  →  baseline  damping  and  occupation.  In  the  limit  Δ  →
 0, 𝐿(Δ)   →  1, recovering the resonant formulas as a special case without changing any step of the workflow.
Conversely, for |Δ| ≫ 𝛤over, 𝐿(Δ) ≪ 1 and the stream’s influence is strongly attenuated, driving 𝑛∗(Δ) toward
𝑛̄ 1; this suppression can be partially offset by increasing 𝜒 or 𝜏, but only within the constraint that 𝜑 = 𝑔𝜏 ≪
1 and 𝑅𝜏 ≪ 1 remain valid for the collisional coarse-graining. If one wishes to revert to the one-subsystem
(2) → 𝑟1,2 (thereby dropping 𝜌𝑛𝑑
counterpart while retaining the phonon bath and detuning, simply replace 𝑟1,2
2 → 𝜑2; the detuning filter 𝐿(Δ) and the ambient terms remain unchanged.
from the coefficients) and set 𝜑2

3.  Results and Discussions

All  results  presented  in  Figures  1–5  and  7  follow  from  direct  evaluation  of  the  closed-form  analytic
expressions for the steady-state photon number derived in Sections 1 and 2. To ensure consistency across
parameter  sweeps,  a  common  baseline  parameter  set  is  adopted  and  summarized  in  Table  2.  These
values,  representative  of  contemporary  circuit-QED  devices,  place  the  system  firmly  within  the  weak-
collision regime (𝜙 = 𝑔𝜏 ≪ 1) and provide a realistic reference point for assessing cooling performance as
individual  parameters  (detuning,  cavity–bath  coupling,  atom–cavity  strength,  and  pair  correlations)  are
varied.

Table 2: Global Baseline Parameters Used in Analytical Evaluations

Quantity

Bath temperature

Atomic (two-qubit) temperature

Cavity frequency
Atom–cavity interaction time
Collision (arrival) rate
Inter-qubit coupling
Jaynes–Cummings coupling
Cavity–bath damping rate
Collision angle

Symbol /
Definition
𝑇𝑏𝑎𝑡ℎ

𝑇𝑎𝑡𝑜𝑚

𝜔1/2𝜋
𝜏
𝑅
𝜆
𝑔/2𝜋
𝜅

Value  Notes

10 K  Cryogenic stage (phonon environment)

50 mK

Effective
reservoir

temperature  of

the  engineered

5 GHz  Fundamental cavity mode
50 ns  Set by flux-pulse duration
5×106 1/s  Mean pair-arrival frequency

5 GHz  Exchange coupling between qubits
0.5 GHz  Coherent qubit–cavity interaction strength
10 kHz  Moderately weak phonon tether

𝜙  =  𝑔𝜏  𝜙 ≈ 0.16  Ensures weak-collision limit (𝜙 ≪ 1)

3.1  Cooling-heating crossover induced by atom-cavity detuning

Figure 1 (a) and (b) summarize the central mechanism by which a correlated atom stream modifies
the  steady-state  temperature  of  a  phonon-tethered  cavity.  The  key  control  parameter  is  the  detuning
Δ = 𝜔 − 𝜔1, which governs both the efficiency and directionality of energy exchange between the incoming
the  single-interaction  rotation  angle  𝜙 = 𝑔𝜏 ≪ 1
atomic  state  and
places the system firmly in the weak-collision regime, the effect of each atomic pair on the cavity is additive
and can be described by an effective birth–death process for the cavity photon number 𝑛∗(Δ).

the  cavity  mode.  Because

As shown in Figure 1 (a), the ratio  𝑇cav/𝑇atom for the one-atom interaction model exhibits a shallow
minimum  near  Δ ≈ 0.  This  behavior  reflects  the  Lorentzian  spectral-overlap  filter  derived  analytically  in
𝑛∗(Δ), Eq. (14), which weights the stream-driven absorption and emission processes. Near resonance, the
cavity  efficiently  absorbs  entropy-carrying  excitations  from  the  incoming  atom  pairs,  and  the  stream-
induced detailed balance

2𝐿(Δ),  becomes  dominant  over  the  intrinsic
with  𝐴↑(Δ) = 𝜅𝑛ˉ 1 + 𝑅 𝑟1
phonon-bath contribution. As a result, the cavity is pulled toward the effective temperature imposed by the
correlated stream, lowering 𝑇cav significantly below the phonon-bath temperature.

2𝐿(Δ), 𝐴↓(Δ) = 𝜅(𝑛ˉ 1 + 1) + 𝑅𝑟2

(2)𝜙2

𝐴↑
𝐴↓ − 𝐴↑

=

𝑛∗
1 + 𝑛∗
(2)𝜙2

12

(a)                                                                             (b)

(c)                                                                             (d)

Figure 1: (a,b) Effective cavity temperature versus atom-cavity detuning. The ratio Tcav/Tatom is shown as a
function of the detuning Δ = 𝜔 − 𝜔1 for one-atom (a) and two-atom (b) interaction models. Near resonance (Δ =
0),  the  correlated  atom-pair  reservoir  efficiently  exchanges  energy  with  the  cavity  and  drives  it  toward  the
reservoir’s temperature. For large detuning, the coherent exchange channel is suppressed, the phonon bath
dominates, and the cavity warms, yielding Tcav > Tatom. Three representative coupling strengths g are shown in
each  panel,  demonstrating  stronger  cooling  for  larger  𝑔.  In  the  one-atom  case  (a),  the  minimum  Tcav/Tatom
remains above unity, approaching a value near ∼ 4. In contrast, in the two-atom case (b), Tcav/Tatom falls below
unity, reaching values near ∼ 0.5 near resonance and for stronger coupling, indicating genuine refrigeration of
the cavity below the atom temperature. (c,d) Steady-state cavity photon number versus detuning. The steady-
state cavity occupation 𝑛∗ is plotted for the same parameters as in (a) and (b), for the one-atom (c) and two-
atom (d) models respectively. The detuning dependence reflects the spectral overlap between the atom and
cavity  modes:  efficient  near  resonance,  resulting  in  reduced  photon  populations,  and  suppressed  at  large
detuning, where the cavity relaxes toward the phonon bath. The behavior of 𝑛∗(Δ) corresponds directly to the
cooling and heating trends observed in panels (a) and (b).

For detunings |∆| ≳ Γover, the stream–cavity energy exchange (𝛿𝐸) is strongly suppressed and 𝐿(Δ) →
0, such that the steady state population 𝑛∗ reduces to the phonon-tethered value 𝑛∗ → 𝑛ˉ 1. This produces
the symmetric heating wings observed in Figure 1 (a): because the stream no longer removes sufficient

13

entropy to compensate bath-induced excitations, the cavity relaxes toward the hotter phonon bath. Thus,
the cooling-heating profile as a function of detuning is an intrinsic consequence of spectral filtering imposed
by the weak-collision channel.

The influence of the coupling strength 𝑔 is highlighted in Figure 1 (a). Increasing 𝑔 enhances the per-
collision  rotation  𝑔𝜏,  thereby  amplifying  the  stream-induced  Lindblad  rates  𝑅𝜙2𝑟1  and  𝑅𝜙2(𝑟2 − 𝑟1).
Physically, this  enlarges the domain in which the atomic stream  overcomes  phonon-bath heating.  As  𝑔
increases, the minimum of 𝑇cav/𝑇atom becomes increasingly pronounced, and efficient cooling persists out
to  larger  values  of  detuning.  This  behavior  is  consistent  with  the  analytic  scaling  in  Eq.  (25),  which
interpolates between a bath-dominated steady state when 𝑔𝜏 → 0 and a stream-dominated regime when
𝑔𝜏 is sufficiently large for the engineered reservoir to dictate the detailed balance.

Figure  1  (b)  also  reveals  the  qualitative  enhancement  provided  by  the  two-atom  interaction  model,
where both members of the correlated pair couple to the cavity. Here the effective upward/downward rates
(2)), which incorporate pair coherence more strongly than
depend on the two-subsystem coefficients (𝑟1
in  the  one-atom  scenario.  As  a  result,  the  cavity  can  be  cooled  below  the  temperature  of  the  atomic
reservoir itself, with 𝑇cav/𝑇atom approaching values as low as ∼ 0.5 near resonance for strong coupling. The
cooling region is also broader: the two-qubit reservoir maintains dominance over the phonon bath across
a wider detuning interval, reflecting its stronger collective interaction with the cavity.

(2), 𝑟2

Figure 1 (c) and (d) plot the corresponding steady-state photon number 𝑛∗(Δ) for the one-atom and
two-atom cases, respectively. The photon-number trends mirror the temperature behavior: in Figure 1 (c),
the minimum of 𝑛∗is modest and remains above the stream-imposed value, whereas in Figure 1 (d) the
cavity photon population is strongly suppressed near resonance, consistent with the significantly deeper
cooling achieved in the two-atom configuration. The close correspondence between 𝑛∗(Δ) and 𝑇cav/𝑇atom
across panels (a-d) confirms that the enhanced refrigeration in the two-atom case arises entirely from the
modified detailed-balance ratios encoded in the correlated-pair coefficients, rather than from any additional
dissipative mechanism.

Together, Figure 1 (a–d) reveal the essential physics of both the one-subsystem and two-subsystem
models.  In  the  one-atom  case,  panels  (a)  and  (c)  show  that  the  cavity  temperature  is  governed  by  a
competition  between  (i)  resonant  entropy  extraction  by  the  correlated  atomic  stream  and  (ii)  thermal
injection from the phonon bath. Detuning suppresses the former through the Lorentzian spectral filter 𝐿(Δ),
while  increasing  coupling  𝑔  strengthens  it  by  enhancing  the  stream-induced  Lindblad  rates  𝑅𝜙2𝑟1  and
𝑅𝜙2(𝑟2 − 𝑟1).  As  a  result,  the  cooling  minimum  at  Δ = 0  marks  the  operational  “sweet  spot’’  of  the
refrigerator,  where  the  stream  imposes  a  nonequilibrium  detailed  balance  that  can  pull  the  cavity
temperature  significantly  below  that  of  the  phonon  reservoir,  though  still  above  the  atom-stream
temperature 𝑇atom.

In contrast, panels (b) and (d) highlight the two-atom interaction case, where both constituents of the
correlated  pair  couple  to  the  cavity.  The  collective  interaction  modifies  the  effective  upward–downward
(2)),  enabling  a  substantially  deeper  cooling  window.  At
rates  via  the  two-subsystem  coefficients  (𝑟1
resonance, the engineered reservoir is strong enough to impose a detailed balance colder than the atomic
reservoir itself, allowing 𝑇cav to drop below 𝑇atom with minima near 𝑇cav/𝑇atom ∼ 0.5. This enhanced cooling
capability is a direct manifestation of pair coherence and collective emission physics, which broaden the
detuning range over which the stream dominates over the phonon bath. Consequently, while the one-atom
model demonstrates stream-assisted refrigeration of a phonon-tethered cavity, the two-atom model reveals
a  qualitatively  stronger  regime  in  which  correlations  within  each  incoming  atomic  pair  enable  genuine
quantum-enhanced refrigeration.

(2), 𝑟2

3.2  Competition between phonon heating and stream-induced cooling

Figure 2 (a–d) illustrates how the cavity steady state emerges from a competition between two distinct
energy-exchange  channels:
the  environmental  phonon  bath,  and
(ii) entropy extraction by the correlated atomic reservoir. The relative strength of these channels determines
whether the cavity is ultimately pulled toward the hot phonon bath or toward the colder effective temperature
imposed by the engineered atom stream.

thermalization  with

(i)

14

(a)                                                                                                                                             (b)

(c)                   (d)

Figure 2: (a,b) Dependence of cavity cooling on phonon coupling. The steady-state temperature ratio 𝑇cav/𝑇atom is
shown as a function of the cavity–phonon energy-damping rate 𝜅 for the one-atom (a) and two-atom (b) interaction
models.  In  the  weak-phonon-coupling  regime,  the  correlated  atomic stream  dominates  the  energy  balance  and
drives the cavity far from the environmental phonon bath. In the one-atom case (a), 𝑇cav/𝑇atom saturates near ∼ 4
as 𝜅 → 0, while in the two-atom case (b) the collective interaction enables genuine refrigeration, with 𝑇cav/𝑇atom →
0.5 in the same limit. As 𝜅 increases, the phonon bath increasingly overwhelms the stream-induced cooling and
forces the cavity back toward its thermal equilibrium value 𝑇cav → 𝑇bath. The crossover marks the departure from
the idealized 𝜅 = 0 limit and quantifies the practical dissipation pathways present in superconducting microwave
cavities. (c,d) Dependence on cavity-atom coupling strength 𝑔. Panels (c) and (d) show 𝑇cav/𝑇atom versus coupling
𝑔 for the one-atom and two-atom models, respectively. Increasing 𝑔 strengthens each weak collision 𝜙 = 𝑔𝜏 ≪ 1,
amplifying the stream-induced Lindblad rates and thereby deepening the cavity cooling window. In both cases the
curves saturate at the same limiting values observed in (a) and (b), near ∼ 4 for the one-atom model and ∼ 0.5 for
the two-atom model, reflecting the intrinsic balance between excitation absorption and emission imposed by the
correlated atomic reservoir. In the weak-coupling regime the stream cannot compete with phonon heating, while
stronger 𝑔 enables the engineered reservoir to impose its own nonequilibrium detailed balance on the cavity mode.
These  panels  highlight  an  experimentally  tunable  transition  between  bath-dominated  and  reservoir-dominated
cavity steady states.

15

In the one-atom model, Figure 2 (a) shows the dependence of 𝑇cav/𝑇atom on the cavity–phonon damping
rate 𝜅. For very small  𝜅, the cavity scarcely exchanges energy with the phonons, and the correlated
atomic stream fully dominates the energy balance. In this limit the cavity saturates at the intrinsic stream-
imposed  detailed  balance,  yielding  𝑇cav/𝑇atom → 4.  This  reproduces  the  behavior  predicted  in  earlier
idealized analyses that neglected external baths.10 As 𝜅 increases, however, the phonon bath injects
excitations  that  gradually  overwhelm  the  weak-collision  channel,  forcing  the  cavity  back  toward  the
hotter environmental temperature. The smooth crossover in Figure 2 (a) therefore captures the realistic
dissipation pathways present in superconducting microwave resonators, where phonon leakage cannot
be ignored.

The two-atom model in Figure 2 (b) exhibits the same qualitative trend but with a dramatically different
limiting behavior. When 𝜅 → 0, the cavity is no longer driven toward a hotter steady state, but instead is
cooled below the atom-pair temperature, reaching 𝑇cav/𝑇atom → 0.5. This quantum-enhanced refrigeration
arises from the collective interaction of both atoms with the cavity, which modifies the upward-downward
transition rates and allows the correlated pair to impose a colder effective detailed balance on the mode.
As 𝜅 increases, phonon-induced excitations again dominate, eventually washing out the cooling advantage
and restoring 𝑇cav → 𝑇bath.

Panels (c) and (d) examine the complementary dependence on the cavity-atom coupling strength 𝑔.
Increasing  𝑔  enhances  the  per-collision  rotation  angle  𝜙 = 𝑔𝜏,  amplifying  the  stream-induced  Lindblad
rates that govern the cooling dynamics. In the one-atom model (c), weak coupling leaves the cavity close
to the phonon-dominated state, while stronger coupling drives it toward the saturated limit near 𝑇cav/𝑇atom ∼
4. In the two-atom case (d), the same trend is observed, but the saturation occurs at the much colder value
𝑇cav/𝑇atom ∼ 0.5.  Increasing  𝑔  therefore  extends  the  domain  where  the  engineered  reservoir  dominates
over the phonon bath and reveals the stronger cooling power of the collective two-atom interaction.

Taken  together,  Figure  2  (a–d)  identifies  the  two  essential  experimental  control  knobs  for
nonequilibrium  cavity  refrigeration:  (i)  the  suppression  of  unwanted  phonon  coupling  (𝜅),  and
(ii) the enhancement of intentional atom-cavity interactions (𝑔). Efficient cooling requires the engineered
reservoir to dominate over environmental heating. This can be achieved either by operating with high cavity
quality factors (small 𝜅) or by increasing 𝑔 so that the correlated atomic stream imposes its own detailed
balance.  The  two-atom  model  demonstrates  that  collective  correlations  can  significantly  outperform  the
one-atom case, enabling the cavity temperature to be driven below the atomic reservoir temperature, an
important capability for integrating such cavities with thermally sensitive quantum devices such as quantum
dots, spin qubits, and bosonic memories.

3.3  Weak influence of pair correlations and global structure of the cooling landscape

Figure 3 (a–d) explores how intra-pair correlations and global parameter variations shape the steady-
state  temperature  of  a  phonon-tethered  cavity.  Whereas  detuning  and  phonon  damping  govern  the
dominant cooling-heating balance, the intra-pair exchange interaction 𝜆 modulates the internal structure of
the correlated atomic reservoir and therefore affects the detailed balance imposed on the cavity. The effect,
however, depends critically on whether one or both atoms interact with the cavity.

, 𝜌𝑑

, 𝜌𝑔

In the one-atom model, Figure 3 (a) shows that 𝑇cav/𝑇atom exhibits only a weak dependence on 𝜆, and
in fact increases slightly as 𝜆 grows. This behavior follows from the structure of the arrival coefficients in
the  one-subsystem  configuration.  Although  the  incoming  thermal  pair  is  described  by  the  weights
, 𝜌𝑛𝑑) of Eqs. (2)-(3), only one spin interacts with the cavity, so the effective upward and downward
(𝜌𝑒
weights  reduce  𝑟1 = 𝜌𝑒 + 𝜌𝑑  and   𝑟2 = 𝜌𝑔 + 𝜌𝑑.  The  coherence  term  𝜌𝑛𝑑,  responsible  for  the  correlated
advantages in the two-atom case, drops out entirely. Consequently, the cooling-relevant difference  𝑟2 −
𝑟1 = 𝜌𝑔 − 𝜌𝑒 depends only on the single-spin splitting 𝜔 and is independent of 𝜆 except for normalization
via  the  partition  function  𝑍.  Increasing  𝜆  increases  the  weight  of  the  𝜌𝑑  manifold,  which  appears
symmetrically in both 𝑟1and 𝑟2. The net effect is a mild reduction in stream-induced cooling efficiency, hence
the slow rise of 𝑇cav/𝑇atom with increasing 𝜆.

The situation is qualitatively different in the two-atom model, where both atoms interact simultaneously
with the cavity. In this case (Figure 3 (b)), the stream-induced rates depend on the full set of two-subsystem

16

(2), 𝑟2

(2), which now contain the pair coherence 𝜌𝑛𝑑. Increasing 𝜆 enhances the coherent mixing
coefficients 𝑟1
between |𝑒𝑔⟩ and |𝑔𝑒⟩, strengthening the collective interaction channel and suppressing the net upward
transition  rate  relative  to  the  downward  one.  The  result  is  a  monotonic  decrease  of  𝑇cav/𝑇atom  with
increasing  𝜆,  signaling  a  genuine  quantum-enhanced  refrigeration  mechanism  unavailable  in  the  one-
subsystem configuration.

(a)                   (b)

(c)                                                                             (d)

Figure 3: (a,b) Influence of the intra-pair exchange coupling 𝜆 on cavity cooling. The ratio 𝑇cav/𝑇atom is plotted as a
function of the exchange coupling 𝜆 for the one-atom (a) and two-atom (b) interaction models. The exchange term
, 𝜌𝑑, 𝜌𝑛𝑑)  and  thereby  modifies  the  stream-imposed
𝜆  reshapes  the  correlated  pair’s  internal  populations  (𝜌𝑒
detailed  balance  acting  on  the  cavity.  In  the  one-atom  model  (a),  increasing  𝜆  raises  𝑇cav/𝑇atom,  reflecting  the
reduced ability of the correlated pair to extract entropy when only a single subsystem interacts with the cavity. In
contrast, in the two-atom model (b), stronger exchange enhances collective coherence and improves the net cooling
power,  leading  to  a  monotonic  decrease  of  𝑇cav/𝑇atom  with  𝜆.  (c,d)  Two-dimensional  cooling  landscape  versus
detuning and coupling strength. Panels (c) and (d) show color maps of 𝑇cav/𝑇atom as a function of both the detuning
Δ and the atom-cavity coupling strength 𝑔, for the one-atom and two-atom models, respectively. Cooling “valleys’’
appear near resonance and deepen with increasing 𝑔, while off-resonant or weak-coupling regions are dominated
by phonon-induced heating. The two-atom model (d) exhibits a markedly stronger refrigeration effect and a broad

, 𝜌𝑔

17

region  where  𝑇cav/𝑇atom < 1,  indicating  true  cavity  cooling  below  the  atom-stream  temperature.  These  maps
delineate  the  operational  regime  in  which  correlated  atomic  pairs  can  overcome  phonon  heating  and  impose  a
colder nonequilibrium steady state on the cavity.

Figure  3  (c)  and  (d)  present  two-dimensional  maps  of  𝑇cav/𝑇atom  as  functions  of  detuning  Δ  and
coupling  strength  𝑔.  These  maps  synthesize,  in  a  single  representation,  the  competing  influences  of
spectral overlap and interaction strength. Near resonance, spectral matching is optimal and increasing  𝑔
amplifies each weak collision, producing the “cooling valleys’’ characteristic of effective stream-dominated
refrigeration. Far from resonance, the Lorentzian overlap 𝐿(Δ) suppresses stream-induced exchange, and
phonon-induced heating drives the cavity toward 𝑇bath, generating broad heating plateaus at large |Δ|.

The contrast between  Figure 3 (c) and (d) reinforces  the fundamental  distinction between the one-
atom  and  two-atom  models:  correlations  play  only  a  minor  role  when  one  subsystem  interacts  with  the
cavity, but become central when both atoms couple collectively. In the two-atom landscape, the enhanced
coherence channels produce a wide region where 𝑇cav/𝑇atom < 1, highlighting a robust refrigeration regime
not accessible to the one-subsystem.

3.4  The effect of phonon damping

Figure 4 directly illustrates how cavity-phonon dissipation reshapes the steady-state temperature in
both  the  one-atom  and  two-atom  interaction  models.  When  the  cavity-phonon  damping  rate  𝜅 → 0,  the
cavity becomes effectively isolated from its environmental phonon bath, and its temperature is governed
almost  entirely  by  the  correlated  atomic  reservoir.  In  this  limit  the  one-atom  model  saturates  at  the
characteristic  value  𝑇cav/𝑇atom ∼ 4,  while  the  two-atom  model  reaches  the  substantially  colder  value
𝑇cav/𝑇atom ∼ 0.5. These limiting ratios reproduce the predictions of the idealized 𝜅 = 0 theory and serve as
reference baselines for evaluating all realistic scenarios.10

(a)                                                                                (b)

Figure 4: Influence of cavity–phonon damping on cavity cooling. The ratio 𝑇cav/𝑇atom is plotted for several cavity-
phonon  damping  rates  𝜅  in  the  one-atom  (a)  and  two-atom  (b)  interaction  models.  These  curves  illustrate  the
crossover between the idealized limit 𝜅 → 0, where the cavity is effectively isolated from its phonon environment,
and the  realistic  phonon-tethered  regime  relevant for superconducting  microwave cavities.  In the  weak-phonon-
coupling limit, the cavity decouples from the environmental bath and the steady state is governed almost entirely
by the correlated atomic reservoir. In this regime the one-atom model saturates near 𝑇cav/𝑇atom ∼ 4, whereas the
collective two-atom model reaches significantly colder values 𝑇cav/𝑇atom ∼ 0.5, reflecting its enhanced refrigeration
capability. As the damping rate 𝜅 increases, phonon-induced excitations progressively suppress the stream’s ability
to extract entropy, and the cavity relaxes toward the phonon-bath temperature 𝑇bath. The sensitivity of these curves
to 𝜅 demonstrates how even modest phonon leakage reshapes the cooling window and establishes practical limits
on stream-dominated refrigeration in realistic superconducting cavity platforms.

18

As soon as 𝜅 becomes nonzero, even at values as small as a few hertz, phonon-induced excitations
begin to compete with the entropy extraction performed by the atomic stream. This additional thermalization
pathway progressively pulls the cavity toward the phonon-bath temperature 𝑇bath, weakening the stream-
imposed detailed balance. The resulting crossover is clearly visible in both panels of Figure 4: for increasing
𝜅, the cavity temperature rises from its stream-dominated value toward the phonon-dominated equilibrium.
Because  the  one-atom  model  begins  from  a  relatively  hot  baseline,  its  temperature  rises  more  rapidly,
whereas the two-atom model retains its refrigeration advantage over a broader window of phonon damping.

These  trends  delineate  the  practical  regime  in  which  engineered  reservoir  cooling  can  outperform
environmental dissipation. Even modest phonon damping significantly narrows the operational window for
strong cooling, emphasizing the importance of high-quality superconducting cavities for realizing stream-
driven refrigeration. At the same time, the two-atom model demonstrates a substantially greater robustness
to  phonon  leakage,  reflecting  the  enhanced  cooling  power  generated  by  pair  coherence  and  collective
interaction.

Thus, Figure 4 confirms both the consistency of our model with the  𝜅 → 0 limit and the necessity of

incorporating phonon coupling when assessing cooling performance in realistic experimental platforms.

A noteworthy distinction between the one-atom and two-atom coupling models becomes evident when
examining the detuning dependence in the idealized limit of vanishing cavity-phonon contact (𝜅 → 0), as
shown  in  Figure  5.  In  the  one-atom  configuration,  the  ratio  𝑇cav/𝑇atom  displays  a  weak  but  discernible
asymmetry as the atomic transition 𝜔 = 𝜔1 + Δ is swept over the range Δ ∈ [−50,50] MHz. This asymmetry
arises from the unequal thermal weights 𝜌𝑒 ∝ 𝑒−𝛽ℏ𝜔 and 𝜌𝑔 ∝ 𝑒+𝛽ℏ𝜔, which feed directly into the detailed
balance through the one-subsystem arrival coefficients 𝑟1 and 𝑟2. Because only a single spin interacts with
the cavity, these coefficients retain the antisymmetric dependence on Δ, leading to a slow monotonic drift
of 𝑇cav/𝑇atom as the atomic frequency is tuned.

Figure  5:  Effective  cavity  temperature  in  the  one-  and  two-atom  interaction  models  in  the  absence  of  phonon
coupling.  The  ratio 𝑇cav/𝑇atomis shown as  a function  of  the  detuning  Δ = 𝜔 − 𝜔1  for 𝜅 = 0,  isolating  the  intrinsic
effect of the correlated atomic reservoir. In the one-atom model, the curve exhibits a weak asymmetry as Δ is swept
over ±50 MHz: 𝑇cav/𝑇atom decreases approximately linearly with increasing detuning and continues to fall even for
Δ > 0. This slight asymmetry originates from the unequal thermal weights 𝜌𝑒 and 𝜌𝑔 that determine the effective
upward and downward transition rates when only a single atom couples to the cavity. The two-atom model displays
a much weaker dependence on Δ. Because both atoms interact collectively with the cavity, the relevant upward–
downward rates incorporate symmetric pair-coherent contributions that suppress the detuning-induced imbalance
present in the one-atom configuration. As a result, the two-atom curve is nearly symmetric about Δ = 0, reflecting
the enhanced robustness of collective cooling against detuning.

19

In  contrast,  when  both  atoms  interact  with  the  cavity,  the  correlated  contributions  from  the  single-
excitation manifold, encoded in 𝜌𝑑 and, crucially, the coherence term 𝜌𝑛𝑑, modify the effective upward and
downward transition rates. These collective terms suppress the antisymmetric  Δ-dependence present in
the one-atom case, producing an almost symmetric and significantly weaker detuning dependence. The
two-atom model thereby exhibits a nearly flat response across the same detuning range.

This contrast highlights the deeper role played by pair correlations: even in the absence of any phonon
bath,  collective  coupling  alters  the  effective  reservoir  seen  by  the  cavity,  smoothing  out  detuning
asymmetries that are otherwise unavoidable in single-spin interaction models.

4.  Experimental Mapping to a cQED Architecture

4.1  Motivation and Cryogenic Context

A central challenge in scaling quantum hardware arises from the steep drop in available cooling power
as one descends below the kelvin scale in standard dilution refrigerators. While the 20 mK stage typically
offers only on the order of 10-20 μW, the 100 mK stage reaches sub-milliwatt levels (≈ 0.5 mW in modern
systems),  illustrating  a  multi-order-of-magnitude  disparity  across  the  cryostat  tiers;  representative
specifications from commercial systems document ≈ 16 μW at 20 mK and ≈ 0.5 mW at 100 mK for LD-
class  units.1  By  contrast,  auxiliary  1  K  stages  and  dedicated  1  K  systems  can  supply  far  larger  cooling
capacities, often at or above the ≳ 100 mW scale, making the 1-4 K envelope an attractive “high-power
zone’’ for co-locating electronics, filtering, and other infrastructure.2

This  imbalance  forces  nearly  all  quantum  devices  (superconducting  qubits  and  resonators,
semiconductor spin qubits, low-noise  amplifiers,  and  sensitive readout circuitry) to reside at the  lowest-
temperature  plate,  where  cooling  resources  are  scarcest  and  wiring  heat  loads  become  the  dominant
architectural  constraint.  Reviews  of  superconducting-qubit  hardware  emphasize  these  system-level
pressures and the importance of cryogenic staging and thermal budgeting.8

A  promising  direction  for  alleviating  this  bottleneck  is  to  relocate  as  much  classical  and  passive
hardware as possible to higher-temperature stages (1-4 K) while employing engineered quantum reservoirs
to  pull  specific  degrees  of  freedom,  selected  cavity  modes,  qubits,  or  mesoscopic  devices,  to  effective
temperatures  far  below  the  ambient  of  the  supporting  stage.  The  broader  idea  is  rooted  in  quantum
thermodynamics:  few-qubit  thermal  machines  and  autonomous  quantum  absorption  refrigerators  can
impose  a  desired  detailed  balance on a target mode  without net  external  work,  functioning as compact
quantum  coolers  that  create  local  regions  of  low  entropy  even  inside  a  comparatively  warm
environment.15,16  In  parallel,  rapid  progress  in  cryo-CMOS  electronics  at  the  4  K  tier  is  validating  the
complementary system design strategy of shifting classical control and readout overhead to warmer plates,
thereby  reserving  the  coldest  stage  primarily  for  components  that  genuinely  require  millikelvin
operation.40,41

Within this context, the two-atom (two-qubit) refrigerator architecture explored here functions as  an
engineered reservoir that can operate at a warmer stage yet impose a colder effective temperature on a
target cavity mode or device degree of freedom. By combining the abundant cooling power available around
1 K with an autonomous quantum-refrigeration mechanism, this approach aims to relieve thermal budgets
at  the  base  plate  while  still  preserving  the  low-entropy  conditions  required  locally  for  high-fidelity
operation.3,11

Figure 6 summarizes the circuit-QED implementation of the two-qubit refrigerator within a cryogenic
environment. A high-Q 3D cavity, hosting the refrigerated mode at effective temperature 𝑇cav, is embedded
in a  phonon  bath at 𝑇bath and coupled coherently to  a pair of superconducting qubits that constitute the
engineered atomic reservoir at 𝑇atom. The schematic highlights the coherent Jaynes–Cummings exchange
between  the  qubits  and  the  cavity,  the  dissipative  reset  channel  that  returns  the  qubit  reservoir  to  a
low-entropy state via a Purcell-filtered decay path, and the resulting closed heat-flow loop in which entropy
is repeatedly pumped from the cavity into the engineered reservoir and then into the 1 K stage. This layout
makes explicit how a locally cold cavity mode can be sustained inside a comparatively warm but high-power
cryogenic tier, setting the stage for the quantitative temperature profiles presented in Figure 7.

20

𝑇𝑏𝑎𝑡ℎ Environment

Mode B (ℏ𝜔𝐵)

Mode A (ℏ𝜔𝐴, 𝑇𝑐𝑎𝑣)

Cavity Resonator

𝑔, 𝜏

𝑇𝑎𝑡𝑜𝑚
Q1

𝜆

Q2

𝑇𝑟𝑒𝑠𝑒𝑡
Two Qubit
Reservoir, 𝑇𝑎𝑡𝑜𝑚
Γ

Purcell
Filter

Bath  (heating)       Cavity Mode A (cooled)
       Two-qubit refrigerator  (absorbs photons)
Reset Channel (entropy dump)
        Bath

Dissipative Reset

𝜏𝑟𝑒𝑠𝑒𝑡

Figure 6: Circuit-QED implementation of the two-qubit refrigerator inside a cryogenic environment. A high-Q 3D
cavity hosting Modes A at an effective temperature 𝑇cav is embedded in a phonon bath at 𝑇bath and coupled via
Jaynes-Cummings  interactions  𝑔  to  a  pair  of  superconducting  qubits  that  constitute  the  engineered  atomic
reservoir  at 𝑇atom.  The  arrow  labeled Γ  represent  dissipative  channel  from  the  qubits  into  a  coarse-grained
two-qubit reservoir, which is reset through a Purcell-filtered decay path at rate τ𝑟𝑒𝑠𝑒𝑡 into the 𝑇bath stage. Together
these  processes  realize  a  repeated-interaction  refrigeration  cycle  that  pumps  entropy  from  the  cavity  into  the
engineered reservoir and ultimately into the bath, allowing the cavity mode A to equilibrate at 𝑇𝑐𝑎𝑣 while residing
on the same high-power cryogenic tier.

Figure 7 illustrates the practical significance of the engineered reservoir by showing the steady-state
cavity temperature as a function of the cavity–bath damping rate 𝜅 for several bath temperatures spanning
the full cryogenic range relevant to superconducting hardware. In the one-subsystem case [Figure 7 (a)],
the  engineered  reservoir  drives  the  cavity  to  an  effective  temperature  of  𝑇cav ≈   219  mK  in  the  limit  of
negligible  bath  contact,  independently  of  𝑇bath.  Increasing  𝜅  couples  the  mode  more  strongly  to  the
environment, gradually pulling the cavity toward 𝑇bath: for a 100 mK bath the engineered refrigeration cools
toward  the  bath  temperature,  while  for  higher  bath  temperatures  (500  mK,  1  K,  4  K)  the  cavity  instead
warms toward the ambient. By contrast, the two-subsystem configuration [Figure 7 (b)] exhibits far stronger
refrigeration:  in  the  idealized  limit  𝜅 → 0,  the  cavity  approaches  𝑇cav ≈  25  mK,  irrespective  of  the  bath
temperature. This demonstrates that pair quantum correlations enable the engineered reservoir to impose
a substantially colder detailed balance on the cavity mode. In realistic devices, where 𝜅 can be engineered
to lie well below the kilohertz scale through careful cavity design and shielding, these results suggest that
sub–100-mK effective cavity temperatures are achievable, even when the surrounding cryostat is at 1 K or
higher. This establishes the feasibility of using a two-qubit refrigerator to create localized millikelvin regions
inside  high-power  cryogenic  stages,  enabling  cold  microwave  modes  for  quantum  memories  or  as
refrigeration interfaces for semiconductor qubits.

A practical reason to employ a two-atom (two-qubit) working medium is that direct “algorithmic” cooling
of a bosonic cavity mode is fundamentally resource-intensive. A cavity has an infinite ladder of Fock states;
cooling  it  algorithmically  would  require  removing  excitations  at  arbitrarily  high  photon  numbers  or
equivalently  engineering  a  highly  selective  dissipative  channel  that  enforces  detailed  balance  at  a
temperature  below  the  ambient  stage.  Few-level  systems,  by  contrast,  admit  fast,  repeatable  resets.  In
superconducting  cQED,  qubits  can  be  reinitialized  either  autonomously  by  engineered  dissipation  or
actively by measurement-based feedback, with mature experimental implementations. Fast unconditional
all-microwave  qubit  reset  and  deterministic  measurement-feedback  reset  have  been  demonstrated  with
sub-microsecond latencies, and driven-dissipative schemes autonomously stabilize target states, providing
a  compact  route  to  entropy  dumping.7,8,9  These  capabilities  make  qubits  efficient  sacrificial  refrigerants,
whereas the cavity serves as the cold stage that is repeatedly cooled through engineered contact with the
qubit reservoir.

21

(a)                                                                           (b)

Figure 7: Steady-state cavity temperature 𝑇cav versus cavity–bath damping rate 𝜅 for (a) one-subsystem coupling
(only one atom interacts with the cavity) and (b) two-subsystem coupling (both atoms interact). Curves are shown
for four ambient bath temperatures 𝑇bath =  100 mK, 500 mK, 1 K, 4 K. In (a), the one-atom configuration yields a
minimum  achievable  cavity  temperature  of  approximately  𝑇cav ≈    219  mK  at  𝜅 = 0,  independent  of  𝑇bath.  As  𝜅
increases, the cavity gradually equilibrates with the bath: at 𝑇bath =  100 mK, the cavity cools toward 100 mK, while
for  higher  𝑇bath  it  warms  toward  the  bath  temperature.  In  (b),  the  two-atom  configuration  enables  significantly
stronger refrigeration: for vanishing 𝜅, the cavity reaches 𝑇cav ≈ 25 mK across all bath temperatures. Increasing 𝜅
again drives the steady state toward 𝑇bath.

Using two qubits rather than one adds capabilities essential to our refrigerator. First, correlations in a
two-qubit pair, embedded, for instance, in the singlet/triplet superpositions |𝑒𝑔⟩ ± |𝑔𝑒⟩, modify the upward
and downward transition rates felt by the cavity. In the collision-model language, the coefficients (𝑟1, 𝑟2) or
(2))  are  shifted  by  pair  coherence,  altering  the  effective  detailed
their  two-subsystem  extensions  (𝑟1
balance  and  enabling  quantum-enhanced  cooling,  where  the  cavity’s  temperature  𝑇cav  drops  below  the
effective temperature of the pair. This mechanism parallels collective emission and absorption in correlated
emitters  (Dicke  super-  and  subradiance)  and  is  closely  related  to  micromaser-style  cavity  pumping  by
successive ancillae.14,42,43

(2), 𝑟2

Second,  a  two-qubit  working  medium  broadens  the  design  space  for  engineered  reservoirs.  By
preparing and resetting appropriate two-qubit manifolds, one can tailor an effective temperature, spectral
response,  and  energy-exchange  directionality  with  the  cavity,  central  objectives  in  quantum  reservoir
engineering and autonomous refrigeration.17,44,45

Finally,  this  mapping  is  directly  compatible  with  modern  hardware:  the  two-atom  system  can  be
realized with two transmons (planar or 3D), or with flux/fluxonium-type qubits, all within the standard circuit-
QED toolset. Strong and even ultrastrong light-matter couplings are routine, and tunable couplers provide
wide, fast control of the qubit-qubit exchange needed to prepare and reset correlated states.9,20,21,22 This
establishes a clear experimental path for implementing the theoretical refrigerator analyzed in this work.

4.2  Realization with Two Superconducting Qubits and a 3D Cavity

The  core  elements  of  the  proposed  refrigerator,  a  high-Q  microwave  cavity  and  a  pair  of
superconducting  qubits,  map  naturally  onto  standard  cQED  hardware.  A  3D  superconducting  cavity
provides long photon lifetimes, clean electromagnetic mode structure, and compatibility with tunable and
fixed-frequency qubits. In this architecture, the cavity plays the role of the cold working medium, while the
two qubits act as a resettable quantum reservoir that repeatedly absorbs entropy from the cavity and dumps
it into a cold load through active reset.

22

Each cooling cycle begins with preparing the two qubits in a prescribed mixed state with well-defined
populations and pair coherence (𝜌𝑒, 𝜌𝑔, 𝜌𝑑, 𝜌𝑛𝑑). This can be achieved with a short sequence of single-qubit
rotations and an entangling gate, optionally combined with engineered dissipation to stabilize the desired
manifold.  Such  driven-dissipative  state-preparation  schemes  are  standard  in  quantum  reservoir
engineering,  which  uses  the  competition  between  coherent  drive  and  controlled  dissipation  to
autonomously stabilize target subspaces or entangled states.23,24,25

These preparation procedures realize the “initial ancilla state’’ assumed in the collision-model section
of our theory. The coarse-grained Lindbladian emerging from repeated ancilla-cavity collisions is rigorously
justified  in  the  open-quantum-systems  literature,12  providing  the  microscopic  basis  for  describing  each
engineered two-qubit pair as a single “collision” imparting an 𝒪(𝜙2) kick to the cavity, where 𝜙 = 𝑔𝜏 is the
Jaynes-Cummings interaction angle.

To implement the interaction time 𝜏 assumed in the theory, the qubits are tuned into (near) resonance
with  the  cavity  for  the  desired  interval  by  applying  nanosecond-scale  flux  pulses.  Experimental
demonstrations across both 3D and planar cQED platforms show that qubit transition frequencies can be
shifted  by  hundreds  of  MHz  with  sub–100-ns  latencies,  enabling  rapid  “engage”  and  “disengage’’
operations.46,47 Tunable-coupler architectures and flux-tunable transmons routinely provide multi-hundred-
MHz to multi-GHz frequency agility, allowing  precise  control of  both the qubit-cavity detuning  Δ and the
inter-qubit exchange coupling 𝜆.

During the engaged interval 𝜏, the two-qubit system and the cavity undergo a weak Jaynes-Cummings
exchange. By choosing 𝜙 = 𝑔𝜏 ≪ 1, a single interaction reproduces the weak-collision limit required in the
theoretical model, including the detuning-filtered energy exchange rate governed by the Lorentzian overlap
𝐿(Δ).  Experimental  control  of  detuning  is  routine  in  cQED,  with  MHz-  to  GHz-scale  offsets  readily
programmed by flux modulation.

After interaction with the cavity, both qubits must be returned rapidly to a low-entropy state before the
next cycle. Reset is the thermodynamic “work step’’ that evacuates the entropy absorbed from the cavity.
Circuit-QED  experiments  now  support  several  mature  reset  modalities,  all  compatible  with  MHz-rate
refrigeration cycles:

1.  Dissipative (Purcell-enhanced) reset:

The qubit is temporarily coupled to a low-Q mode or Purcell-filtered decay channel, enabling
unconditional relaxation with time constants of order 102 ns and residual excitations at or below
10−3 − 10−2.26,27,28,29

2.  Measurement-and-feedback reset:

High-fidelity projective measurement followed by a conditional 𝜋-pulse yields near-ground-state
preparation on microsecond timescales and with high robustness24,48

3.  Autonomous (driven-dissipative) reset:

Parametric or multi-tone microwave drives engineer an attractor in the qubit’s state space,
enabling unconditional ground-state reset in sub–500-ns without classical feedback.23,32,33

4.  Stimulated-emission and parametric-sideband reset:

Recently demonstrated architectures use broadband or parametrically driven couplings to funnel
qubit excitations into a cold bath with sub–100-ns latencies while suppressing unwanted Purcell
decay during idle periods.22,31

These reset modalities have been extensively optimized in the cQED community and reliably maintain
effective  qubit  temperatures  in  the  30-80  mK  range  at  5-7  GHz  transition  frequencies,  consistent  with
residual  populations  of  ~1-3%.  Because  the  1-K  stage  provides  abundant  cooling  power,  the  classical
microwave  resources  needed  to  drive,  measure,  and  reset  the  qubits  are  easily  supported  at  that
temperature.

Once  preparation,  interaction,  and  reset  are  available,  the  full  repeated-interaction  refrigerator  is

straightforward to realize. Each “cooling cycle’’ consists of:

1.  Prepare correlated two-qubit state.
2.  Bring both qubits into resonance with the cavity for time 𝜏.
3.  Allow an 𝒪(𝜙2) energy exchange event with adjustable detuning Δ.

23

4.  Disengage and reset both qubits.
5.  Repeat at a programmable rate 𝑅.

Randomizing  the  start  time  of  each  cycle  (or  using  an  aperiodic  digital  trigger)  reproduces  the
Poissonian arrival statistics underlying the collision model and the master equation used in sections 1 and
2. This mapping has deep roots in the micromaser literature, where successive atoms passing through a
cavity generate an effective Lindblad generator with upward and downward rates proportional to 𝑅𝑟1𝜙2 and
𝑅𝑟2𝜙2.14 Under realistic parameters (MHz-class arrival rates, small angles  𝜙 ≲ 0.1, detunings in the 10-
100-MHz range,  and cavity damping 10-100 kHz) the  resulting  effective  damping  Γ↓ and source term  𝐽↑
yield cavity-cooling time constants of order 10-100 μs, compatible with standard cQED coherence windows.

Thus, the repeated-interaction refrigerator studied in our theory can be implemented almost directly
using off-the-shelf cQED components. All required ingredients (fast state preparation, rapid tunability, high-
fidelity reset, and multimode cavity engineering) are already experimentally demonstrated, establishing the
feasibility of recreating our idealized two-atom model within present-day superconducting-qubit hardware.

4.3  Reset Mechanisms and Achievable Temperatures

Reset plays a central role in enabling the two-qubit refrigerator, as it constitutes the entropy-dumping
step required to maintain the working medium at low effective temperature. Without a fast and high-fidelity
return to a low-entropy state, the qubits would accumulate excitations extracted from the cavity and the
refrigeration  cycle  would  saturate.  Within  circuit  QED,  qubit  reset  has  matured  into  a  well-developed
technological capability, with several experimentally established modalities that operate at rates compatible
with the MHz-class repeated-interaction cycles assumed in our theoretical model. These reset channels
exploit  either  engineered  dissipation  or  active  measurement  and  control  to  funnel  population  out  of  the
excited state and thereby maintain the qubit subsystem at a low effective temperature.

Dissipative reset is often realized through Purcell-enhanced relaxation, where the qubit is temporarily
coupled to a low-Q auxiliary mode or a dedicated Purcell filter so that the excited state decays rapidly into
a  cold  load.  Experiments  have  demonstrated  unconditional  ground-state  reset  with  characteristic  time
constants on the order of 102 ns and with residual excited-state populations as low as 10−3 − 10−2; more
recent  architectures  employ  tunable  or  multi-mode  Purcell  filters  to  combine  rapid  reset  with  protection
against  unwanted  decay  during  coherent  operations.21,22,26,29  A  complementary  approach  uses
measurement followed by classical feedback, wherein a high-fidelity projective measurement is performed
and  an  appropriate  conditional  𝜋-pulse  is  applied  to  return  the  qubit  to  its  ground  state.  Deterministic,
measurement-based  reset  routinely  achieves  near-ground-state  preparation  within  microsecond  time
intervals and has enabled repeated initialization well beyond the limits set by intrinsic 𝑇1 relaxation.24,48

Autonomous,  measurement-free  reset  provides  yet  another  path  to  rapid  entropy  removal.  In  such
schemes,  the  competition  between  coherent  microwave  drive  and  engineered  dissipation  creates  an
attractor  in  the  qubit  state  space  that  deterministically  funnels  population  toward  the  ground  state,
eliminating  the  need  for  feedback  loops.  Driven-dissipative  protocols  have  prepared  fixed-frequency
transmons to their ground state within sub–500-ns windows and have been extended to the unconditional
stabilization  of  entangled  two-qubit  states,  making  the  approach  particularly  appealing  for  thermal
machines based on correlated qubit pairs.23,32,33 Related ideas activate stimulated emission or parametric
sideband transitions to accelerate population transfer into a lossy bath. Architectures employing broadband
waveguides or parametrically modulated couplers have reported unconditional reset latencies below 100
ns,  together  with  the  ability  to  suppress  Purcell  decay  during  idle  times  through  tunable  coupling
elements.22,27

These  reset  modalities  are  powerful  enough  to  maintain  the  two-qubit  subsystem  at  an  effective
temperature in the 30-80 mK range even while operating at a 1 K cryostat stage. A residual excited-state
probability of only a few percent at qubit transition frequencies in the 5-7 GHz range corresponds directly
to effective temperatures of this order, consistent with the targets needed for the refrigerator. The classical
microwave resources required to implement these resets (high-power drives, broadband control lines, and
fast measurement electronics) can be supported readily at the 1 K tier, which typically offers cooling powers
in the  hundreds of milliwatts. Commercial 1 K  platforms provide between  200-700 mW  at temperatures
around  1-1.2  K,  and  dedicated  1  K-loop  stages  integrated  into  dilution  refrigerators  offer  similarly  high

24

capacities.1,5  These  conditions  make  the  1  K  environment  particularly  attractive  for  hosting  the  qubit-
reservoir infrastructure, with the cavity mode acting as the object to be cooled inside this warm but powerful
stage.

The resulting steady-state temperatures predicted by our collision-model dynamics reflect this balance.
Under realistic conditions (MHz-scale repetition rates, small interaction angles  𝜙 = 𝑔𝜏, weak cavity-bath
coupling 𝜅 in the 10-100 Hz to 1 kHz range, and detuning-controlled exchange governed by the spectral
filter 𝐿(Δ)) the cavity cools toward a steady state set predominately by the engineered qubit reservoir rather
than by the ambient 1 K bath. The resulting cavity temperature  𝑇cav lies in the 50-120 mK range across
broad  regions  of  parameter  space,  and  the  effective  cooling  time  constants  fall  in  the  10-100 𝜇s  band,
consistent  with  the  𝒪(𝑅𝜙2)scaling  of  repeated-interaction  refrigerators  and  with  the  coherence  times
accessible in high-Q 3D cavities.

The multimode nature of 3D cavities further expands the utility of these reset mechanisms. When a
second cavity mode (Mode B) is used as a memory or as an interface to a semiconductor quantum dot, it
can be cooled opportunistically, only when it stores no  quantum information, by temporarily activating a
parametric  beam-splitter  interaction  𝑔𝐴𝐵(𝑡)  that  swaps  excitations  with  the  refrigerated  Mode  A.  Recent
demonstrations  of  high-isolation,  high-fidelity  parametric  mode  converters,  including  SNAIL-based  and
parity-protected  devices,  confirm  that  such  inter-cavity  beam-splitters  can  be  activated  with  large  on-off
ratios and low added loss.37,38 In this “cool-before-load’’ protocol, entropy flows from Mode B into Mode A
and then into the engineered two-qubit reservoir through the reset channel, leaving Mode B cold and ready
to store quantum information or couple to an external device.

Direct  algorithmic  cooling  of  a  harmonic  oscillator,  by  contrast,  remains  fundamentally  difficult.
Because of the cavity’s infinite-dimensional Hilbert space, removing population from arbitrarily high Fock
states  would
require  non-Gaussian,  number-resolved  operations  or  photon-number-specific
measurements  that  are  experimentally  demanding  even  in  strongly  dispersive  cQED  architectures.  The
literature  on  heat-bath  algorithmic  cooling  (HBAC)  targets  finite-dimensional  registers  and  consistently
relies on rapidly resettable ancilla qubits as the fundamental resource for entropy compression.12,47 General
constraints on Gaussian thermal operations in bosonic systems reinforce this limitation, as achieving very
low  effective  temperatures  requires  either  non-Gaussian  resources  or  auxiliary  systems  that  can  be
efficiently reset, returning us to the two-qubit reservoir as the minimal realistic module for cooling a cavity
mode.30,47,49,53,54

Taken  together,  these  developments  establish  that  the  two  superconducting  qubits  used  in  our
refrigerator can reliably be reset to effective temperatures in the tens of millikelvin range at rates exceeding
those  required  for  the  repeated-interaction  dynamics.  They  therefore  provide  exactly  the  type  of  low-
entropy, rapidly recyclable ancilla needed for enforcing a cold detailed balance on a cavity mode situated
within a high-power 1 K environment.

5.  Outlook

The  experimental  analogue  of  the  theoretical  “two-atom  refrigerator”  is  a  pair  of  actively  reset
superconducting qubits that repeatedly interact with a high-Q 3D microwave cavity mode. In this picture
the  qubits  function  as  a  sacrificial,  low-entropy  reservoir  maintained  far  below  the  ambient  cryostat
temperature by fast dissipative or measurement-based resets; their repeated, weak, and detuned-tunable
collisions impose an effective detailed balance on the cavity, cooling it to tens of millikelvin even when the
cryostat  itself  operates  near  1  K.  This  is  precisely  the  regime  in  which  1  K  platforms  provide  abundant
cooling power, hundreds of milliwatts to the watt class, so the entropy removal required for frequent qubit
resets  can  be  supplied  without  stressing  the  base-plate  budget,  while  the  cavity  experiences  a  tailored
quantum  reservoir  rather  than  the  warm  phonon  bath.  The  repeated-interaction  framework  gives  a
microscopically  controlled  route  from  discrete  ancilla  shots  to  a  Lindbladian  for  the  cavity,  with  the
micromaser  literature  and  modern  collision-model  theory  offering  both  the  conceptual  and  quantitative
underpinnings for this mapping.14,6

Near-term milestones are within reach. Fast, high-fidelity qubit resets at the 102-ns scale, via Purcell-
enhanced  dissipation,  parametric  channels,  or  autonomous  reservoir  engineering,  are  now  routine,  and
autonomous  entanglement  stabilization  of  two-qubit  manifolds  has  been  demonstrated  in  cavity-  and

25

waveguide-based settings. In our architecture these elements provide the “reset and reprepare” primitive
that clamps the ancilla pair to a prescribed mixed state with controllable correlations, which in turn fixes the
upward/downward  coefficients  in  the  collision  map  and  therefore  the  cavity’s  steady-state  temperature.
Experimentally, one can validate the cooling by cavity thermometry (e.g., sideband spectroscopy on a weak
probe qubit or mode) and by measuring the ancilla energy flow during cycles; both readouts are compatible
with MHz repetition and small-angle exchanges.32,46

Once  established,  a  precooled  cavity  enables  device-level  payoffs.  Gate-defined  semiconductor
double quantum dots (DQDs)50 have reached strong coupling with microwave cavities in III-V and Si/SiGe
platforms;  in  such  devices  a  colder  cavity  directly  suppresses  thermal  photon  occupation  at  the  qubit
transition  and  reduces  phonon-assisted  leakage  to  unwanted  orbitals,  improving  spin-qubit  initialization
and charge-noise resilience at elevated cryostat temperatures. A second, complementary pathway is inter-
cavity  refrigeration:  designate  one  cavity  mode  as  the  refrigeration  interface  and  use  a  programmable
beam-splitter to swap entropy from a memory or DQD-coupling mode when it is empty (“cool-before-load”),
a  control  primitive  already  standard  in  bosonic-code  experiments.  Together  these  uses  make  the
refrigerated cavity a versatile low-entropy resource embedded in a warm 1 K environment.35,36,39

From  a  systems  perspective,  the  architecture  addresses  a  central  scalability  bottleneck:  millikelvin
stages offer only tens of microwatts at the mixing chamber, whereas 1 K stages can sustain two to three
orders of magnitude more power for control electronics, pumps, and fast flux drives. Relocating control to
1 K while generating local sub–100-mK pockets via qubit-engineered reservoirs provides a realistic path
toward larger channel counts without saturating the base plate. Theoretical control is aided by the detuning-
aware  collision  model:  the  same  small-angle  interaction  can  be  spectrally  gated  by  𝐿(Δ),  allowing
robustness to drift and deliberate trade-offs between interaction time and overlap bandwidth, while retaining
a closed-form birth-death description for the cavity photon number. This separation (thermodynamics in the
ancilla state, geometry in the two-atom enhancement, and spectroscopy in the detuning filter) offers clear
handles for calibration and stability in the lab.2,7

Open technical challenges frame the near-term research agenda. Heat loads from rapid qubit resets
and control pulses must be budgeted to avoid excess quasiparticles and stray photons; high-isolation, low-
loss switching between “engage” and “idle” configurations is needed to maintain the small-angle regime;
and Poissonization of cycle timing should be verified so the coarse-grained Lindblad limit applies. Multi-
mode crowding in 3D enclosures can be turned from a nuisance into a feature by using engineered filters
and parametric couplers to direct entropy flow, but this requires careful microwave design. On the metrology
side,  QND  tools  from  circuit  QED,  micromaser-inspired  counting  statistics  and  modern  bosonic-code
diagnostics, can be repurposed to quantify the effective temperature and detailed balance imposed by the
ancilla reservoir. Finally, integrating this refrigerator with error-corrected bosonic encodings (cat, binomial,
GKP) suggests a route to hardware that both cools and stabilizes logical states at the 1 K tier, leveraging
the same reservoir-engineering toolbox demonstrated in recent autonomous QEC experiments.14,34

In sum, by combining the high-power 1 K environment with a two-qubit, resettable working medium,
the proposed refrigerator creates localized low-entropy regions, 𝑇cav in the 50-120 mK range, without relying
on the dilution refrigerator’s base plate. The collision-model derivation, the experimental maturity of fast
reset  and  autonomous  stabilization,  and
to  semiconductor
nanostructures collectively support a credible path to large-scale, higher-temperature quantum hardware
in which refrigeration is delivered where and when it is needed, rather than being globally imposed by the
coldest stage.

the  demonstrated  strong  coupling

Conclusion

We have developed a comprehensive theoretical framework for cooling a phonon-tethered microwave
cavity using a stream of correlated two-level systems, including realistic phonon environments, detuning
filters, finite cavity decay, and both single- and double-atom coupling configurations. By deriving explicit
Lindblad-rate equations and analytically tractable steady-state solutions for the cavity photon number, we
have  shown  how  the  interplay  of  weak  collisions,  intra-pair  correlations,  detuning,  and  phonon  loss
determines the nonequilibrium temperature imposed on the cavity. Our results demonstrate that while a
single  interacting  atom  in  each  correlated  pair  can  partially  suppress  phonon-induced  heating,  genuine
refrigeration,  where  the  cavity  is  driven  below  the  temperature  of  the  atomic  reservoir,  requires  the

26

collective interaction of both atoms in the pair. This two-atom configuration harnesses exchange-enhanced
correlations and coherence to reshape the effective detailed balance seen by the cavity, producing cooling
minima as deep as 𝑇cav/𝑇atom ≈ 0.5 under realistic coupling strengths and spectral conditions.

Systematic evaluation of the analytic steady-state expressions across a wide parameter space reveals
the operational boundaries of this quantum refrigerator, including its sensitivity to detuning, the crucial role
of  the  cavity–bath  leakage  rate,  and  the  enhancement  provided  by  strong  atom–cavity  coupling.  The
resulting cooling landscapes identify broad regions in which the engineered atomic reservoir can overcome
environmental heating and impose a colder nonequilibrium steady state. By comparing the one-atom and
two-atom  models,  we  have clarified the  qualitative advantages of collective coupling, including stronger
refrigeration, wider detuning tolerance, and reduced sensitivity to asymmetries in the thermal weights of
the incoming states.

Motivated  by  these  theoretical  insights,  we  have  mapped  the  two-atom  cooling  mechanism  onto  a
practical cQED architecture. In this platform, two superconducting qubits serve as the working medium, a
high-Q  3D  cavity  provides  the  target  bosonic  mode,  and  fast  dissipative  or  measurement-based  resets
maintain the qubit pair at an effective temperature far below that of a 1 K cryogenic stage. This experimental
mapping connects our theoretical model directly to state-of-the-art techniques in cQED, including Purcell-
enhanced  reset,  parametric  sideband  interactions,  tunable  couplers,  and  intercavity  beam-splitters.  The
combined  system  offers  a  realistic  pathway  for  creating  local  sub–100-mK  cold  spots  inside  a  warm
enclosure,  potentially  enabling  scalable  quantum  information  architectures  in  which  most  control
electronics  reside  at  high-power  cryogenic  stages  while  only  selected  modes  or  devices  are  actively
refrigerated via engineered quantum reservoirs.

Overall, this work bridges quantum thermodynamics, collision-model open-system theory, and modern
cQED hardware to propose a feasible route toward autonomous cavity cooling far below ambient cryogenic
temperatures.  The  enhanced  performance  of  the  two-atom  reservoir  underscores  the  central  role  of
quantum correlations in nonequilibrium quantum machines and opens opportunities for integrating similar
cooling modules with semiconductor quantum dots, spin qubits, bosonic memories, and other temperature-
sensitive  quantum  devices.  Future  work  may  extend  this  framework  to  driven  dissipative  steady-state
engineering,  multi-mode  refrigeration,  and  autonomous  thermal  management  in  large-scale  quantum
processors.

Acknowledgement

This study is partially based on work supported by AFOSR and LPS under contract numbers FA9550-

23-1-0302 and FA9550-23-1-0763, and by the NSF under grant number CBET-2110603.

References

1 Bluefors, “Dilution Refrigerator Measurement System”, accessed November 28, 2025,
https://bluefors.com/products/dilution-refrigerator-measurement-systems/ld-dilution-refrigerator-
measurement-system/.
2 Bluefors, LDHe Measurement System, accessed November 28, 2025, https://bluefors.com/products/1k-
systems/ldhe-measurement-system/.
3 Uhlig, Kurt. "Dry dilution refrigerator with 4He-1 K-loop." Cryogenics 66 (2015): 6-12.
4 Bluefors, “Introducing the XLDHe High Power System - Ultimate Cooling for 1 K Experiments,” accessed
November 28, 2025, https://bluefors.com/news/introducing-the-xldhe-high-power-system-ultimate-cooling-
for-1-k-experiments/
5 Bluefors, “XLDHe High Power System,” accessed November 28, 2025,
https://bluefors.com/products/1k-systems/xldhe-high-power-system
6 Bluefors, “Introducing the Ultra-Compact Dilution Refrigerator System”, accessed November 28, 2025,
https://bluefors.com/news/introducing-the-ultra-compact-dilution-refrigerator-system
7  Bluefors,  “LD  Dilution  Refrigerator  Measurement  System”,  ,  accessed  November  28,  2025,
https://bluefors.com/products/dilution-refrigerator-measurement-systems/ld-dilution-refrigerator-
measurement-system

27

8 Krantz, Philip, Morten Kjaergaard, Fei Yan, Terry P. Orlando, Simon Gustavsson, and William D. Oliver.
"A quantum engineer's guide to superconducting qubits." Applied physics reviews 6, no. 2 (2019).
9  Blais,  Alexandre,  Arne  L.  Grimsmo,  Steven  M.  Girvin,  and  Andreas  Wallraff.  "Circuit  quantum
electrodynamics." Reviews of Modern Physics 93, no. 2 (2021): 025005.
10 Dillenschneider, Raoul, and Eric Lutz. "Energetics of quantum correlations." Europhysics Letters 88, no.
5 (2009): 50003.
11 Gardiner, Crispin W., and Matthew J. Collett. "Input and output in damped quantum systems: Quantum
stochastic differential equations and the master equation." Physical Review A 31, no. 6 (1985): 3761.
12  Ciccarello,  Francesco,  Salvatore  Lorenzo,  Vittorio  Giovannetti,  and  G.  Massimo  Palma.  "Quantum
collision models: Open system dynamics from repeated interactions." Physics Reports 954 (2022): 1-70.
13 Clerk, Aashish A. "Quantum noise and quantum measurement." Quantum Machines: Measurement and
Control of Engineered Quantum Systems (2008).
14  Filipowicz,  P.,  Javanainen,  J.,  &  Meystre,  P.,  Theory  of  a  microscopic  maser,  Phys.  Rev. A  34,  3077
(1986)
15  Linden,  Noah,  Sandu  Popescu,  and  Paul  Skrzypczyk.  "How  small  can  thermal  machines  be?  The
smallest possible refrigerator." Physical review letters 105, no. 13 (2010): 130401.
16  Correa,  Luis A.,  José  P.  Palao,  Daniel Alonso,  and  Gerardo Adesso.  "Quantum-enhanced  absorption
refrigerators." Scientific reports 4, no. 1 (2014): 3949.
17 Bhandari, Bibek, and Andrew N. Jordan. "Minimal two-body quantum absorption refrigerator." Physical
Review B 104, no. 7 (2021): 075442.
18 Englert, Berthold-Georg. "Elements of micromaser physics." arXiv preprint quant-ph/0203052 (2002).
19 Walls, D. F., and Gerard J. Milburn. "Quantum Optics and Quantum Foundations." In Quantum Optics,
pp. 225-244. Cham: Springer Nature Switzerland, 2025.
20 Forn-Díaz, P., L. Lamata, E. Rico, J. Kono, and E. Solano. "Ultrastrong coupling regimes of light-matter
interaction." Reviews of Modern Physics 91, no. 2 (2019): 025005.
21 Mundada, Pranav, Gengyan Zhang, Thomas Hazard, and Andrew Houck. "Suppression of qubit crosstalk
in a tunable coupling superconducting circuit." Physical Review Applied 12, no. 5 (2019): 054023.
22 Campbell, Daniel L., Archana Kamal, Leonardo Ranzani, Michael Senatore, and Matthew D. LaHaye.
"Modular tunable coupler for superconducting circuits." Physical Review Applied 19, no. 6 (2023): 064043.
23 Magnard, Paul, Philipp Kurpiers, Baptiste Royer, Theo Walter, J-C. Besse, Simone Gasparinetti, Marek
Pechal  et  al.  "Fast  and  unconditional  all-microwave  reset  of  a  superconducting  qubit." Physical  review
letters 121, no. 6 (2018): 060502.
24 Ristè, D., C. C. Bultink, Konrad W. Lehnert, and L. DiCarlo. "Feedback control of a solid-state qubit using
high-fidelity projective measurement." Physical review letters 109, no. 24 (2012): 240502.
25 Shankar, S., M. Hatridge, Z. Leghtas, K. M. Sliwa, A. Narla, U. Vool, S. M. Girvin, L. Frunzio, M. Mirrahimi,
and M. H. Devoret. "Stabilizing entanglement autonomously between two superconducting qubits." arXiv
preprint arXiv:1307.4349 (2013).
26 Gu, Xu-Yang, Da'er Feng, Zhen-Yu Peng, Gui-Han Liang, Yang He, Yongxi Xiao, Ming-Chuan Wang et
al.  "Engineering  a Multi-Mode Purcell Filter for Superconducting-Qubit  Reset and Readout with Intrinsic
Purcell Protection." arXiv preprint arXiv:2507.04676 (2025).
27 Sunada, Yoshiki, Shingo Kono, Jesper Ilves, Shuhei Tamate, Takanori Sugiyama, Yutaka Tabuchi, and
Yasunobu Nakamura. "Fast readout and reset of a superconducting qubit coupled to a resonator with an
intrinsic Purcell filter." Physical Review Applied 17, no. 4 (2022): 044016.
28  Ding,  Jiayu,  Yulong  Li,  He  Wang,  Guangming  Xue,  Tang  Su,  Chenlu  Wang,  Weijie  Sun  et  al.
"Multipurpose  architecture  for  fast  reset  and  protective  readout  of  superconducting  qubits." Physical
Review Applied 23, no. 1 (2025): 014012.
29  Reed,  Matthew  D.,  Blake  R.  Johnson, Andrew A.  Houck,  Leonardo  DiCarlo,  Jerry  M.  Chow,  David  I.
Schuster, Luigi Frunzio, and Robert J. Schoelkopf. "Fast reset and suppressing spontaneous emission of
a superconducting qubit." Applied Physics Letters 96, no. 20 (2010).
30  Alhambra,  Álvaro  M.,  Matteo  Lostaglio,  and  Christopher  Perry.  "Heat-bath  algorithmic  cooling  with
optimal thermalization strategies." Quantum 3 (2019): 188.
31 Zhou, Yu, Zhenxing Zhang, Zelong Yin, Sainan Huai, Xiu Gu, Xiong Xu, Jonathan Allcock et al. "Rapid
tunable  superconducting  qubits." Nature
and  unconditional  parametric
Communications 12, no. 1 (2021): 5924.

reset  protocol

for

28

32  Shah,  Parth  S.,  Frank  Yang,  Chaitali  Joshi,  and  Mohammad  Mirhosseini.  "Stabilizing  remote
entanglement via waveguide dissipation." PRX Quantum 5, no. 3 (2024): 030346.
33  Li,  Ziqian,  Tanay  Roy,  Yao  Lu,  Eliot  Kapit,  and  David  I.  Schuster.  "Autonomous  stabilization  with
programmable stabilized state." Nature Communications 15, no. 1 (2024): 6978.
34  Lachance-Quirion,  Dany,  Marc-Antoine  Lemonde,  Jean  Olivier  Simoneau,  Lucas  St-Jean,  Pascal
Lemieux, Sara Turcotte, Wyatt Wright et al. "Autonomous quantum error correction of Gottesman-Kitaev-
Preskill states." Physical Review Letters 132, no. 15 (2024): 150607.
35  Stockklauser,  Anna,  Pasquale  Scarlino,  Jonne  V.  Koski,  Simone  Gasparinetti,  Christian  Kraglund
Andersen,  Christian  Reichl,  Werner  Wegscheider,  Thomas  Ihn,  Klaus  Ensslin,  and  Andreas  Wallraff.
"Strong  coupling  cavity  QED  with  gate-defined  double  quantum  dots  enabled  by  a  high  impedance
resonator." Physical Review X 7, no. 1 (2017): 011030.
36 Mi, Xiao, J. V. Cady, D. M. Zajac, P. W. Deelman, and Jason R. Petta. "Strong coupling of a single electron
in silicon to a microwave photon." Science 355, no. 6321 (2017): 156-158.
37 Sirois, Adam J., M. A. Castellanos-Beltran, M. P. DeFeo, L. Ranzani, F. Lecocq, R. W. Simmonds, J. D.
Teufel, and J. Aumentado. "Coherent-state storage and retrieval between superconducting cavities using
parametric frequency conversion." Applied Physics Letters 106, no. 17 (2015).
38 Chapman, Benjamin J., Stijn J. De Graaf, Sophia H. Xue, Yaxing Zhang, James Teoh, Jacob C. Curtis,
Takahiro  Tsunoda  et  al.  "High-on-off-ratio  beam-splitter  interaction  for  gates  on  bosonically  encoded
qubits." PRX Quantum 4, no. 2 (2023): 020355.
39  Cai,  Weizhou,  Yuwei  Ma,  Weiting  Wang,  Chang-Ling  Zou,  and  Luyan  Sun.  "Bosonic  quantum  error
correction codes in superconducting quantum circuits." Fundamental Research 1, no. 1 (2021): 50-67.
40  Underwood,  Devin,  Joseph A.  Glick,  Ken  Inoue,  David  J.  Frank,  John  Timmerwilke,  Emily  Pritchett,
Sudipto  Chakraborty  et  al.  "Using  cryogenic  CMOS  control  electronics  to  enable  a  two-qubit  cross-
resonance gate." PRX Quantum 5, no. 1 (2024): 010326.
41  Pellerano,  Stefano,  Sushil  Subramanian,  Jong-Seok  Park,  Bishnu  Patra,  Todor  Mladenov,  Xiao  Xue,
Lieven MK Vandersypen, Masoud Babaie, Edoardo Charbon, and Fabio Sebastiano. "Cryogenic CMOS
for  qubit  control  and  readout."  In 2022  IEEE  Custom  Integrated  Circuits  Conference  (CICC),  pp.  01-08.
IEEE, 2022.
42 Gross, Michel, and Serge Haroche. "Superradiance: An essay on the theory of collective spontaneous
emission." Physics reports 93, no. 5 (1982): 301-396.
43 Dicke, Robert H. "Coherence in spontaneous radiation processes." Physical review 93, no. 1 (1954): 99.
44 Poyatos, J. F., J. Ignacio Cirac, and P. Zoller. "Quantum reservoir engineering with laser cooled trapped
ions." Physical review letters 77, no. 23 (1996): 4728.
45 Poyatos, J. F., J. Ignacio Cirac, and P. Zoller. "Quantum reservoir engineering with laser cooled trapped
ions." Physical review letters 77, no. 23 (1996): 4728.
46  Ding,  Leon,  Max  Hays,  Youngkyu  Sung,  Bharath  Kannan,  Junyoung An, Agustin  Di  Paolo, Amir  H.
Karamlou  et  al.  "High-fidelity,
transmon
frequency-flexible
coupler." Physical Review X 13, no. 3 (2023): 031035.
47 Gargiulo, O., S. Oleschko, J. Prat-Camps, M. Zanner, and G. Kirchmair. "Fast flux control of 3D transmon
qubits using a magnetic hose." Applied Physics Letters 118, no. 1 (2021).
48 Han, Lian-Chen, Yu Xu, Jin Lin, Fu-Sheng Chen, Shao-Wei Li, Cheng Guo, Na Li et al. "Active reset of
superconducting qubits using the electronics based on RF switches." AIP Advances 13, no. 9 (2023).
49 Serafini, A., M. Lostaglio, S. Longden, U. Shackerley-Bennett, C-Y. Hsieh, and G. Adesso. "Gaussian
thermal  operations  and  the  limits  of  algorithmic  cooling." Physical  Review  Letters 124,  no.  1  (2020):
010602.
50  Vashaee,  Daryoosh,  and  Jahanfar  Abouie.  "Microwave-induced  cooling  in  double  quantum  dots:
Achieving millikelvin temperatures to reduce thermal noise around spin qubits." Physical Review B 111, no.
1 (2025): 014305.

fluxonium  gates  with  a

two-qubit

29

