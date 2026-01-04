# Phase 2: Robustness & Extensions

## Priority Actions (Address Reviewer Concerns)

### P1: Critical Fixes (Must-Do Before PRL)

- [ ] **RWA Validation for Best Results**

  - Run GRAPE optimal pulses through Tier 2 (full TDSE)
  - If >10% discrepancy, re-optimize with lower g_max

- [ ] **Explicit Qubit Reset**

  - Add reset-to-ground between Floquet cycles
  - Or document that continuous thermalization is the model

- [ ] **Resource-Fair Comparison**
  - Define control budget: time-averaged |g(t)|^2 + |Delta(t)|^2
  - Show Floquet wins under equal resource constraints

### P2: Strengthen Claims

- [ ] **Finite-Size Convergence Test**

  - Increase n_fock from 5 to 10, 15, 20
  - Confirm results are converged

- [ ] **Improved Noise Model**
  - Implement proper 1/f (pink) noise spectrum
  - Add resonant TLS defects

### P3: New Physics (Extensions)

- [ ] **Analytic Floquet Master Equation**

  - Derive H_eff via Magnus expansion
  - Prove non-commutativity is necessary analytically

- [ ] **Thermodynamic Work Accounting**
  - Define work input from control fields
  - Compute true COP with work costs

---

## Breakthrough Ideas (Future Papers)

### Idea 1: Floquet Refrigerator Paper

- Frame as quantum thermal machine
- Compute Carnot-like efficiency bounds
- Compare autonomous vs driven machines

### Idea 2: Experimental Proposal

- Realistic IBM/Google transmon parameters
- AWG-compatible pulse sequences
- Measurable signatures

### Idea 3: Multi-Mode Extension

- Two cavity modes
- Cool one using other as work reservoir

### Idea 4: ML for Control Discovery

- Neural network predicts optimal pulses
- Generalize across parameter space

---

## File Changes Needed

| File                            | Change                                    |
| ------------------------------- | ----------------------------------------- |
| `src/floquet/cycle.py`          | Add optional qubit reset step             |
| `src/validation/tier2.py`       | Add function to validate GRAPE results    |
| `src/validation/tier3.py`       | Improve 1/f noise to proper pink spectrum |
| `src/analysis/resource_fair.py` | NEW: Resource-constrained comparison      |
| `src/physics/operators.py`      | Add n_fock parameter scaling test         |
| `docs/project_guide.md`         | Document all improvements                 |
