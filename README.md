# Quantum Cavity Cooling

Based on: **"Quantum Correlation-Assisted Cooling of Microwave Cavities Below the Ambient Temperature"**
by Vashaee & Abouie

## Project Structure

```
src/
  physics/          # Lindblad master equation simulator
    cavity_cooling.py   - Core physics engine (cavity + 2 qubits)
  optimization/     # Pulse optimization scripts
    optimize_cavity.py      - Gradient descent pulse optimization
    optimize_cavity2.py     - Alternative optimizer
    train_neural_pulse.py   - Neural network pulse generator
papers/             # Reference papers
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run physics baseline test
python -m src.physics.cavity_cooling

# Run neural pulse optimization
python -m src.optimization.train_neural_pulse
```

## Physics

The simulator implements the Jaynes-Cummings model for a microwave cavity coupled to two superconducting qubits, with Lindblad dissipation to model cavity decay:

- **Hilbert space**: 5 Fock states x 2 qubits x 2 qubits = 20 dimensions
- **Hamiltonian**: H = g(t) \* (a^dag S- + a S+)
- **Dissipation**: D[sqrt(kappa) * a]
