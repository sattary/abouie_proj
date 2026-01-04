"""
Physics engine for cavity-qubit cooling.

Core modules:
- operators: Static quantum operators
- hamiltonian: Time-dependent Hamiltonian with dual control
- lindblad: Full Lindblad master equation
- initial_states: Quantum state preparation
"""

from .operators import (
    SystemParams,
    Operators,
    build_operators,
    thermal_occupation,
)

from .hamiltonian import (
    build_hamiltonian_func,
    build_hamiltonian_from_arrays,
    check_floquet_condition,
)

from .lindblad import (
    DissipationRates,
    compute_dissipation_rates,
    build_lindblad_superoperator,
    build_master_equation,
    build_master_equation_real,
    simulate,
)

from .initial_states import (
    thermal_state,
    thermal_cavity_ground_qubits,
    cold_cavity_ground_qubits,
    hot_cavity_ground_qubits,
    correlated_qubit_pair,
)

__all__ = [
    # Parameters and operators
    "SystemParams",
    "Operators",
    "build_operators",
    "thermal_occupation",
    # Hamiltonian
    "build_hamiltonian_func",
    "build_hamiltonian_from_arrays",
    "check_floquet_condition",
    # Lindblad
    "DissipationRates",
    "compute_dissipation_rates",
    "build_lindblad_superoperator",
    "build_master_equation",
    "build_master_equation_real",
    "simulate",
    # Initial states
    "thermal_state",
    "thermal_cavity_ground_qubits",
    "cold_cavity_ground_qubits",
    "hot_cavity_ground_qubits",
    "correlated_qubit_pair",
]
