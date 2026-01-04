"""
Utilities for saving and loading experiment results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from src.floquet import FloquetCycleParams


def save_grape_results(
    history: List[float],
    optimal_cycle: FloquetCycleParams,
    config: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: str = "results",
    name: str = None,
) -> str:
    """
    Save GRAPE optimization results to JSON.

    Args:
        history: Loss history from optimization
        optimal_cycle: Optimal FloquetCycleParams
        config: GRAPEConfig as dict
        params: SystemParams as dict
        output_dir: Directory to save results
        name: Optional name for the file

    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        filename = f"{name}_{timestamp}.json"
    else:
        filename = f"grape_{timestamp}.json"

    filepath = Path(output_dir) / filename

    data = {
        "timestamp": timestamp,
        "config": {k: v for k, v in config._asdict().items()}
        if hasattr(config, "_asdict")
        else config,
        "params": {k: v for k, v in params._asdict().items()}
        if hasattr(params, "_asdict")
        else params,
        "history": [float(x) for x in history],
        "optimal_cycle": {
            "T_cycle": float(optimal_cycle.T_cycle),
            "n_steps": int(optimal_cycle.n_steps),
            "g_sequence": [float(x) for x in optimal_cycle.g_sequence],
            "delta_sequence": [float(x) for x in optimal_cycle.delta_sequence],
        },
        "final_n": float(history[-1]),
        "n_iterations": len(history),
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to {filepath}")
    return str(filepath)


def load_grape_results(filepath: str) -> Dict[str, Any]:
    """Load GRAPE results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruct FloquetCycleParams
    cycle_data = data["optimal_cycle"]
    data["optimal_cycle"] = FloquetCycleParams(
        T_cycle=cycle_data["T_cycle"],
        n_steps=cycle_data["n_steps"],
        g_sequence=np.array(cycle_data["g_sequence"]),
        delta_sequence=np.array(cycle_data["delta_sequence"]),
    )

    data["history"] = np.array(data["history"])

    return data


def save_sac_results(
    results: List[Dict],
    g_seq: np.ndarray,
    delta_seq: np.ndarray,
    output_dir: str = "results",
    name: str = None,
) -> str:
    """Save SAC training results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name or 'sac'}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    data = {
        "timestamp": timestamp,
        "results": results,
        "g_sequence": [float(x) for x in g_seq] if g_seq is not None else None,
        "delta_sequence": [float(x) for x in delta_seq]
        if delta_seq is not None
        else None,
        "best_n": min(r["n_cav"] for r in results) if results else None,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved SAC results to {filepath}")
    return str(filepath)


if __name__ == "__main__":
    # Test save/load
    from src.floquet import create_bang_bang_cycle

    cycle = create_bang_bang_cycle(T_cycle=0.5, n_steps=10)
    history = [1.5, 1.3, 1.1, 1.0]

    path = save_grape_results(
        history,
        cycle,
        config={"n_steps": 10, "n_iterations": 4},
        params={"kappa": 0.05},
        output_dir="results",
        name="test",
    )

    loaded = load_grape_results(path)
    print(f"Loaded: final_n={loaded['final_n']}")
