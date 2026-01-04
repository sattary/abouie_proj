# Analysis modules
from .thermodynamics import (
    compute_cooling_power,
    compute_cop,
    compute_entropy_efficiency,
    ThermodynamicMetrics,
)
from .figures import (
    generate_figure_1_comparison,
    generate_figure_2_waveforms,
    generate_figure_3_nogo_sweep,
)

__all__ = [
    "compute_cooling_power",
    "compute_cop",
    "compute_entropy_efficiency",
    "ThermodynamicMetrics",
    "print_thermodynamic_report",
    "analyze_grape_result",
]
