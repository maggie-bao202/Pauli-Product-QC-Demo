"""Pauli product quantum computation demo.

Clifford+T circuit -> pi/8 Pauli product rotations -> Pauli product
measurements, with each rotation consuming one magic state.
"""
from .compiler import PPMProgram, Rotation, compile_to_ppm, cnot, controlled, h, rot, s, t, t_layers
from .circuits import litinski_fig4
from .execute import RunResult, build_branch_circuit, run
from .magic import BareMagic, H6Magic
from .reference import circuit_distribution, ppm_distribution, total_variation

__all__ = [
    "PPMProgram", "Rotation", "compile_to_ppm", "cnot", "controlled", "h", "rot", "s", "t",
    "t_layers", "litinski_fig4", "RunResult", "build_branch_circuit", "run", "BareMagic",
    "H6Magic", "circuit_distribution", "ppm_distribution", "total_variation",
]
