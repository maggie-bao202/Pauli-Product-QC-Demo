"""Dense statevector reference, independent of the sampler. Outcome
distributions are arrays of length 2**n indexed by sum(bit_i << i), where
bit_i is the outcome of the original Z measurement of qubit i."""
from typing import Sequence

import numpy as np
import stim

from .compiler import PPMProgram, Rotation


def _matrix(p: stim.PauliString) -> np.ndarray:
    return p.to_unitary_matrix(endian="little")


def _rotate(state: np.ndarray, p: stim.PauliString, angle: float) -> np.ndarray:
    return np.cos(angle) * state - 1j * np.sin(angle) * (_matrix(p) @ state)


def _zero(n: int) -> np.ndarray:
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    return state


def circuit_distribution(circuit: Sequence[Rotation], num_qubits: int) -> np.ndarray:
    """Z-measurement distribution of the original circuit on |0...0>."""
    state = _zero(num_qubits)
    for op in circuit:
        state = _rotate(state, op.pauli, float(op.angle) * np.pi)
    return np.abs(state) ** 2


def ppm_distribution(program: PPMProgram) -> np.ndarray:
    """Outcome distribution of the compiled program: the pi/8 rotations
    applied as unitaries, then the Pauli product measurements."""
    n = program.num_qubits
    state = _zero(n)
    for p in program.rotations:
        state = _rotate(state, p, np.pi / 8)
    mats = [_matrix(m) for m in program.measurements]
    dist = np.zeros(2 ** n)
    for outcome in range(2 ** n):
        v = state
        for i, m in enumerate(mats):
            sign = -1 if (outcome >> i) & 1 else 1
            v = (v + sign * (m @ v)) / 2
        dist[outcome] = np.vdot(v, v).real
    return dist


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
