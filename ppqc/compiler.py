"""Clifford+T circuits as Pauli product rotations, following Litinski,
"A Game of Surface Codes" (arXiv:1808.02892), Sec. 1.

A rotation is P_phi = exp(-i * phi * P). pi/8 rotations are the T-type
gates, pi/4 rotations are Clifford, pi/2 rotations are Paulis.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence

import stim

PI_8 = Fraction(1, 8)
PI_4 = Fraction(1, 4)
PI_2 = Fraction(1, 2)


@dataclass(frozen=True)
class Rotation:
    """exp(-i * angle * pi * pauli). ``angle`` is 1/8, 1/4 or 1/2; a
    negative angle is stored as a sign on ``pauli``."""
    pauli: stim.PauliString
    angle: Fraction


@dataclass(frozen=True)
class PPMProgram:
    """``rotations`` are signed pi/8 Pauli product rotations, in time order.
    ``measurements[i]`` is the signed Pauli product whose outcome is the
    Z measurement of qubit i in the original circuit."""
    num_qubits: int
    rotations: List[stim.PauliString]
    measurements: List[stim.PauliString]


def rot(pauli: str, angle, num_qubits: int) -> Rotation:
    """``rot("X0*Z2", "-1/4", 4)`` is (X on qubit 0, Z on qubit 2) at -pi/4."""
    angle = Fraction(angle)
    p = stim.PauliString(num_qubits)
    for term in pauli.split("*"):
        p[int(term[1:])] = term[0]
    if angle < 0:
        p, angle = -p, -angle
    if angle not in (PI_8, PI_4, PI_2):
        raise ValueError(f"angle must be +-1/8, 1/4 or 1/2 (units of pi); got {angle}.")
    return Rotation(p, angle)


def t(q: int, n: int) -> List[Rotation]:
    return [rot(f"Z{q}", PI_8, n)]


def s(q: int, n: int) -> List[Rotation]:
    return [rot(f"Z{q}", PI_4, n)]


def h(q: int, n: int) -> List[Rotation]:
    return [rot(f"Z{q}", PI_4, n), rot(f"X{q}", PI_4, n), rot(f"Z{q}", PI_4, n)]


def controlled(p1: str, p2: str, n: int) -> List[Rotation]:
    """C(P1, P2) = (P1 P2)_{pi/4} (P1)_{-pi/4} (P2)_{-pi/4}  (Litinski Fig. 5c)."""
    return [rot(f"{p1}*{p2}", PI_4, n), rot(p1, -PI_4, n), rot(p2, -PI_4, n)]


def cnot(control: int, target: int, n: int) -> List[Rotation]:
    return controlled(f"Z{control}", f"X{target}", n)


def _push_past(clifford: Rotation, pauli: stim.PauliString) -> stim.PauliString:
    """Move ``clifford`` (earlier in time) to the right of an operation about
    ``pauli``, and return what that operation becomes (Litinski Fig. 4a/c)."""
    if clifford.pauli.commutes(pauli):
        return pauli
    if clifford.angle == PI_2:
        return -pauli
    return 1j * clifford.pauli * pauli


def compile_to_ppm(circuit: Sequence[Rotation], num_qubits: int) -> PPMProgram:
    """Commute every Clifford to the end of ``circuit`` and absorb it into
    the final Z measurements of all qubits."""
    cliffords: List[Rotation] = []
    rotations: List[stim.PauliString] = []
    for op in circuit:
        if op.angle != PI_8:
            cliffords.append(op)
            continue
        p = op.pauli
        for c in reversed(cliffords):
            p = _push_past(c, p)
        rotations.append(p)
    measurements = []
    for q in range(num_qubits):
        p = stim.PauliString(num_qubits)
        p[q] = "Z"
        for c in reversed(cliffords):
            p = _push_past(c, p)
        measurements.append(p)
    return PPMProgram(num_qubits, rotations, measurements)


def t_layers(rotations: Sequence[stim.PauliString]) -> List[List[stim.PauliString]]:
    """Greedily group pi/8 rotations into layers of mutually commuting
    rotations (Litinski Fig. 6). The number of layers is the T depth."""
    layers: List[List[stim.PauliString]] = []
    for p in rotations:
        k = len(layers)
        while k > 0 and all(p.commutes(q) for q in layers[k - 1]):
            k -= 1
        if k == len(layers):
            layers.append([])
        layers[k].append(p)
    return layers
