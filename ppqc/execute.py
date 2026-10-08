"""Run a compiled PPM program on Clifft, one magic state per pi/8 rotation.

The pi/8 rotation gadget (Litinski Fig. 7) for a rotation about P:

  1. measure P x (joint axis of the magic state)   -> outcome m
  2. if m = 1, a Clifford correction P_{pi/4} is owed
  3. measure X on the magic state                  -> outcome x
  4. if x = 1, a Pauli correction P is owed

Neither correction is applied. A Pauli correction is a frame update: it
flips the sign of every later operator that anticommutes with P, and so is
a parity of measurement records. A Clifford correction changes *which*
Pauli products are measured later, which a compile-once sampler cannot do
mid-shot. So each of the 2**t patterns of m outcomes is compiled as its own
circuit, post-selected on that pattern. The accepted shots of all branches
together are exactly the shots of the adaptive protocol.
"""
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Sequence, Set, Tuple

import clifft
import numpy as np
import stim

from .compiler import PPMProgram
from .magic import Emitter, MagicSource, MagicState


@dataclass
class _Op:
    pauli: stim.PauliString  # static signed Pauli
    frame: Set[MagicState]   # magic states whose x outcome flips the sign


@dataclass
class RunResult:
    distribution: np.ndarray  # over outcomes, indexed by sum(bit_i << i)
    accepted: int
    acceptance: float         # accepted shots per attempted run of the protocol


def _targets(p: stim.PauliString) -> str:
    return "*".join(f"{'_XYZ'[p[q]]}{q}" for q in range(len(p)) if p[q])


def _negative(p: stim.PauliString) -> int:
    return int(p.sign.real < 0)


def build_branch_circuit(program: PPMProgram, branch: Sequence[int],
                         source: MagicSource) -> Tuple[str, List[int], List[int]]:
    """Circuit for one pattern ``branch`` of Clifford corrections.

    Returns (clifft text, expected detector parities, observable flips).
    Every detector is a post-selection check; observable i, XORed with
    flips[i], is the outcome of the original Z measurement of qubit i.
    """
    n = program.num_qubits
    em = Emitter()
    em.add("R " + " ".join(map(str, range(n))))
    ops = [_Op(p, set()) for p in program.rotations + program.measurements]
    num_rot = len(program.rotations)
    # (records, frame, expected parity) of each decoded joint outcome m.
    checks: List[Tuple[Set[int], Set[MagicState], int]] = []

    for j in range(num_rot):
        op = ops[j]
        state = source.request(em)
        recs = state.joint(em, _targets(op.pauli))
        checks.append((recs, set(op.frame), branch[j] ^ _negative(op.pauli)))
        for later in ops[j + 1:]:
            if later.pauli.commutes(op.pauli):
                continue
            if branch[j]:
                later.pauli = 1j * op.pauli * later.pauli
                later.frame ^= op.frame
            later.frame ^= {state}
    source.flush(em)
    outputs = [set(em.measure(f"MPP {_targets(op.pauli)}")) for op in ops[num_rot:]]

    def line(kind: str, recs: Set[int], frame: Set[MagicState]) -> str:
        recs = set(recs)
        for state in frame:
            recs ^= state.x_recs
        return kind + " " + " ".join(em.rec(r) for r in sorted(recs))

    expected = []
    for recs in source.post_selected:
        em.add("DETECTOR " + " ".join(em.rec(r) for r in sorted(recs)))
        expected.append(0)
    for recs, frame, value in checks:
        em.add(line("DETECTOR", recs, frame))
        expected.append(value)
    flips = []
    for i, op in enumerate(ops[num_rot:]):
        em.add(line(f"OBSERVABLE_INCLUDE({i})", outputs[i], op.frame))
        flips.append(_negative(op.pauli))
    return "\n".join(em.lines), expected, flips


def run(program: PPMProgram, make_source: Callable[[], MagicSource], shots_per_branch: int,
        seed: int = 0, threads="auto") -> RunResult:
    """Sample every Clifford-correction branch and pool the accepted shots."""
    n = program.num_qubits
    counts = np.zeros(2 ** n, dtype=np.int64)
    weights = 1 << np.arange(n)
    branches = list(product((0, 1), repeat=len(program.rotations)))
    for k, branch in enumerate(branches):
        text, expected, flips = build_branch_circuit(program, branch, make_source())
        compiled = clifft.compile(text, postselection_mask=[1] * len(expected),
                                  expected_detectors=expected)
        result = clifft.sample_survivors(compiled, shots_per_branch, seed=seed + k,
                                         keep_records=True, threads=threads)
        bits = np.asarray(result.observables, dtype=np.int64) ^ np.asarray(flips, dtype=np.int64)
        counts += np.bincount(bits @ weights, minlength=2 ** n)
    accepted = int(counts.sum())
    return RunResult(counts / max(accepted, 1), accepted, accepted / shots_per_branch)
