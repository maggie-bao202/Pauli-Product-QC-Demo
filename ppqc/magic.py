"""Magic state sources for the pi/8 rotation gadget.

A source hands out magic states one at a time. Each state knows how to
measure (data Pauli) x (its own "joint" axis), and exposes the measurement
records of its final X measurement once the source has been flushed.

Circuits are emitted as Clifft text (Stim syntax plus T / T_DAG), because
real magic states are not stabilizer states.
"""
from typing import List, Optional, Set

import stim

_NOISE_GATES = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1",
                "PAULI_CHANNEL_2")


class Emitter:
    """Accumulates circuit text and numbers measurements absolutely, so that
    detectors and observables can refer to records across spliced blocks."""

    def __init__(self):
        self.lines: List[str] = []
        self.num_measurements = 0

    def add(self, line: str) -> None:
        self.lines.append(line)

    def measure(self, line: str, count: int = 1) -> List[int]:
        self.lines.append(line)
        first = self.num_measurements
        self.num_measurements += count
        return list(range(first, first + count))

    def rec(self, index: int) -> str:
        return f"rec[{index - self.num_measurements}]"


class MagicState:
    """One magic state. ``x_recs`` is filled in when its source is flushed."""

    def __init__(self):
        self.x_recs: Optional[Set[int]] = None

    def joint(self, em: Emitter, data_term: str) -> Set[int]:
        raise NotImplementedError


class MagicSource:
    def __init__(self):
        # Records that must all have parity 0 for the shot to be accepted.
        self.post_selected: List[Set[int]] = []

    def request(self, em: Emitter) -> MagicState:
        raise NotImplementedError

    def flush(self, em: Emitter) -> None:
        """Emit any pending X measurements."""

    def _t(self, em: Emitter, qubit: int, dagger: bool = False) -> None:
        em.add(f"{'T_DAG' if dagger else 'T'} {qubit}")
        if self.p_t > 0:
            em.add(f"DEPOLARIZE1({self.p_t}) {qubit}")


class _BareState(MagicState):
    def __init__(self, qubit: int):
        super().__init__()
        self.qubit = qubit

    def joint(self, em, data_term):
        return set(em.measure(f"MPP {data_term}*Z{self.qubit}"))


class BareMagic(MagicSource):
    """Undistilled |T> = T|+> on a single physical qubit; each T gate is
    followed by single-qubit depolarizing noise of strength ``p_t``."""

    def __init__(self, num_data: int, p_t: float = 0.0):
        super().__init__()
        self.p_t = p_t
        self.qubit = num_data
        self._open: Optional[_BareState] = None

    def _last_se_round(self):
        """(start, end, first local record) of LightStim's last syndrome
        extraction round: from the syndrome reset to the syndrome measurement."""
        def on_syn(inst, name):
            return inst.name == name and {t.value for t in inst.targets_copy()} == self._syn
        end = max(i for i, inst in enumerate(self._instructions) if on_syn(inst, "M"))
        start = max(i for i, inst in enumerate(self._instructions[:end]) if on_syn(inst, "R"))
        first_rec = sum(inst.num_measurements for inst in self._instructions[:end])
        return start, end + 1, first_rec

    def _post_check_se(self, em: Emitter) -> None:
        """Repeat LightStim's SE round after the H check, post-selecting on
        each syndrome bit agreeing with the previous round. The proxy does
        not need this: its final MX readout is blind to X errors. A real
        magic state is consumed through Y_L, which is not."""
        first = self._se_first_rec
        previous = self._local_to_global[first:first + len(self._syn)]
        for _ in range(self._post_check_rounds):
            current: List[int] = []
            for inst in self._instructions[self._se_start:self._se_end]:
                current += self._emit_plain(em, inst)
            self.post_selected += [{a, b} for a, b in zip(previous, current)]
            previous = current

    def request(self, em):
        self.flush(em)
        em.add(f"RX {self.qubit}")
        self._t(em, self.qubit)
        self._open = _BareState(self.qubit)
        return self._open

    def flush(self, em):
        if self._open is not None:
            self._open.x_recs = set(em.measure(f"MX {self.qubit}"))
            self._open = None


class _H6State(MagicState):
    def __init__(self, support: List[int]):
        super().__init__()
        self.support = support

    def joint(self, em, data_term):
        # |H+> lies in the XZ plane, so the joint axis is the logical Y.
        # With Y_L = i X_L Z_L and X_L, Z_L of weight 3, -Y_L = +Y Y Y.
        return set(em.measure("MPP " + "*".join([data_term] + [f"Y{q}" for q in self.support])))


class H6Magic(MagicSource):
    """Magic states from the [[6,2,2]] H6 "0-level distillation" protocol
    (arXiv:2506.14688), built by ``lightstim.protocols.h6_distillation``.

    LightStim builds the protocol as a Clifford proxy, with |+> standing in
    for |H+> and CX for controlled-H. Here the proxy is turned back into the
    real protocol:

    - the two encoder inputs are prepared in |H+> (a noisy T gate each);
    - each CX from a Bell-check ancilla onto a data qubit becomes a
      controlled-H (two noisy T-type gates each);
    - the final transversal MX is kept: after the joint measurements it is
      exactly the X measurement that discards the two consumed magic states.

    One block yields two magic states and is post-selected on every detector
    LightStim emits. ``p_t`` is the depolarizing strength after each T gate;
    ``p`` is LightStim's circuit-level noise on the Clifford operations.
    """

    def __init__(self, num_data: int, p_t: float = 0.0, p: float = 0.0, rounds: int = 1,
                 post_check_rounds: int = 1):
        super().__init__()
        from lightstim.protocols.h6_distillation import build_h6_distillation_circuit, inject_noise

        self.p_t = p_t
        circuit, _, system = build_h6_distillation_circuit(rounds=rounds, variant="e")
        self._data = sorted(system.data_indices)
        self._aux = {q for q, owner in system.index_to_owner_map.items() if owner == "h6_check_anc"}
        self._syn = set(system.syndrome_indices) - self._aux
        self._post_check_rounds = post_check_rounds
        if p > 0:
            circuit = inject_noise(circuit, p=p)
        self._instructions = list(circuit.flattened())
        self._num_block_qubits = circuit.num_qubits
        self._split = self._readout_start()
        self._se_start, self._se_end, self._se_first_rec = self._last_se_round()
        self._base = num_data
        self._open: List[_H6State] = []
        self._handed_out = 0
        self._local_to_global: List[int] = []
        self._inputs_prepared = False

    def _readout_start(self) -> int:
        """Index of the final transversal MX, including the noise before it."""
        data = set(self._data)
        last = max(i for i, inst in enumerate(self._instructions)
                   if inst.name == "MX" and {t.value for t in inst.targets_copy()} == data)
        while last > 0 and self._instructions[last - 1].name in _NOISE_GATES:
            last -= 1
        return last

    def _last_se_round(self):
        """(start, end, first local record) of LightStim's last syndrome
        extraction round: from the syndrome reset to the syndrome measurement."""
        def on_syn(inst, name):
            return inst.name == name and {t.value for t in inst.targets_copy()} == self._syn
        end = max(i for i, inst in enumerate(self._instructions) if on_syn(inst, "M"))
        start = max(i for i, inst in enumerate(self._instructions[:end]) if on_syn(inst, "R"))
        first_rec = sum(inst.num_measurements for inst in self._instructions[:end])
        return start, end + 1, first_rec

    def _post_check_se(self, em: Emitter) -> None:
        """Repeat LightStim's SE round after the H check, post-selecting on
        each syndrome bit agreeing with the previous round. The proxy does
        not need this: its final MX readout is blind to X errors. A real
        magic state is consumed through Y_L, which is not."""
        first = self._se_first_rec
        previous = self._local_to_global[first:first + len(self._syn)]
        for _ in range(self._post_check_rounds):
            current: List[int] = []
            for inst in self._instructions[self._se_start:self._se_end]:
                current += self._emit_plain(em, inst)
            self.post_selected += [{a, b} for a, b in zip(previous, current)]
            previous = current

    def request(self, em):
        if self._handed_out == len(self._open):
            self.flush(em)
            self._local_to_global = []
            self._inputs_prepared = False
            em.add("R " + " ".join(str(self._base + q) for q in range(self._num_block_qubits)))
            self._emit(em, self._instructions[:self._split])
            self._post_check_se(em)
            b = self._base
            self._open = [_H6State([b + q for q in self._data[0::2]]),
                          _H6State([b + q for q in self._data[1::2]])]
            self._handed_out = 0
        state = self._open[self._handed_out]
        self._handed_out += 1
        return state

    def flush(self, em):
        if self._open:
            self._emit(em, self._instructions[self._split:])
            self._open = []
            self._handed_out = 0

    def _emit(self, em: Emitter, instructions) -> None:
        b = self._base
        inputs = set(self._data[:2])
        for inst in instructions:
            name = inst.name
            targets = inst.targets_copy()
            args = inst.gate_args_copy()
            if name in ("QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
                continue
            if name == "DETECTOR":
                self.post_selected.append(self._records(targets))
                continue
            if name == "OBSERVABLE_INCLUDE":
                self._open[int(args[0])].x_recs = self._records(targets)
                continue
            qubits = [t.value for t in targets]
            if name == "RX" and not self._inputs_prepared and inputs <= set(qubits):
                # Encoder inputs: |+> proxy -> |H+> = SQRT_X T |+>.
                em.add("RX " + " ".join(str(b + q) for q in qubits))
                for q in sorted(inputs):
                    self._t(em, b + q)
                    em.add(f"SQRT_X {b + q}")
                self._inputs_prepared = True
                continue
            if name == "CX" and any(c in self._aux and t not in self._aux
                                    for c, t in zip(qubits[0::2], qubits[1::2])):
                for c, t in zip(qubits[0::2], qubits[1::2]):
                    if c in self._aux and t not in self._aux:
                        self._controlled_h(em, b + c, b + t)
                    else:
                        em.add(f"CX {b + c} {b + t}")
                continue
            self._local_to_global += self._emit_plain(em, inst)

    def _emit_plain(self, em: Emitter, inst) -> List[int]:
        """Emit ``inst`` unchanged on this block's qubits; return its records."""
        if inst.name == "TICK":
            return []
        args = inst.gate_args_copy()
        arg_text = "(" + ",".join(repr(a) for a in args) + ")" if args else ""
        line = f"{inst.name}{arg_text} " + " ".join(str(self._base + t.value) for t in inst.targets_copy())
        if inst.num_measurements:
            return em.measure(line, inst.num_measurements)
        em.add(line)
        return []

    def _records(self, targets) -> Set[int]:
        n = len(self._local_to_global)
        recs: Set[int] = set()
        for t in targets:
            recs ^= {self._local_to_global[n + t.value]}
        return recs

    def _controlled_h(self, em: Emitter, control: int, target: int) -> None:
        """Controlled-H = A CZ A^dagger with A = exp(-i pi/8 Y), so that
        A Z A^dagger = H. A is a T gate conjugated by SQRT_X."""
        em.add(f"SQRT_X {target}")
        self._t(em, target, dagger=True)
        em.add(f"SQRT_X_DAG {target}")
        em.add(f"CZ {control} {target}")
        em.add(f"SQRT_X {target}")
        self._t(em, target)
        em.add(f"SQRT_X_DAG {target}")
