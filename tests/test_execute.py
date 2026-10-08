import clifft
import numpy as np
import pytest
import stim

from ppqc import (BareMagic, H6Magic, build_branch_circuit, circuit_distribution, compile_to_ppm,
                  litinski_fig4, ppm_distribution, run, total_variation)
from ppqc.compiler import PPMProgram

SOURCES = [BareMagic, H6Magic]


def _program(rotations, measurements):
    return PPMProgram(len(measurements), [stim.PauliString(p) for p in rotations],
                      [stim.PauliString(p) for p in measurements])


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("rotations", [["X"], ["-Y"], ["X", "Z", "X"], ["X", "-Z", "Y", "-X", "Z"]])
def test_one_qubit_programs(source, rotations):
    # Anticommuting rotations exercise both the Clifford and the Pauli corrections.
    program = _program(rotations, ["-X"] if len(rotations) == 5 else ["Z"])
    result = run(program, lambda: source(1), 40_000, threads=1)
    assert total_variation(result.distribution, ppm_distribution(program)) < 0.015


@pytest.mark.parametrize("source", SOURCES)
def test_fig4_noiseless(source):
    circuit = litinski_fig4()
    program = compile_to_ppm(circuit, 4)
    result = run(program, lambda: source(4), 60_000, threads=1)
    assert total_variation(result.distribution, circuit_distribution(circuit, 4)) < 0.02
    # Every branch is hit with probability 1/16 and nothing else is rejected.
    assert result.acceptance == pytest.approx(1.0, abs=0.02)


def test_h6_checks_are_deterministic_without_noise():
    # The real protocol (|H+> inputs, controlled-H check) passes every
    # LightStim detector and the post-check SE round with certainty.
    source = H6Magic(1)
    em_text, expected, _ = build_branch_circuit(_program(["X"], ["Z"]), (0,), source)
    num_h6 = len(source.post_selected)
    mask = [1] * num_h6 + [0] * (len(expected) - num_h6)
    compiled = clifft.compile(em_text, postselection_mask=mask, expected_detectors=expected)
    result = clifft.sample_survivors(compiled, 5_000, seed=0)
    assert result.passed_shots == result.total_shots


def test_h6_suppresses_t_gate_noise():
    circuit = litinski_fig4()
    program = compile_to_ppm(circuit, 4)
    ideal = circuit_distribution(circuit, 4)
    bare = run(program, lambda: BareMagic(4, p_t=0.05), 200_000, threads=1)
    h6 = run(program, lambda: H6Magic(4, p_t=0.05), 200_000, threads=1)
    assert total_variation(h6.distribution, ideal) < 0.5 * total_variation(bare.distribution, ideal)
    assert h6.acceptance < 1


def test_error_orders_on_a_deterministic_program():
    # Four pi/8 rotations about X make a Pauli X, so the ideal outcome is always 1
    # and the error rate has no sampling floor. Bare states give a first-order
    # error; H6 gives second order, but only with the post-check SE round.
    program = _program(["X"] * 4, ["Z"])

    def error(make_source):
        return run(program, make_source, 400_000, threads=1).distribution[0]

    assert error(lambda: H6Magic(1)) == 0
    low, high = 0.02, 0.04
    order = {
        "bare": np.log2(error(lambda: BareMagic(1, p_t=high)) / error(lambda: BareMagic(1, p_t=low))),
        "h6": np.log2(error(lambda: H6Magic(1, p_t=high)) / error(lambda: H6Magic(1, p_t=low))),
        "h6 without post-check": np.log2(
            error(lambda: H6Magic(1, p_t=high, post_check_rounds=0))
            / error(lambda: H6Magic(1, p_t=low, post_check_rounds=0))),
    }
    assert order["bare"] == pytest.approx(1, abs=0.2)
    assert order["h6"] == pytest.approx(2, abs=0.4)
    assert order["h6 without post-check"] == pytest.approx(1, abs=0.2)
