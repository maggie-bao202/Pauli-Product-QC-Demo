import numpy as np
import stim

from ppqc import (circuit_distribution, cnot, compile_to_ppm, h, litinski_fig4, ppm_distribution,
                  s, t, t_layers)


def test_fig4_matches_the_paper():
    program = compile_to_ppm(litinski_fig4(), 4)
    # Litinski Fig. 4, bottom right. The rotations all commute, so compare as a set.
    assert {str(p) for p in program.rotations} == {"+Z___", "+_XY_", "-___Y", "-ZZZY"}
    assert [str(p) for p in program.measurements] == ["+YZZY", "+XX__", "-__Z_", "+X__X"]
    assert len(t_layers(program.rotations)) == 1


def test_fig4_compilation_preserves_the_distribution():
    circuit = litinski_fig4()
    program = compile_to_ppm(circuit, 4)
    assert np.allclose(circuit_distribution(circuit, 4), ppm_distribution(program))


def test_h_t_measure():
    # H then T then H: Z_{pi/8} becomes X_{pi/8}, and the two H cancel.
    n = 1
    circuit = h(0, n) + t(0, n) + h(0, n)
    program = compile_to_ppm(circuit, n)
    assert [str(p) for p in program.rotations] == ["+X"]
    assert [str(p) for p in program.measurements] == ["+Z"]


def test_random_clifford_t_circuits():
    rng = np.random.default_rng(0)
    n = 3
    for _ in range(20):
        circuit = []
        for _ in range(25):
            kind = rng.integers(4)
            a, b = rng.choice(n, size=2, replace=False)
            circuit += [h, s, t][kind](int(a), n) if kind < 3 else cnot(int(a), int(b), n)
        program = compile_to_ppm(circuit, n)
        assert np.allclose(circuit_distribution(circuit, n), ppm_distribution(program))


def test_t_layers_separates_anticommuting_rotations():
    layers = t_layers([stim.PauliString("X_"), stim.PauliString("Z_"), stim.PauliString("_Z")])
    assert [[str(p) for p in layer] for layer in layers] == [["+X_", "+_Z"], ["+Z_"]]
