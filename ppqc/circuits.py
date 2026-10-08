"""Example logical circuits."""
from typing import List

from .compiler import Rotation, controlled, rot


def litinski_fig4() -> List[Rotation]:
    """The 4-qubit example circuit of Litinski, "A Game of Surface Codes"
    (arXiv:1808.02892), Fig. 4, top left. Every qubit starts in |0> and is
    measured in Z at the end. Qubits q1..q4 of the figure are 0..3 here."""
    n = 4
    c: List[Rotation] = []
    # Column 1
    c.append(rot("Z0", "1/8", n))
    c += controlled("X1", "Z2", n)
    c.append(rot("X3", "-1/4", n))
    # Column 2
    c += controlled("X0", "Z1", n)
    c.append(rot("X2", "1/4", n))
    c.append(rot("Z3", "1/8", n))
    # Column 3
    c += controlled("X0", "Z3", n)
    # Column 4
    c.append(rot("Z0", "1/8", n))
    c.append(rot("Z1", "1/4", n))
    c.append(rot("Z2", "1/8", n))
    c.append(rot("Z3", "1/4", n))
    # Column 5
    c.append(rot("X0", "-1/4", n))
    c.append(rot("X1", "1/4", n))
    c.append(rot("X2", "1/4", n))
    c.append(rot("X3", "1/4", n))
    return c
