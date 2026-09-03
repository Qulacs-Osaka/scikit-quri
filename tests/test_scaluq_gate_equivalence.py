"""Every gate the scaluq adapter can be handed must agree with qulacs.

`quri_parts_scaluq/` is vendored because no `quri-parts-scaluq` package exists, so its
correctness is this repository's responsibility. The only coverage it had was one
end-to-end iris comparison, which cannot say which gate is wrong when it fails.
"""

import numpy as np
import pytest
from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

from scikit_quri.backend import ScaluqEstimator

QULACS = create_qulacs_vector_concurrent_estimator()
OPERATOR = Operator(
    {
        pauli_label("Z 0"): 1.0,
        pauli_label("X 0"): 0.5,
        pauli_label("Y 1"): 0.3,
        pauli_label("Z 1"): 0.7,
    }
)
THETA = 0.6
UNITARY = [
    [complex(np.cos(THETA)), complex(-np.sin(THETA))],
    [complex(np.sin(THETA)), complex(np.cos(THETA))],
]

GATES = {
    "X": lambda c: c.add_X_gate(0),
    "Y": lambda c: c.add_Y_gate(0),
    "Z": lambda c: c.add_Z_gate(0),
    "H": lambda c: c.add_H_gate(0),
    "S": lambda c: c.add_S_gate(0),
    "Sdag": lambda c: c.add_Sdag_gate(0),
    "T": lambda c: c.add_T_gate(0),
    "Tdag": lambda c: c.add_Tdag_gate(0),
    "SqrtX": lambda c: c.add_SqrtX_gate(0),
    "SqrtXdag": lambda c: c.add_SqrtXdag_gate(0),
    "SqrtY": lambda c: c.add_SqrtY_gate(0),
    "SqrtYdag": lambda c: c.add_SqrtYdag_gate(0),
    "RX": lambda c: c.add_RX_gate(0, THETA),
    "RY": lambda c: c.add_RY_gate(0, THETA),
    "RZ": lambda c: c.add_RZ_gate(0, THETA),
    "U1": lambda c: c.add_U1_gate(0, THETA),
    "U2": lambda c: c.add_U2_gate(0, THETA, THETA / 2),
    "U3": lambda c: c.add_U3_gate(0, THETA, THETA / 2, THETA / 3),
    "CNOT": lambda c: c.add_CNOT_gate(0, 1),
    "CZ": lambda c: c.add_CZ_gate(0, 1),
    "SWAP": lambda c: c.add_SWAP_gate(0, 1),
    "TOFFOLI": lambda c: c.add_TOFFOLI_gate(0, 1, 2),
    "Pauli": lambda c: c.add_Pauli_gate([0, 1], [1, 3]),
    "PauliRotation": lambda c: c.add_PauliRotation_gate([0, 1], [1, 3], THETA),
    "UnitaryMatrix": lambda c: c.add_gate(gates.UnitaryMatrix([0], UNITARY)),
    "Identity": lambda c: c.add_Identity_gate(0),
}


@pytest.mark.parametrize("name", sorted(GATES), ids=sorted(GATES))
def test_gate_matches_qulacs(name):
    """Compare from a non-trivial input state: |0> hides phase and rotation errors."""
    circuit = QuantumCircuit(3)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    GATES[name](circuit)

    state = GeneralCircuitQuantumState(3, circuit)
    expected = list(QULACS([OPERATOR], [state]))[0].value.real
    got = list(ScaluqEstimator().estimate([OPERATOR], [state]))[0].value.real
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", ["MCX", "MCRZ", "MCH"])
def test_gates_without_a_conversion_raise(name):
    """A gate quri-parts knows but the adapter does not must fail loudly.

    These fell through to `assert False, "Unreachable"`, which carries no message and
    is removed entirely under `python -O` — the conversion would then return None and
    the circuit would be silently wrong.
    """
    from quri_parts_scaluq.circuit import convert_gate

    template = QuantumCircuit(2)
    template.add_CNOT_gate(0, 1)
    reference = template.gates[0]
    unhandled = type(reference)(
        name=name,
        target_indices=reference.target_indices,
        control_indices=reference.control_indices,
    )
    with pytest.raises(ValueError, match="Unsupported gate"):
        convert_gate(unhandled)


def test_no_bare_asserts_remain_in_the_converter():
    """`assert False` disappears under python -O, turning a hard failure into a wrong
    result. The converter must raise instead."""
    import inspect

    import quri_parts_scaluq.circuit as adapter

    assert "assert False" not in inspect.getsource(adapter)
