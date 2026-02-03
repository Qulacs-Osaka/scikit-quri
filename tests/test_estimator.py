from scikit_quri.backend import SimEstimator
from quri_parts.core.operator import pauli_label
from quri_parts.core import QuantumCircuit, quantum_state
from quri_parts.core.state import GeneralCircuitQuantumState


def create_simple_circuit() -> GeneralCircuitQuantumState:
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_CNOT_gate(0, 1)
    state = quantum_state(n_qubits=2, circuit=circuit)
    return state


def test_sim_estimator() -> None:
    """Test for SimEstimator."""
    circuit = create_simple_circuit()
    estimator = SimEstimator()
    estimate = estimator.estimate([pauli_label("Z0 Z1")], [circuit])
    assert estimate is not None


def test_oqtopus_estimator() -> None:
    """Test for OqtopusEstimator."""
    from scikit_quri.backend import OqtopusEstimator

    circuit = create_simple_circuit()
    estimator = OqtopusEstimator("qulacs")
    estimate = estimator.estimate([pauli_label("Z0 Z1")], [circuit])
    assert estimate is not None
