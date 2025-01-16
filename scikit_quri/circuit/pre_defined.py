import numpy as np
from numpy.random import default_rng, Generator
from functools import reduce
from typing import Optional
from .circuit import LearningCircuit
from numpy.typing import NDArray
from quri_parts.circuit import QuantumGate,CZ, CNOT


def create_qcl_ansatz(
    n_qubit: int, c_depth: int, time_step: float = 0.5, seed: Optional[int] = 0
) -> LearningCircuit:
    """Create a circuit used in this page: https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html
    Args:
        n_qubit: number of qubits
        c_depth: circuit depth
        time_step: the evolution time used for the hamiltonian dynamics
        seed: seed for random numbers. used for determining the interaction strength of the hamiltonian simulation
    Examples:
        >>> n_qubit = 4
        >>> circuit = create_qcl_ansatz(n_qubit, 3, 0.5)
        >>> qnn = QNNRegressor(circuit)
        >>> qnn.fit(x_train, y_train)
    """

    def preprocess_x(x: NDArray[np.float64], index: int) -> float:
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        # Capture copy of i by `i=i`.
        # Without this, i in lambda is a reference to the i, so the lambda always
        # recognize i as n_qubit - 1.
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    time_evol_gate = _create_time_evol_gate(n_qubit, time_step)
    for _ in range(c_depth):
        circuit.add_gate(time_evol_gate)
        for i in range(n_qubit):
            circuit.add_parametric_RX_gate(i)
            circuit.add_parametric_RZ_gate(i)
            circuit.add_parametric_RX_gate(i)
    return circuit


def _create_time_evol_gate(
    n_qubit, time_step=0.77, rng: Generator = None, seed: Optional[int] = 0
) -> QuantumGate:
    """create a hamiltonian dynamics with transverse field ising model with random interaction and random magnetic field
    Args:
        n_qubit: number of qubits
        time_step: evolution time
        rng: random number generator
        seed: seed for random number
    Return:
        qulacs' gate object
    """
    if rng is None:
        rng = default_rng(seed)

    ham = _make_hamiltonian(n_qubit, rng)
    # Create time evolution operator by diagonalization.
    # H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(
        np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
    )  # e^-iHT

    # Convert to a qulacs gate
    time_evol_gate = QuantumGate(
        name="UnitaryMatrix",
        target_indices=[i for i in range(n_qubit)],
        unitary_matrix=time_evol_op,
    )

    return time_evol_gate


def _make_hamiltonian(n_qubit, rng: Generator = None, seed: Optional[int] = 0):
    if rng is None:
        rng = default_rng(seed)
    X_mat = np.array([[0, 1], [1, 0]])
    Z_mat = np.array([[1, 0], [0, -1]])
    ham = np.zeros((2**n_qubit, 2**n_qubit), dtype=complex)
    for i in range(n_qubit):
        Jx = rng.uniform(-1.0, 1.0)
        ham += Jx * _make_fullgate([[i, X_mat]], n_qubit)
        for j in range(i + 1, n_qubit):
            J_ij = rng.uniform(-1.0, 1.0)
            ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)
    return ham


def _make_fullgate(list_SiteAndOperator, n_qubit):
    """
    Receive `list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...]` and
    insert identity to qubits which is not present in the list to create (2**n_qubit, 2**n_qubit) matrix
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    """
    I_mat = np.eye(2, dtype=complex)
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []
    cnt = 0
    for i in range(n_qubit):
        if i in list_Site:
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:
            list_SingleGates.append(I_mat)
    return reduce(np.kron, list_SingleGates)


def preprocess_x(x: np.ndarray, i: int) -> float:
    xa = x[i % len(x)]
    clamped = min(1, max(-1, xa))
    return clamped


def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> LearningCircuit:
    circuit = LearningCircuit(n_qubit)
    rng = default_rng(seed)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    zyu = list(range(n_qubit))

    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            circuit.circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i])
            circuit.add_parametric_RY_gate(zyu[i])
            circuit.circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RY_gate(zyu[i])
            circuit.add_parametric_RX_gate(zyu[i])
    return circuit


def create_ibm_embedding_circuit(n_qubit: int) -> LearningCircuit:
    """create circuit proposed in https://arxiv.org/abs/1802.06002.
    Args:
        n_qubits: number of qubits
    """

    def preprocess_x(x: NDArray[np.float64], index: int) -> float:
        xa: float = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_H_gate(i)
    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j, lambda x, i=i: (np.pi - preprocess_x(x, i) * (np.pi - preprocess_x(x, j)))
        )
        circuit.add_CNOT_gate(i, j)
    for i in range(n_qubit):
        circuit.add_H_gate(i)
    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j, lambda x, i=i: (np.pi - preprocess_x(x, i) * (np.pi - preprocess_x(x, j)))
        )
        circuit.add_CNOT_gate(i, j)
    return circuit

def create_dqn_cl(n_qubit: int, c_depth: int, s_qubit: int) -> LearningCircuit:
    # from https://arxiv.org/abs/2112.15002
    def preprocess_x(x: np.ndarray, i: int) -> float:
        xa = x[i % len(x)]
        clamped = min(1, max(-1, xa))
        return clamped
    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x,i=i: preprocess_x(x,i))
        circuit.add_parametric_RY_gate(i)
    
    for _ in range(c_depth):
        for i in range(n_qubit):
            # 元の論文ではすべての組に対して張っていたっぽいが、それはゲート数が多すぎるだろ
            circuit.add_gate(CZ(i,(i+1)%n_qubit))
        for i in range(s_qubit, n_qubit - 1):
            circuit.add_gate(CNOT(i,(i+1)%n_qubit))
        circuit.add_gate(CNOT(n_qubit-1,s_qubit))
        for i in range(n_qubit):
            circuit.add_parametric_RX_gate(i)
            circuit.add_parametric_RY_gate(i)
            circuit.add_parametric_RX_gate(i)
    
    return circuit

def create_dqn_cl_no_cz(n_qubit: int, c_depth: int) -> LearningCircuit:
    # from https://arxiv.org/abs/2112.15002
    def preprocess_x(x: np.ndarray, i: int) -> float:
        xa = x[i % len(x)]
        clamped = min(1, max(-1, xa))
        return clamped
    circuit = LearningCircuit(n_qubit)

    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x,i=i: preprocess_x(x,i))
        circuit.add_parametric_RY_gate(i)
    
    for _ in range(c_depth):
        for i in range(n_qubit):
            circuit.add_gate(CNOT(i, (i+1)%n_qubit))
            circuit.add_parametric_RX_gate(i)
            circuit.add_parametric_RY_gate(i)
            circuit.add_parametric_RX_gate(i)
        circuit.add_gate(CNOT(n_qubit-1,0))
    
    return circuit
