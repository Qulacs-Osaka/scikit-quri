import numpy as np
from manual_calc import RX, RY, RZ, I, X, Y, Z, commutator, dagger, expectation_value, kron_n_gate

pi = np.pi


def compute_gradients(params, gates, generators, hamiltonian, init_state, n_qubits):
    ans = []
    len_params = len(params)
    for i in range(len_params):
        psi_copy = init_state.copy()
        # generator_applied = []
        # 順方向で作用
        for j in range(len_params):
            psi_copy = gates[j] @ psi_copy
        # 逆方向で作用
        for j in range(len_params - 1, i, -1):
            psi_copy = dagger(gates[j]) @ psi_copy
            

        # 勾配計算
        gen_gate = generators[i]
        if gen_gate[0] == "X":
            gen = kron_n_gate(X / 2, gen_gate[1], n_qubits)
        elif gen_gate[0] == "Y":
            gen = kron_n_gate(Y / 2, gen_gate[1], n_qubits)
        elif gen_gate[0] == "Z":
            gen = kron_n_gate(Z / 2, gen_gate[1], n_qubits)
        else:
            raise ValueError("Unknown generator")

        comm = commutator(gen, hamiltonian)
        grad_i = 1j * expectation_value(psi_copy, comm)
        ans.append(grad_i.real)
    return ans


test_cases = [
    (
        [pi / 4, pi / 4, pi / 2],
        [RY(pi / 4), RZ(pi / 4), RY(pi / 2)],
        [("Y", 0), ("Z", 0), ("Y", 0)],
        X,
        [-0.7071, 0.0, -0.5],
        1,
    ),
    (
        [pi / 4],
        [RX(pi / 4)],
        [("X", 0)],
        Z,
        [-0.7071],
        1,
    ),
    (
        [pi / 3, pi / 6],
        [RX(pi / 3), RZ(pi / 6)],
        [("X", 0), ("Z", 0)],
        Y,
        [-0.433, 0.433],
        1,
    ),
    (
        [pi / 4, pi / 2, pi / 4, pi / 2],
        [RY(pi / 4, 0, 2), RZ(pi / 2, 0, 2), RZ(pi / 4, 1, 2), RY(pi / 2, 1, 2)],
        [("Y", 0), ("Z", 0), ("Y", 1), ("Z", 1)],
        np.kron(X, I) + np.kron(I, X),
        [0.0, -0.7071, 0, 0.0],
        2,
    ),
]

for case in test_cases:
    params, gates, generators, hamiltonian, expected, n_qubits = case
    init_state = np.zeros(2**n_qubits, dtype=complex)
    init_state[0] = 1.0
    grad = compute_gradients(
        params, gates, generators, hamiltonian, np.array(init_state, dtype=complex), n_qubits
    )
    print(np.round(grad, 4), expected)
