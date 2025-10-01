import numpy as np

# 基本パウリ行列
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
pi = np.pi


# dagger
def dagger(U):
    return U.conj().T


# n量子ビット系で、target番目にgateを作用させる
def kron_n_gate(gate, target, n_qubits):
    ops = [I] * n_qubits
    ops[target] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# 単一量子ビット回転ゲートを多量子ビット拡張
def RX(theta, target=0, n_qubits=1):
    g = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X
    return kron_n_gate(g, target, n_qubits)


def RY(theta, target=0, n_qubits=1):
    g = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Y
    return kron_n_gate(g, target, n_qubits)


def RZ(theta, target=0, n_qubits=1):
    g = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Z
    return kron_n_gate(g, target, n_qubits)


# 交換子
def commutator(A, B):
    return A @ B - B @ A


# 期待値
def expectation_value(psi, Observable):
    return np.vdot(psi, Observable @ psi)


# 勾配計算
def compute_gradients(params, gates, generators, hamiltonian, init_state, n_qubits):
    ans = []
    len_params = len(params)
    for i in range(len_params):
        # 順方向で作用
        psi_copy = init_state.copy()
        for j in range(i + 1):
            psi_copy = gates[j] @ psi_copy

        # 逆方向でハミルトニアンを変換
        H_copy = hamiltonian.copy()
        for j in range(len_params - 1, i, -1):
            H_copy = dagger(gates[j]) @ H_copy @ gates[j]

        gen, target = generators[i]

        if gen == "X":
            Obs = 0.5j * commutator(kron_n_gate(X, target, n_qubits), H_copy)
        elif gen == "Y":
            Obs = 0.5j * commutator(kron_n_gate(Y, target, n_qubits), H_copy)
        elif gen == "Z":
            Obs = 0.5j * commutator(kron_n_gate(Z, target, n_qubits), H_copy)
        else:
            raise ValueError("Unknown generator")

        ans.append(expectation_value(psi_copy, Obs).real)
    return np.round(ans, 4)


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
