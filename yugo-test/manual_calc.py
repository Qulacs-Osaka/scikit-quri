import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)
pi = np.pi


def RX(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X


def RY(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Y


def RZ(theta):
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Z


def commutator(A, B):
    return A @ B - B @ A


def expectation_value(psi, Observable):
    return dagger(psi) @ Observable @ psi


def dagger(U):
    return U.conj().T


# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
test_cases = [
    (
        [pi / 4, pi / 4, pi / 2],
        [RY(pi / 4), RZ(pi / 4), RY(pi / 2)],
        ["Y", "Z", "Y"],
        X,
        [-0.7071, 0.0, -0.5],
    ),
    (
        [pi / 4],
        [RX(pi / 4)],
        ["X"],
        Z,
        [-0.7071],
    ),
    (
        [pi / 3, pi / 6],
        [RX(pi / 3), RZ(pi / 6)],
        ["X", "Z"],
        Y,
        [-0.433, 0.433],
    ),
]


def compute_gradients(params, gates, generators, hamiltonian, init_state):
    ans = []
    len_params = len(params)
    for i in range(len_params):
        psi_copy = init_state.copy()
        for j in range(i + 1):
            psi_copy = gates[j] @ psi_copy

        H_copy = hamiltonian.copy()
        for j in range(len_params - 1, i, -1):
            H_copy = dagger(gates[j]) @ H_copy @ gates[j]

        if generators[i] == "X":
            Obs = 0.5j * commutator(X, H_copy)
        elif generators[i] == "Y":
            Obs = 0.5j * commutator(Y, H_copy)
        elif generators[i] == "Z":
            Obs = 0.5j * commutator(Z, H_copy)
        else:
            raise ValueError("Unknown generator")

        ans.append(expectation_value(psi_copy, Obs).real)
    return np.round(ans, 4)


for case in test_cases:
    params, gates, generators, hamiltonian, expected = case
    grad = compute_gradients(
        params, gates, generators, hamiltonian, np.array([1, 0], dtype=complex)
    )
    print(np.round(grad, 4), expected)
