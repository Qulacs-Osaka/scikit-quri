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
    return psi.conj().T @ Observable @ psi


# hamiltonian = np.kron(X, I) + np.kron(I, Y)
# print(hamiltonian)
# print()

# Gj = np.kron(Y, I)
# ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
# print(ans)
# print()

# Gj = np.kron(I, X)
# ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
# print(ans)

# hamiltonian = X
# Gj = Y
# ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
# print(ans)
# print()


# hamiltonian = Y
# Gj = X
# ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
# print(ans)
# print()

# psi = np.array([1, 0], dtype=complex)
# U = RZ(pi / 2) @ RY(pi / 2) @ RX(pi / 2)
# psi_final = U @ psi

# hamiltonian = X
# Oj = 0.5j * commutator(X, X)

# print(Oj)
# ans = psi_final.conj().T @ Oj @ psi_final
# print(ans)


hamiltonian = Z
U = RZ(pi / 2) @ RY(pi / 2) @ RX(pi / 2)
psi = np.array([1, 0], dtype=complex)
psi_final = U @ psi

Oj = 0.5j * commutator(X, Z)
ans = expectation_value(psi_final, Oj)
print(ans)

Oj = 0.5j * commutator(Y, Z)
ans = expectation_value(psi_final, Oj)
print(ans)

Oj = 0.5j * commutator(Z, Z)
ans = expectation_value(psi_final, Oj)
print(ans)
