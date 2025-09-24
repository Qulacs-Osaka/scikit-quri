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


# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

hamiltonian = X
psi = np.array([1, 0], dtype=complex)

psi1 = RX(pi / 2) @ psi
Oj = (
    dagger(RZ(pi / 2))
    @ dagger(RY(pi / 2))
    @ (1j * commutator(0.5 * X, hamiltonian))
    @ RY(pi / 2)
    @ RZ(pi / 2)
)
ans = expectation_value(psi1, Oj)
print(f"{dagger(psi1)} \n{Oj} \n{psi1} \n={ans}")
print()

psi2 = RY(pi / 2) @ RX(pi / 2) @ psi
Oj = dagger(RZ(pi / 2)) @ (1j * commutator(0.5 * Y, hamiltonian)) @ RZ(pi / 2)
ans = expectation_value(psi2, Oj)
print(f"{dagger(psi2)} \n{Oj} \n{psi2} \n={ans}")
print()

psi3 = RZ(pi / 2) @ RY(pi / 2) @ RX(pi / 2) @ psi
Oj = 1j * commutator(0.5 * Z, hamiltonian)
ans = expectation_value(psi3, Oj)
print(f"{dagger(psi3)} \n{Oj} \n{psi3} \n={ans}")

# Oj = 0.5j * commutator(Y, Z)
# ans = expectation_value(psi_final, Oj)
# print(ans)

# Oj = 0.5j * commutator(Z, Z)
# ans = expectation_value(psi_final, Oj)
# print(ans)


def calc_grad(psi, hamiltonian, axis, params):
    match axis:
        case "X":
            U = RX(params[0])
            Gj = X
        case "Y":
            U = RY(params[0])
            Gj = Y
        case "Z":
            U = RZ(params[0])
            Gj = Z
        case _:
            U = I
            Gj = I
    psi_final = U @ psi
    Oj = 0.5j * commutator(Gj, hamiltonian)
    ans = expectation_value(psi_final, Oj)
    print(ans)


psi = np.array([1, 0], dtype=complex)

# calc_grad(psi, X, "X", [pi / 4])
# calc_grad(psi, X, "Y", [pi / 4])
# calc_grad(psi, X, "Z", [pi / 4])
# calc_grad(psi, Y, "X", [pi / 4])
# calc_grad(psi, Y, "Y", [pi / 4])
# calc_grad(psi, Y, "Z", [pi / 4])
# calc_grad(psi, Z, "X", [pi / 4])
# calc_grad(psi, Z, "Y", [pi / 4])
# calc_grad(psi, Z, "Z", [pi / 4])
