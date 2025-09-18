import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)
pi = np.pi

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

hamiltonian = X
Gj = Y
ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
print(ans)
print()


hamiltonian = Y
Gj = X
ans = 0.5j * (Gj @ hamiltonian - hamiltonian @ Gj)
print(ans)
print()
