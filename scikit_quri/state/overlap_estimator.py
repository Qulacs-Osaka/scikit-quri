import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.inverse import inverse_circuit
from quri_parts.core.sampling import ConcurrentSampler


class overlap_estimator:
    """Alternative implementation of quri-parts' overlap estimator."""

    def __init__(self, concurrent_sampler: ConcurrentSampler, n_shots: int = 1000):
        """
        Args:
            concurrent_sampler: Concurrent sampler function.
            n_shots: Number of shots per circuit execution. Defaults to 1000.

        """
        self.concurrent_sampler = concurrent_sampler
        self.n_shots = n_shots
        self.cache = {}

    def create_overlap_circuit(
        self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """Create a circuit to compute the overlap between two quantum states.
        Operates non-destructively on the input circuits.

        Args:
            ket_circuit: Quantum circuit representing the ket state.
            bra_circuit: Quantum circuit representing the bra state.

        Returns:
            A quantum circuit of the form U_ket U_bra†, whose |0⟩ measurement
            probability approximates |⟨ψ_ket|ψ_bra⟩|².

        """
        ket_circuit = ket_circuit.get_mutable_copy()
        bra_circuit = bra_circuit.get_mutable_copy()
        return ket_circuit + inverse_circuit(bra_circuit)

    def estimate(self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit) -> float:
        """Estimate the squared overlap |⟨ψ_ket|ψ_bra⟩|² between two quantum states.
        Operates non-destructively on the input circuits.

        Args:
            ket_circuit: Quantum circuit representing the ket state.
            bra_circuit: Quantum circuit representing the bra state.

        Returns:
            Estimated value of |⟨ψ_ket|ψ_bra⟩|².

        """
        circuit = self.create_overlap_circuit(ket_circuit, bra_circuit)
        sampling_count = list(self.concurrent_sampler([(circuit, self.n_shots)]))
        count_zero = sampling_count[0].get(0)
        if not count_zero:
            count_zero = 0
        p = count_zero / self.n_shots
        return p

    def estimate_concurrent(
        self, ket_circuits: list[QuantumCircuit], bra_circuits: list[QuantumCircuit]
    ) -> NDArray[np.float64]:
        """Estimate |⟨ψ_i|ψ_j⟩|² for all combinations of ket and bra circuits.

        Args:
            ket_circuits: List of quantum circuits representing ket states.
            bra_circuits: List of quantum circuits representing bra states.

        Returns:
            Flat array of shape (n_ket * n_bra,) containing the squared overlaps
            for all (ket, bra) pairs in row-major order.
        """
        overlap_circuits = []
        n_ket = len(ket_circuits)
        n_bra = len(bra_circuits)
        for i in range(n_ket):
            for j in range(n_bra):
                overlap_circuits.append(
                    self.create_overlap_circuit(ket_circuits[i], bra_circuits[j])
                )
        sampling_counts = self.concurrent_sampler(
            [(circuit, self.n_shots) for circuit in overlap_circuits]
        )
        overlaps = np.array([count.get(0, 0) / self.n_shots for count in sampling_counts])
        return overlaps
