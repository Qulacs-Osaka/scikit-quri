from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.inverse import inverse_circuit
from quri_parts.core.sampling import Sampler


class overlap_estimator:
    """quri-partsのoverlap_estimatorの代替Class."""

    def __init__(self, sampler: Sampler, n_shots: int = 1000):
        """
        Args:
            sampler (Sampler): サンプリング関数
            n_shots (int): ショット数 defaults to 1000.

        """
        self.sampler = sampler
        self.n_shots = n_shots
        self.cache = {}

    def estimate(self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit) -> float:
        """与えられた量子回路のi番目とj番目の内積の絶対値の二乗を計算
        引数に対して非破壊的に動作
        Args:
            ket_circuit (QuantumCircuit): 量子回路(ket)
            bra_circuit (QuantumCircuit): 量子回路(bra)

        Returns:
            float: |<φi|φj>|^2

        """
        ket_circuit = ket_circuit.get_mutable_copy()
        bra_circuit = bra_circuit.get_mutable_copy()
        circuit = ket_circuit
        # 逆順にcombine
        inv_circuit = inverse_circuit(bra_circuit)
        circuit += inv_circuit
        sampling_count = self.sampler(circuit, self.n_shots)
        count_zero = sampling_count.get(0)
        if not count_zero:
            count_zero = 0
        p = count_zero / self.n_shots
        return p
