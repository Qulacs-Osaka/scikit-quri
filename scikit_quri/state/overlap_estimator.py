from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.inverse import inverse_circuit
from quri_parts.core.sampling import Sampler


class overlap_estimator:
    """quri-partsのoverlap_estimatorの代替Class
    (n_data:500のとき,x60 faster)
    """

    def __init__(self, sampler: Sampler, n_shots: int = 1000):
        """Args:
        circuits (List[QuantumCircuit]): 量子回路のリスト

        """
        self.sampler = sampler
        self.n_shots = n_shots
        self.cache = {}

    def estimate(self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit) -> float:
        """与えられた量子回路のi番目とj番目の内積の絶対値の二乗を計算
        引数に対して非破壊的に動作
        Args:
            i (int): 量子状態のindex(ket)
            j (int): 量子状態のindex(bra)

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
        if count_zero is None:
            count_zero = 0
        p = count_zero / self.n_shots
        return p
