from typing import List
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.inverse import inverse_circuit
from quri_parts.backend import SamplingBackend
from quri_parts.core.sampling import create_sampler_from_sampling_backend


class overlap_estimator_real_device:
    """
    quri-partsのoverlap_estimatorの代替Class
    (n_data:500のとき,x60 faster)
    """

    def __init__(
        self, circuit: List[QuantumCircuit], sampler: SamplingBackend, n_shots: int = 1000
    ):
        """
        Args:
            states (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        # self.states = states
        self.circuits = circuit
        self.sampler = create_sampler_from_sampling_backend(sampler)
        self.n_shots = n_shots
        # self._generate_state()

    def add_data(self, circuits: List[QuantumCircuit]) -> None:
        """
        量子状態を追加
        Args:
            states (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        self.circuits.extend(circuits)

    def estimate(self, i: int, j: int) -> float:
        # ? これi,jじゃなくて数値でhash取った方が使いやすそう
        """
        与えられた量子状態のi番目とj番目の内積の絶対値の二乗を計算
        Args:
            i (int): 量子状態のindex(ket)
            j (int): 量子状態のindex(bra)
        Returns:
            float: |<φi|φj>|^2
        """
        ket_circuit = self.circuits[i].get_mutable_copy()
        bra_circuit = self.circuits[j].get_mutable_copy()
        circuit = ket_circuit
        # 逆順にcombine
        inv_circuit = inverse_circuit(bra_circuit)
        for inv_gate in inv_circuit.gates:
            circuit.add_gate(inv_gate)
        sampling_count = self.sampler(circuit, self.n_shots)
        count_zero = sampling_count.get(0)
        if count_zero is None:
            count_zero = 0
        # sampling_result = sampling_job.result()
        # count_zero = sampling_result.counts.get(0)
        p = count_zero / self.n_shots
        return p
