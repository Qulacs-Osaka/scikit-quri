import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.inverse import inverse_circuit
from quri_parts.core.sampling import Sampler, ConcurrentSampler


class overlap_estimator:
    """quri-partsのoverlap_estimatorの代替Class."""

    def __init__(self, concurrent_sampler: ConcurrentSampler, n_shots: int = 1000):
        """
        Args:
            sampler (ConcurrentSampler): サンプリング関数
            n_shots (int): ショット数 defaults to 1000.

        """
        self.concurrent_sampler = concurrent_sampler
        self.n_shots = n_shots
        self.cache = {}

    def create_overlap_circuit(
        self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """ket_circuitとbra_circuitからoverlapを計算するための量子回路を作成する関数
        引数に対して非破壊的に動作
        Args:
            ket_circuit (QuantumCircuit): 量子回路(ket)
            bra_circuit (QuantumCircuit): 量子回路(bra)

        Returns:
            QuantumCircuit: overlapを計算するための量子回路

        """
        ket_circuit = ket_circuit.get_mutable_copy()
        bra_circuit = bra_circuit.get_mutable_copy()
        return ket_circuit + inverse_circuit(bra_circuit)

    # TODO list化に対応したい
    def estimate(self, ket_circuit: QuantumCircuit, bra_circuit: QuantumCircuit) -> float:
        """与えられた量子回路のi番目とj番目の内積の絶対値の二乗を計算
        引数に対して非破壊的に動作
        Args:
            ket_circuit (QuantumCircuit): 量子回路(ket)
            bra_circuit (QuantumCircuit): 量子回路(bra)

        Returns:
            float: |<φi|φj>|^2

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
        """
        circuitsの各量子回路の相関行列を計算する関数
        ket_circuitsとbra_circuitsのすべての組み合わせに対してestimateを行う。
        predictが正方行列ではないため、相関行列のflattenを返す
        Args:
            ket_circuits (list[QuantumCircuit]): 量子回路のリスト(ket)
            bra_circuits (list[QuantumCircuit]): 量子回路のリスト(bra)
        Returns:
            list[list[float]]: 内積の絶対値の二乗の行列
        """
        overlap_circuits = []
        n_ket = len(ket_circuits)
        n_bra = len(bra_circuits)
        for i in range(n_ket):
            for j in range(n_bra):
                # 一旦後先考えず全部Circuit作る
                overlap_circuits.append(
                    self.create_overlap_circuit(ket_circuits[i], bra_circuits[j])
                )
        sampling_counts = self.concurrent_sampler(
            [(circuit, self.n_shots) for circuit in overlap_circuits]
        )
        overlaps = np.array([count.get(0, 0) / self.n_shots for count in sampling_counts])
        return overlaps
