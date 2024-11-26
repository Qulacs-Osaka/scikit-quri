from typing import List
from quri_parts.core.state import GeneralCircuitQuantumState,QuantumState
import numpy as np
from quri_parts.qulacs.circuit import convert_circuit
from quri_parts.qulacs.overlap_estimator import (
    create_qulacs_vector_overlap_estimator,
    _create_qulacs_initial_state,
)
from qulacs.state import inner_product
from qulacs import QuantumState
from quri_parts.core.state import GeneralCircuitQuantumState

class overlap_estimator:
    """
    quri-partsのoverlap_estimatorの代替Class
    (n_data:500のとき,x60 faster)
    """

    def __init__(self, states: List[GeneralCircuitQuantumState]):
        """
        Args:
            states (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        self.states = states
        self.qula_states = np.full(len(states), fill_value=None, dtype=object)

    def _state_to_qula_state(self, state: GeneralCircuitQuantumState) -> QuantumState:
        """
        量子状態をqulacsのstateに変換
        Args:
            state (GeneralCircuitQuantumState): quri-partsの量子状態
        Returns:
            qulacs_state (QuantumState): qulacsの量子状態
        """
        circuit = convert_circuit(state.circuit)
        qulacs_state = _create_qulacs_initial_state(state)
        circuit.update_quantum_state(qulacs_state)
        return qulacs_state

    def add_state(self, states:List[GeneralCircuitQuantumState]):
        """
        量子状態を追加
        Args:
            states (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        self.states.extend(states)
        self.qula_states = np.append(self.qula_states, np.full(len(states), fill_value=None, dtype=object))

    def calc_all_qula_states(self):
        """
        cache用に予め全ての量子状態をqulacsのstateに変換
        """
        for i in range(len(self.states)):
            self.qula_states[i] = self._state_to_qula_state(self.states[i])

    def estimate(self, i: int, j: int):
        # ? これi,jじゃなくて数値でhash取った方が使いやすそう
        """
        与えられた量子状態のi番目とj番目の内積の絶対値の二乗を計算
        Args:
            i (int): 量子状態のindex(ket)
            j (int): 量子状態のindex(bra)
        Returns:
            float: |<φi|φj>|^2
        """
        ket = self.qula_states[i]
        # qulacsのstateを使いまわす
        if ket is None:
            ket = self._state_to_qula_state(self.states[i])
            self.qula_states[i] = ket
        bra = self.qula_states[j]
        if bra is None:
            bra = self._state_to_qula_state(self.states[j])
            self.qula_states[j] = bra
        overlap = inner_product(bra, ket)
        overlap_mag_sqrd = abs(overlap) ** 2
        return overlap_mag_sqrd