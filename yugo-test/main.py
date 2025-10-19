from enum import Enum

import modules.qiskit_module as qiskit_module
import modules.scikit_module as scikit_module
import numpy as np
from modules.model import GateInfo, GateMode, GateType, TestData

pi = np.pi

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
data = [
    TestData(
        n_qubits=3,
        gates=[
            GateInfo(GateType.H, 0),
            GateInfo(GateType.RX, 1, param=0.5),
            GateInfo(GateType.RX, 0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, 1, gate_mode=GateMode.INPUT),
            GateInfo(GateType.RZ, 2, gate_mode=GateMode.LEARNING),
        ],
        x=[0.123],
        theta=[0.456, 0.789],
        observable=[
            ("X0", 1.0),
            ("X1", 1.0),
            ("X2", 1.0),
        ],
        enabled=False,
    ),
    TestData(
        n_qubits=1,
        gates=[
            GateInfo(GateType.RX, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RX, t_bit=0, gate_mode=GateMode.LEARNING),
        ],
        x=[],
        theta=[pi / 4, pi / 4, pi / 2],
        observable=[
            ("Z0", 1.0),
        ],
        enabled=False,
    ),
    TestData(
        n_qubits=3,
        gates=[
            GateInfo(GateType.RX, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=1, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RZ, t_bit=1, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RX, t_bit=2, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RZ, t_bit=2, gate_mode=GateMode.LEARNING),
        ],
        x=[],
        theta=[pi / 4, pi / 4, pi / 4, pi / 4, pi / 4, pi / 4],
        observable=[
            ("Z0", 1.0),
            ("X1", 1.0),
            ("Y2", 1.0),
        ],
        enabled=False,
    ),
    TestData(
        n_qubits=3,
        gates=[
            GateInfo(GateType.RX, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RZ, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=1, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RX, t_bit=1, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RZ, t_bit=1, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RX, t_bit=2, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RY, t_bit=2, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RZ, t_bit=2, gate_mode=GateMode.LEARNING),
        ],
        x=[],
        theta=[pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3],
        observable=[
            ("Z0", 1.0),
            ("X1", 1.0),
            ("Y2", 1.0),
        ],
        enabled=False,
    ),
    TestData(
        n_qubits=3,
        gates=[
            GateInfo(GateType.RY, t_bit=0, gate_mode=GateMode.LEARNING),
            GateInfo(GateType.RX, t_bit=1, gate_mode=GateMode.LEARNING),
        ],
        x=[],
        theta=[
            0,
            0,
        ],
        observable=[
            ("X0", 1.0),
            ("Y1", 1.0),
            ("Z2", 1.0),
        ],
        enabled=False,
    ),
    TestData(
        n_qubits=2,
        gates=[
            GateInfo(GateType.H, t_bit=0),
            GateInfo(GateType.CX, t_bit=0, c_bit=1),
            GateInfo(GateType.RY, t_bit=1, gate_mode=GateMode.LEARNING),
        ],
        x=[],
        theta=[
            pi / 4,
        ],
        observable=[
            ("Z0", 1.0),
            ("Z1", 1.0),
        ],
        enabled=True,
    ),
]

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
for test in data:
    if not test.enabled:
        continue
    qiskit_module.execute(test)
    print()
    scikit_module.execute(test)
    print("//＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
