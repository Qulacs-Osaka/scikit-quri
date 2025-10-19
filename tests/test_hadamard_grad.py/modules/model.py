from enum import Enum


class GateType(Enum):
    X = "x"
    Y = "y"
    Z = "z"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    H = "h"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    SWAP = "swap"


class GateMode(Enum):
    LEARNING = "learning"
    INPUT = "input"


class GateInfo:
    def __init__(
        self,
        gate_type: GateType,
        t_bit: int,
        gate_mode: GateMode | None = None,
        c_bit: int = -1,
        param: float = 0.0,
    ):
        self.gate_type = gate_type
        self.gate_mode = gate_mode
        self.t_bit = t_bit
        self.c_bit = c_bit
        self.param = param


class TestData:
    def __init__(
        self,
        n_qubits: int,
        gates: list[GateInfo],
        x: list[float],
        theta: list[float],
        observable: list[tuple[str, float]],
        enabled: bool = True,
    ):
        self.gates = gates
        self.n_qubits = n_qubits
        self.observable = observable
        self.x = x
        self.theta = theta
        self.enabled = enabled

        input_index = 0
        lp_index = 0
        for gate in gates:
            if gate.gate_mode == None:
                continue
            elif gate.gate_mode == GateMode.INPUT:
                gate.param = x[input_index]
                input_index += 1
            elif gate.gate_mode == GateMode.LEARNING:
                gate.param = theta[lp_index]
                lp_index += 1
