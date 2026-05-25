from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.circuit import (
    Parameter,
    ParametricQuantumGate,
    QuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
)
from quri_parts.core.estimator import ConcurrentQuantumEstimator
from quri_parts.core.operator import Operator, commutator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.circuit import convert_parametric_circuit
from quri_parts.rust.circuit.circuit_parametric import ImmutableBoundParametricQuantumCircuit

from .parameters import InputFunc, InputFuncWithParam, ParameterRegistry


class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""

    X = auto()
    Y = auto()
    Z = auto()


@dataclass
class LearningCircuit:
    """Parametric quantum circuit for quantum machine learning.

    Manages three types of circuit parameters:

    - **Fixed gates**: Non-parametric gates (X, Y, Z, H, CNOT, etc.).
    - **Input gates** (``add_input_R*``): Rotation angle is computed from input data ``x``
      via a user-supplied function at inference time.
    - **Learnable gates** (``add_parametric_R*``): Rotation angle is a trainable parameter
      updated during optimization. Multiple gates can share a single parameter via
      ``share_with`` / ``share_with_coef``.
    - **Parametric-input gates** (``add_parametric_input_R*``): Angle depends on both a
      learnable parameter and input data (e.g. ``f(theta, x)``).

    Args:
        n_qubits: Number of qubits in the circuit.

    Example:
        >>> circuit = LearningCircuit(n_qubits=4)
        >>> circuit.add_input_RY_gate(0, lambda x: np.arcsin(x[0]))
        >>> param_id = circuit.add_parametric_RX_gate(1)
        >>> bound = circuit.bind_input_and_parameters(x, theta)
    """

    n_qubits: int
    circuit: UnboundParametricQuantumCircuit = field(init=False)
    _registry: ParameterRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.circuit = UnboundParametricQuantumCircuit(self.n_qubits)
        self._registry = ParameterRegistry(lambda: self.circuit.parameter_count)

    def _new_parameter_position(self) -> int:
        return self.circuit.parameter_count

    # --- Fixed gates --------------------------------------------------------

    def add_gate(self, gate: QuantumGate) -> None:
        """Add arbitrary gate."""
        self.circuit.add_gate(gate)

    def add_X_gate(self, index: int) -> None:
        self.circuit.add_X_gate(index)

    def add_Y_gate(self, index: int) -> None:
        self.circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int) -> None:
        self.circuit.add_Z_gate(index)

    def add_RX_gate(self, index: int, angle: float) -> None:
        self._add_R_gate_inner(index, angle, _Axis.X)

    def add_RY_gate(self, index: int, angle: float) -> None:
        self._add_R_gate_inner(index, angle, _Axis.Y)

    def add_RZ_gate(self, index: int, angle: float) -> None:
        self._add_R_gate_inner(index, angle, _Axis.Z)

    def add_CNOT_gate(self, control_index: int, target_index: int) -> None:
        self.circuit.add_CNOT_gate(control_index, target_index)

    def add_H_gate(self, index: int) -> None:
        self.circuit.add_H_gate(index)

    # --- Input gates (angle = f(x)) ----------------------------------------

    def add_input_RX_gate(self, qubit: int, input_function: InputFunc) -> None:
        """Add an RX gate whose angle is determined by input data at inference time."""
        self._add_input_R_gate_inner(qubit, _Axis.X, input_function)

    def add_input_RY_gate(self, qubit: int, input_function: InputFunc) -> None:
        """Add an RY gate whose angle is determined by input data at inference time."""
        self._add_input_R_gate_inner(qubit, _Axis.Y, input_function)

    def add_input_RZ_gate(self, qubit: int, input_function: InputFunc) -> None:
        """Add an RZ gate whose angle is determined by input data at inference time."""
        self._add_input_R_gate_inner(qubit, _Axis.Z, input_function)

    # --- Learnable gates ----------------------------------------------------

    def add_parametric_RX_gate(
        self,
        qubit: int,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """Add a trainable RX gate and return its parameter ID.

        Args:
            qubit: Index of the target qubit.
            share_with: If given, share the learnable parameter with the gate whose
                ``parameter_id`` equals ``share_with``.
            share_with_coef: Coefficient applied to the shared parameter value
                (angle = shared_value * coef). Used only when ``share_with`` is set.

        Returns:
            parameter_id: Index of the learnable parameter assigned to this gate.
        """
        return self._add_parametric_R_gate_inner(qubit, _Axis.X, share_with, share_with_coef)

    def add_parametric_RY_gate(
        self,
        qubit: int,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """Add a trainable RY gate and return its parameter ID."""
        return self._add_parametric_R_gate_inner(qubit, _Axis.Y, share_with, share_with_coef)

    def add_parametric_RZ_gate(
        self,
        qubit: int,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """Add a trainable RZ gate and return its parameter ID."""
        return self._add_parametric_R_gate_inner(qubit, _Axis.Z, share_with, share_with_coef)

    def add_parametric_multi_Pauli_rotation_gate(
        self, targets: List[int], pauli_ids: List[int]
    ) -> Parameter:
        """Add a trainable multi-qubit Pauli rotation gate.

        Note: this gate is not integrated with the share_with mechanism.
        """
        return self.circuit.add_ParametricPauliRotation_gate(targets, pauli_ids)

    # --- Parametric-input gates (angle = f(theta, x)) ----------------------

    def add_parametric_input_RX_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RX gate whose angle depends on both a learnable parameter and input data."""
        self._add_parametric_input_R_gate_inner(index, _Axis.X, input_func)

    def add_parametric_input_RY_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RY gate whose angle depends on both a learnable parameter and input data."""
        self._add_parametric_input_R_gate_inner(index, _Axis.Y, input_func)

    def add_parametric_input_RZ_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RZ gate whose angle depends on both a learnable parameter and input data."""
        self._add_parametric_input_R_gate_inner(index, _Axis.Z, input_func)

    # --- Private builders ---------------------------------------------------

    def _add_R_gate_inner(self, index: int, angle: float, target: _Axis) -> None:
        if target == _Axis.X:
            self.circuit.add_RX_gate(index, angle)
        elif target == _Axis.Y:
            self.circuit.add_RY_gate(index, angle)
        elif target == _Axis.Z:
            self.circuit.add_RZ_gate(index, angle)
        else:
            raise NotImplementedError

    def _add_parametric_axis_gate(self, index: int, target: _Axis) -> None:
        if target == _Axis.X:
            self.circuit.add_ParametricRX_gate(index)
        elif target == _Axis.Y:
            self.circuit.add_ParametricRY_gate(index)
        elif target == _Axis.Z:
            self.circuit.add_ParametricRZ_gate(index)
        else:
            raise ValueError("Invalid target axis")

    def _add_input_R_gate_inner(self, index: int, target: _Axis, input_function: InputFunc) -> None:
        new_gate_pos = self._new_parameter_position()
        self._registry.register_input_param(
            new_gate_pos, input_function, companion_parameter_id=None
        )
        self._add_parametric_axis_gate(index, target)

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        share_with: Optional[int],
        share_with_coef: Optional[float],
    ) -> int:
        new_gate_pos = self._new_parameter_position()
        parameter_id = self._registry.register_learning_param(
            gate_pos=new_gate_pos, share_with=share_with, coef=share_with_coef
        )
        self._add_parametric_axis_gate(index, target)
        return parameter_id

    def _add_parametric_input_R_gate_inner(
        self, index: int, target: _Axis, input_func: InputFuncWithParam
    ) -> None:
        new_gate_pos = self._new_parameter_position()
        parameter_id = self._registry.register_learning_param(
            gate_pos=new_gate_pos, share_with=None, coef=None, is_input=True
        )
        self._registry.register_input_param(
            new_gate_pos, input_func, companion_parameter_id=parameter_id
        )
        self._add_parametric_axis_gate(index, target)

    # --- Counts & index queries --------------------------------------------

    @property
    def parameter_count(self) -> int:
        """Total number of parametric slots in the underlying circuit (input + learnable)."""
        return self.circuit.parameter_count

    @property
    def input_params_count(self) -> int:
        """Number of input-data-driven parameters."""
        return self._registry.input_params_count

    @property
    def learning_params_count(self) -> int:
        """Number of unique learnable parameters (i.e. the length of the theta vector)."""
        return self._registry.learning_params_count

    def get_learning_params_indexes(self) -> List[int]:
        """Circuit-level indices of all learnable parameter slots.

        Positions belonging to the same parameter via share_with appear separately.
        """
        return self._registry.learning_param_positions()

    def get_minimum_learning_param_indexes(self) -> List[int]:
        """One representative circuit-level index per unique learnable parameter."""
        return self._registry.minimum_learning_param_positions()

    def get_input_params_indexes(self) -> List[int]:
        """Circuit-level indices of all input-data-driven parameter slots."""
        return self._registry.input_param_positions()

    def get_learning_param_grad_aggregators(self) -> List[List[Tuple[int, float]]]:
        """For each learning parameter, list ``(gate_pos, coef)`` tuples that
        must be summed to compose the per-learning-param gradient from per-gate
        gradients. ``share_with`` produces multiple entries per learning param.
        """
        result: List[List[Tuple[int, float]]] = []
        for lp in self._registry.learning_parameters:
            result.append(
                [
                    (pos.gate_pos, pos.coef if pos.coef is not None else 1.0)
                    for pos in lp.positions_in_circuit
                ]
            )
        return result

    # --- Binding & resolution ----------------------------------------------

    def bind_input_and_parameters(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> ImmutableBoundParametricQuantumCircuit:
        """Bind input data and learnable parameters to produce a concrete circuit."""
        bound_parameters = self.generate_bound_params(x, parameters)
        return self.circuit.bind_parameters(bound_parameters)

    def generate_bound_params(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> Sequence[float]:
        """Compute the full gate-level parameter list from input data and learnable parameters."""
        return list(self._registry.resolve_bound(x, parameters))

    def to_batched(
        self, data: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> Tuple[UnboundParametricQuantumCircuit, NDArray[np.float64]]:
        """Build a batched parameter array for use with scaluq (quri-parts-scaluq).

        Returns:
            Tuple of ``(circuit, batched_params)`` where ``batched_params`` has shape
            ``(n_data, parameter_count)``.
        """
        return self.circuit, self._registry.resolve_batched(data, parameters)

    def to_batched_for_gradient(
        self,
        data: NDArray[np.float64],
        parameters: NDArray[np.float64],
        delta: float = 1e-5,
    ) -> Tuple[UnboundParametricQuantumCircuit, NDArray[np.float64]]:
        """Build shifted parameter arrays for numerical gradient estimation.

        For each sample and each learning parameter, creates two rows with the
        parameter shifted by +delta/2 and -delta/2.

        Row layout: for sample i and learning param j,
          plus  row = i * 2 * learning_params_count + 2 * j
          minus row = i * 2 * learning_params_count + 2 * j + 1
        """
        n_samples = len(data)
        n_learning = self.learning_params_count

        _, base_params = self.to_batched(data, parameters)
        shifted_params = np.repeat(base_params, 2 * n_learning, axis=0)

        half_delta = delta / 2
        sample_offsets = np.arange(n_samples, dtype=np.int64) * (2 * n_learning)
        for j, lp in enumerate(self._registry.learning_parameters):
            plus_rows = sample_offsets + 2 * j
            minus_rows = plus_rows + 1
            gate_positions = np.array([p.gate_pos for p in lp.positions_in_circuit], dtype=np.int64)
            coefs = np.array(
                [p.coef if p.coef is not None else 1.0 for p in lp.positions_in_circuit],
                dtype=np.float64,
            )
            shifts = half_delta * coefs
            np.add.at(
                shifted_params, (plus_rows[:, None], gate_positions[None, :]), shifts[None, :]
            )
            np.add.at(
                shifted_params, (minus_rows[:, None], gate_positions[None, :]), -shifts[None, :]
            )

        return self.circuit, shifted_params

    # --- Backprop gradient (qulacs) ----------------------------------------

    def backprop_inner_product(
        self, x: NDArray[np.float64], theta: NDArray[np.float64], state: QulacsQuantumState
    ) -> NDArray[np.float64]:
        """Compute gradients of learnable parameters via qulacs backpropagation using inner product.

        Args:
            x: Input data array.
            theta: Learnable parameter vector of length ``learning_params_count``.
            state: Target qulacs quantum state used in the inner product.

        Returns:
            Gradient array of shape ``(learning_params_count,)``.
        """
        params = self.generate_bound_params(x, theta)
        (qulacs_circuit, param_mapper) = convert_parametric_circuit(self.circuit)
        for i, v in enumerate(param_mapper(params)):
            qulacs_circuit.set_parameter(i, v)
        ret = qulacs_circuit.backprop_inner_product(state)
        ans = np.zeros(self.learning_params_count)
        for param in self._registry.learning_parameters:
            if param.is_input:
                continue
            for pos in param.positions_in_circuit:
                ans[param.parameter_id] += ret[pos.gate_pos] * (pos.coef or 1.0)
        return ans

    # Deprecated alias for the old typo'd name. Remove once callers migrate.
    backprop_innner_product = backprop_inner_product

    # --- Hadamard test gradient --------------------------------------------

    def _calc_gradient_observable(
        self,
        generator: _Axis,
        qubit_index: int,
        hamiltonian: Operator,
    ) -> Operator:
        """Calculate the gradient observable O_j = i[G_j, H]."""
        symbol = {_Axis.X: "X", _Axis.Y: "Y", _Axis.Z: "Z"}[generator]
        generator_operator = Operator({pauli_label(f"{symbol}{qubit_index}"): 0.5})
        return 1j * commutator(generator_operator, hamiltonian)

    def _get_gate_axis(self, gate: QuantumGate) -> _Axis:
        match gate.name:
            case "ParametricRX":
                return _Axis.X
            case "ParametricRY":
                return _Axis.Y
            case "ParametricRZ":
                return _Axis.Z
            case _:
                raise NotImplementedError("Unknown gate type found: ", gate.name)

    def _apply_gates_to_qc(
        self,
        qc: QuantumCircuit,
        gates: Sequence[QuantumGate],
        parameters: Sequence[float],
    ) -> None:
        """Apply gates with parameters to QuantumCircuit."""
        i = 0
        for gate in gates:
            if isinstance(gate, QuantumGate):
                qc.add_gate(gate)
            elif isinstance(gate, ParametricQuantumGate):
                param = parameters[i]
                g_axis = self._get_gate_axis(gate)
                g_qubit = gate.target_indices[0]
                match g_axis:
                    case _Axis.X:
                        qc.add_RX_gate(g_qubit, param)
                    case _Axis.Y:
                        qc.add_RY_gate(g_qubit, param)
                    case _Axis.Z:
                        qc.add_RZ_gate(g_qubit, param)
                i += 1
            else:
                raise NotImplementedError("Unknown gate type found: ", gate.name)

    def _get_inverse_gate(self, gate: QuantumGate) -> QuantumGate:
        return QuantumGate(
            name=gate.name,
            target_indices=gate.target_indices,
            control_indices=gate.control_indices,
            classical_indices=gate.classical_indices,
            params=[-p for p in gate.params],
            pauli_ids=gate.pauli_ids,
            unitary_matrix=gate.unitary_matrix,
        )

    def _create_hadamard_test_circuit(
        self,
        x: NDArray[np.float64],
        theta: NDArray[np.float64],
        gate_index: int,
    ) -> QuantumCircuit:
        """Create a circuit for Hadamard test.

        When differentiating with respect to θj:
        U = U{>j} Uj(θj) U{<j}, G is the generator of Uj(θj): RX->X/2, RY->Y/2, RZ->Z/2.
        The circuit is: U{>j} control{G} U†{>j} U |+ψ〉.
        """
        _circuit = QuantumCircuit(self.n_qubits + 1)
        ancilla_index = self.n_qubits
        _circuit.add_H_gate(ancilla_index)
        bound_params = self.generate_bound_params(x, theta)
        gates_length = len(self.circuit.gates)

        # U |+ψ〉
        self._apply_gates_to_qc(_circuit, self.circuit.gates, bound_params)

        # U†{>j}
        gates_backward: List[QuantumGate] = []
        params_backward: List[float] = []
        j = len([_ for _ in self.circuit.gates if isinstance(_, ParametricQuantumGate)])
        for i in range(gates_length - 1, gate_index, -1):
            gate = self.circuit.gates[i]
            if isinstance(gate, QuantumGate):
                gates_backward.append(self._get_inverse_gate(gate))
            elif isinstance(gate, ParametricQuantumGate):
                gates_backward.append(gate)
                params_backward.append(-bound_params[j - 1])
                j -= 1
        self._apply_gates_to_qc(_circuit, gates_backward, params_backward)

        # control{G}
        gate = self.circuit.gates[gate_index]
        if isinstance(gate, ParametricQuantumGate):
            axis = self._get_gate_axis(gate)
            target_qubit = gate.target_indices[0]
            match axis:
                case _Axis.X:
                    _circuit.add_CNOT_gate(ancilla_index, target_qubit)
                case _Axis.Y:
                    _circuit.add_Sdag_gate(target_qubit)
                    _circuit.add_CNOT_gate(ancilla_index, target_qubit)
                    _circuit.add_S_gate(target_qubit)
                case _Axis.Z:
                    _circuit.add_CZ_gate(ancilla_index, target_qubit)
                case _:
                    raise NotImplementedError

        # U{>j}
        gates_forward: List[QuantumGate] = []
        params_forward: List[float] = []
        for i in range(gate_index + 1, gates_length):
            gate = self.circuit.gates[i]
            gates_forward.append(gate)
            if isinstance(gate, ParametricQuantumGate):
                params_forward.append(bound_params[j])
                j += 1
        self._apply_gates_to_qc(_circuit, gates_forward, params_forward)

        return _circuit

    def _calc_hadamard_gradient_observable(self, operator: Operator) -> Operator:
        # O ⊗ Y
        result_terms: dict = {}
        for p1, c1 in operator.items():
            new_label = pauli_label(f"{str(p1)} Y{self.n_qubits}")
            result_terms[new_label] = result_terms.get(new_label, 0) + c1
        return Operator(result_terms)

    def hadamard_gradient(
        self,
        x: NDArray[np.float64],
        theta: NDArray[np.float64],
        operator: Operator,
        estimator: ConcurrentQuantumEstimator,
    ) -> NDArray[np.float64]:
        """Compute gradients of learnable parameters via the Hadamard test.

        Args:
            x: Input data array.
            theta: Learnable parameter vector of length ``learning_params_count``.
            operator: Observable whose expectation value gradient is computed.
            estimator: Concurrent quantum estimator used to evaluate the Hadamard-test circuits.

        Returns:
            Gradient array of shape ``(learning_params_count,)``.
        """
        operator = self._calc_hadamard_gradient_observable(operator)
        learning_param_indexes = self.get_learning_params_indexes()

        states = []
        param_gate_count = -1
        for i, gate in enumerate(self.circuit.gates):
            if not isinstance(gate, ParametricQuantumGate):
                continue
            param_gate_count += 1
            if param_gate_count not in learning_param_indexes:
                continue
            test_circuit = self._create_hadamard_test_circuit(x, theta, i)
            states.append(GeneralCircuitQuantumState(self.n_qubits + 1, test_circuit))

        operators = [operator] * len(states)
        results = estimator(operators, states)
        return np.array([res.value for res in results])
