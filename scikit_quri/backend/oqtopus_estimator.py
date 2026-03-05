from collections.abc import Iterable, Sequence
from typing import Optional

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import (
    SequentialTranspiler,
    SingleQubitUnitaryMatrix2RYRZTranspiler,
)
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts_oqtopus.backend import OqtopusConfig, OqtopusEstimationBackend
from quri_parts.qulacs.estimator import _Estimate

from .base_estimator import BaseEstimator


class OqtopusEstimator(BaseEstimator):
    """Estimator class that computes expectation values on real quantum hardware via quri-parts-oqtopus.
    Requires an OQTOPUS configuration file at ``~/.oqtopus``.
    See: https://quri-parts-oqtopus.readthedocs.io/en/stable/usage/getting_started/#prepare-oqtopus-configuration-file

    Args:
        device_id: ID of the device to run on.
        shots: Number of shots per circuit execution. Defaults to 1000.
        config: OQTOPUS configuration. Defaults to None.

    """

    def __init__(
        self,
        device_id: str,
        shots: int = 1000,
        config: Optional[OqtopusConfig] = None,
    ) -> None:
        self.backend = OqtopusEstimationBackend(config)
        self.device_id = device_id
        self.shots = shots

    def estimate(self, operators, states):
        """Compute expectation values for combinations of operators and states.
        If either operators or states contains a single element, it is broadcast
        to match the length of the other. If both contain multiple elements, they
        must have the same length and are paired one-to-one.

        Args:
            operators: List of operators for which to compute expectation values.
            states: List of quantum states.

        Returns:
            List of expectation values for each (operator, state) pair.

        Raises:
            ValueError: If operators or states is empty, or if both have multiple
                elements with mismatched lengths.
            BackendError: If execution on OQTOPUS fails.

        """
        num_ops = len(operators)
        num_states = len(states)

        if num_ops == 0:
            raise ValueError("No operator specified.")

        if num_states == 0:
            raise ValueError("No state specified.")

        if num_ops > 1 and num_states > 1 and num_ops != num_states:
            raise ValueError(
                f"Number of operators ({num_ops}) does not matchnumber of states ({num_states}).",
            )

        if num_states == 1:
            # Reuse the same transpiled circuit for all operators (shallow copy for memory efficiency)
            circuits = [self._transpile_circuit(states[0].circuit)] * num_ops
            return self._estimate_concurrently(operators, circuits)
        if num_ops == 1:
            operators = [next(iter(operators))] * num_states
        circuits = [self._transpile_circuit(state.circuit) for state in states]
        return self._estimate_concurrently(operators, circuits)

    def _estimate_concurrently(
        self,
        operators: Sequence[Estimatable],
        circuits: Sequence[NonParametricQuantumCircuit],
    ) -> Iterable[Estimate[complex]]:
        """Compute expectation values for one-to-one pairs of operators and circuits.

        Args:
            operators: List of operators for which to compute expectation values.
            circuits: List of quantum circuits, paired with operators by index.

        Returns:
            List of expectation values for each (operator, circuit) pair.

        Raises:
            BackendError: If execution on OQTOPUS fails.

        """
        results: list[Estimate[complex]] = []
        for circuit, operator in zip(circuits, operators):
            # Normalize Estimatable to Operator
            if isinstance(operator, PauliLabel):
                operator = Operator({operator: 1.0})
            job = self.backend.estimate(
                circuit,
                operator=operator,
                device_id=self.device_id,
                shots=self.shots,
            )
            result = job.result()
            exp_real = result.exp_value
            # On failure the backend raises an exception, so exp_value is None only when the result is 0
            if exp_real is None:
                exp_real = 0.0
            results.append(_Estimate(value=complex(exp_real, 0.0)))
        return results

    def _transpile_circuit(
        self,
        circuit: NonParametricQuantumCircuit,
    ) -> NonParametricQuantumCircuit:
        """Transpile a circuit for submission to OQTOPUS.

        Args:
            circuit: Circuit before transpilation.

        Returns:
            Transpiled circuit.

        """
        transpiler = SequentialTranspiler(
            [
                # quri-parts' QASM does not support UnitaryMatrix gates; convert them to RY/RZ
                SingleQubitUnitaryMatrix2RYRZTranspiler(),
            ],
        )
        transpiled_circuit = transpiler(circuit)
        return transpiled_circuit
