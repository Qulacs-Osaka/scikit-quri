import numpy as np
from scikit_quri.circuit import LearningCircuit
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.estimator.sampling import create_sampling_concurrent_estimator
from quri_parts.core.sampling.shots_allocator import create_proportional_shots_allocator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement

pi = np.pi

# Learning Parameters
input_params = []
learning_params = [pi/4, pi/3]

operator = Operator({
    pauli_label("Z0"): 1.0,
    pauli_label("Z1"): 1.0
    })

# Create LearningCircuit instance
learning_circuit = LearningCircuit(2)
learning_circuit.add_H_gate(0)
learning_circuit.add_CNOT_gate(0, 1)
learning_circuit.add_parametric_RX_gate(0)
learning_circuit.add_parametric_RY_gate(1)

# Sampler and Estimator
from concurrent.futures import ThreadPoolExecutor
from quri_parts_oqtopus.sampler import create_oqtopus_concurrent_sampler

shots = 1024

concurrency = len(learning_params)
executor = ThreadPoolExecutor(concurrency)
sampler = create_oqtopus_concurrent_sampler(executor, concurrency)

measurement_factory = bitwise_commuting_pauli_measurement
shots_allocator = create_proportional_shots_allocator()

# Execute hadamard_grad
estimator = create_sampling_concurrent_estimator(
    total_shots=shots,
    sampler=sampler,
    measurement_factory=measurement_factory,
    shots_allocator=shots_allocator,
    )

gradients = learning_circuit.hadamard_gradient(
    np.array(input_params),
    np.array(learning_params),
    operator,
    estimator,
)

print("Gradients:", gradients)
