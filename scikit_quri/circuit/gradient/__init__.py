"""Gradient computation algorithms for LearningCircuit.

These live outside ``circuit.py`` so that the circuit-definition layer does not
depend on specific quantum backends (qulacs, etc.). Each gradient method is a
free function taking the circuit and inputs explicitly.
"""

from .adjoint import adjoint_expectation_gradients
from .backprop import backprop_inner_product
from .hadamard import hadamard_gradient

__all__ = [
    "adjoint_expectation_gradients",
    "backprop_inner_product",
    "hadamard_gradient",
]
