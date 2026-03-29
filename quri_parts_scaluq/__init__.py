# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
from typing import Any, Sequence, Union, cast

from numpy.typing import ArrayLike
from typing_extensions import TypeAlias, TypeVar

from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)


scaluqStateT: TypeAlias = Union[
    CircuitQuantumState,
    QuantumStateVector,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
]
scaluqParametricStateT: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]

Numerics = TypeVar("Numerics", int, float, complex)


def cast_to_list(int_sequence: Union[Sequence[Numerics], ArrayLike]) -> list[Numerics]:
    return cast(list[Numerics], int_sequence)


# Select precision from environment variable
_precision = os.environ.get("SCALUQ_PRECISION", "f64").lower()
if _precision not in ["f32", "f64"]:
    raise ImportError(
        f"Invalid value '{_precision}' for environment variable SCALUQ_PRECISION."
        " Choose 'f32' or 'f64'."
    )

# Dynamically import the selected backend module
_module_name = f"scaluq.default.{_precision}"
try:
    _backend: Any = importlib.import_module(_module_name)
except ImportError as e:
    raise ImportError(
        f"Failed to import scaluq backend '{_module_name}'."
    ) from e


def get_scaluq_accuracy() -> str:
    return _precision
