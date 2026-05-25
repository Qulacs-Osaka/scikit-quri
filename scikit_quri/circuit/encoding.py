"""Input-data encoding helpers used by predefined ansatz factories.

These map a single input feature vector to a sequence of rotation angles by
cycling through its elements; ``clamped_cyclic_index`` additionally clamps each
value to ``[-1, 1]`` so it is safe to feed into ``arcsin`` / ``arccos``.
"""

import numpy as np
from numpy.typing import NDArray


def cyclic_index(x: NDArray[np.float64], i: int) -> float:
    """Return ``x[i % len(x)]`` as a float."""
    return float(x[i % len(x)])


def clamped_cyclic_index(x: NDArray[np.float64], i: int) -> float:
    """Cyclic index of ``x`` clamped to ``[-1, 1]``."""
    xa = float(x[i % len(x)])
    return min(1.0, max(-1.0, xa))
