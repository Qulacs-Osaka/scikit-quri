"""Backward-compatible factory for the old ``SimEstimator(use_scaluq=...)`` API.

Prefer constructing :class:`~scikit_quri.backend.QulacsEstimator` or
:class:`~scikit_quri.backend.ScaluqEstimator` directly. ``SimEstimator`` is
kept so that existing call sites (tests, samples, downstream code) continue
to work without modification.
"""

import warnings

from .base_estimator import BaseEstimator
from .qulacs_estimator import QulacsEstimator
from .scaluq_estimator import ScaluqEstimator


def SimEstimator(use_scaluq: bool = False) -> BaseEstimator:
    """Construct a simulation estimator.

    Args:
        use_scaluq: If True, return a :class:`ScaluqEstimator` (batched scaluq
            backend). Otherwise return a :class:`QulacsEstimator`
            (per-sample qulacs backend).

    Returns:
        Either a ``QulacsEstimator`` or a ``ScaluqEstimator`` depending on
        ``use_scaluq``.

    .. deprecated::
        Construct ``QulacsEstimator()`` or ``ScaluqEstimator()`` directly.
    """
    warnings.warn(
        "SimEstimator is deprecated; use QulacsEstimator() or ScaluqEstimator() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ScaluqEstimator() if use_scaluq else QulacsEstimator()
