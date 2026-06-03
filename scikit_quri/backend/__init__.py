import os

# scaluq is built on Kokkos, which prints an "OMP_PROC_BIND ... not set" advisory to
# stderr at OpenMP initialization when the variable is unset. Provide an overridable
# default before scaluq is imported below so the import stays quiet. "false" disables
# thread pinning, which avoids clashing with other OpenMP runtimes (BLAS, etc.) in the
# same process; set OMP_PROC_BIND=spread and OMP_PLACES=threads yourself for maximum
# scaluq throughput.
os.environ.setdefault("OMP_PROC_BIND", "false")

from .base_estimator import BaseEstimator, BatchedSimEstimator
from .base_sampler import BaseSampler
from .oqtopus_estimator import OqtopusEstimator
from .oqtopus_gradient_estimator import OqtopusGradientEstimator
from .oqtopus_sampler import (
    OqtopusSampler,
    create_oqtopus_concurrent_sampler,
    create_oqtopus_sampler,
)
from .qulacs_estimator import QulacsEstimator
from .qulacs_sampler import QulacsSampler
from .scaluq_estimator import ScaluqEstimator
from .sim_estimator import SimEstimator
from .sim_gradient_estimator import SimGradientEstimator

__all__ = [
    "BaseEstimator",
    "BaseSampler",
    "BatchedSimEstimator",
    "OqtopusEstimator",
    "OqtopusGradientEstimator",
    "OqtopusSampler",
    "QulacsEstimator",
    "QulacsSampler",
    "ScaluqEstimator",
    "SimEstimator",
    "SimGradientEstimator",
    "create_oqtopus_concurrent_sampler",
    "create_oqtopus_sampler",
]
