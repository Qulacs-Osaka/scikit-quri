from typing import TypeGuard, Optional
from quri_parts.backend import SamplingBackend

def is_real_device(
    sampling_backend: Optional[SamplingBackend], is_sim: bool
) -> TypeGuard[SamplingBackend]:
    """TypeGuard for sampling_backend"""
    return not is_sim
