from quri_parts.circuit import NonParametricQuantumCircuit
from typing import Optional
from quri_parts.core.sampling import Sampler, MeasurementCounts
from quri_parts_oqtopus.backend import OqtopusConfig, OqtopusSamplingBackend, OqtopusSamplingResult


class OqtopusSampler:
    def __init__(self, device_id: str, config: Optional[OqtopusConfig]) -> None:
        self.backend = OqtopusSamplingBackend(config)
        self.device_id = device_id
        self.config = config

    def sample(
        self, circuit: NonParametricQuantumCircuit | list[NonParametricQuantumCircuit], shots: int
    ) -> MeasurementCounts:
        """
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        result = self.backend.sample(circuit, device_id=self.device_id, shots=shots).result()
        counts = result.counts
        return counts


def create_oqtopus_sampler(device_id: str, config: Optional[OqtopusConfig] = None) -> Sampler:
    """Oqtopus用のSamplerを生成する関数

    Returns:
        Sampler: Oqtopus用のSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, config)
    return oqtopusSampler.sample
