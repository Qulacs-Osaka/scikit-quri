from quri_parts.circuit import NonParametricQuantumCircuit
from typing import Optional, Iterable
from quri_parts.core.sampling import MeasurementCounts
from quri_parts_oqtopus.backend import OqtopusConfig, OqtopusSamplingBackend


class OqtopusSampler:
    def __init__(self, device_id: str, config: Optional[OqtopusConfig]) -> None:
        self.backend = OqtopusSamplingBackend(config)
        self.device_id = device_id
        self.config = config

    def sample(
        self, circuit: NonParametricQuantumCircuit | list[NonParametricQuantumCircuit], shots: int
    ) -> MeasurementCounts:
        # self, circuits: list[NonParametricQuantumCircuit], shots: int) -> Iterable[MeasurementCounts]:
        """
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        result = self.backend.sample(circuit, device_id=self.device_id, shots=shots).result()
        return result.counts

        # result = self.backend.sample(circuits, device_id=self.device_id, shots=shots).result()
        # counts: dict[int, MeasurementCounts] = {}
        # if isinstance(circuits, list):
        #     if result.divided_counts is None:
        #         raise RuntimeError("Expected divided_counts for list of circuits.")
        #     counts = result.divided_counts
        # else:
        #     counts = {0: result.counts}
        # return [counts[i] for i in range(len(counts))]


# Sampler :TypeAlias = Callable[[NonParametricQuantumCircuit | list[NonParametricQuantumCircuit], int], MeasurementCounts]
# TODO これ、samplerの型とlist[circuit]の型があってない
def create_oqtopus_sampler(device_id: str, config: Optional[OqtopusConfig] = None):
    """Oqtopus用のSamplerを生成する関数

    Returns:
        Sampler: Oqtopus用のSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, config)
    return oqtopusSampler.sample
