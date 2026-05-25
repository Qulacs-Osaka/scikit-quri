import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.sampling import MeasurementCounts
from quri_parts_oqtopus.backend import (
    OqtopusConfig,
    OqtopusDeviceBackend,
    OqtopusSamplingBackend,
)

from .base_sampler import BaseSampler


class OqtopusSampler(BaseSampler):
    def __init__(self, device_id: str, config: Optional[OqtopusConfig]) -> None:
        self.backend = OqtopusSamplingBackend(config)
        self.device_id = device_id
        self.config = config
        self.device_backend = OqtopusDeviceBackend(config)

    def get_device_qubit_count(self) -> int:
        """Oqtopusのデバイスの量子ビット数を取得する関数

        Returns:
            int: Oqtopusのデバイスの量子ビット数

        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        device_info = self.device_backend.get_device(self.device_id)
        return device_info.n_qubits

    def sample_one(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> MeasurementCounts:
        """Sample a single circuit ``n_shots`` times.

        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        result = self.backend.sample(circuit, device_id=self.device_id, shots=n_shots).result()
        return result.counts

    def sample(
        self,
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        """
        concurrentにsampleする関数
        quri-partsのConcurrentSamplerに合わせるため、shotsはtupleの最大値を使用する
        devicesの量子ビット数を超える量子回路の組み合わせに対しては、複数回に分けて実行する
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        circuit_shots_list = list(circuit_shots_tuples)
        max_shots = max(shots for _, shots in circuit_shots_list)
        circuits = [circuit for circuit, _ in circuit_shots_list]
        # batched_circuits = [[]]
        # device_qubits = self.get_device_qubit_count()
        # prefix_qubits = 0
        # for circuit in circuits:
        #     prefix_qubits += circuit.qubit_count
        #     if prefix_qubits > device_qubits:
        #         prefix_qubits = circuit.qubit_count
        #         batched_circuits.append([circuit])
        #     else:
        #         batched_circuits[-1].append(circuit)
        #! qiskitのtranspilerで物理layoutを考慮してしまうので、combineすると相互作用でoverlapが1.0じゃなくなる
        #! qulacsのsimでも物理layoutが考慮されているっぽい？
        batched_circuits = [[c] for c in circuits]
        counts = []
        transpiler_info = {
            "transpiler_lib": "qiskit",
            "transpiler_options": {
                "optimization_level": 2,
            },
        }

        def submit_sample(
            batched_circuits: list[NonParametricQuantumCircuit],
        ) -> dict[int, MeasurementCounts]:
            req = self.backend.sample(
                batched_circuits,
                device_id=self.device_id,
                shots=max_shots,
                transpiler_info=transpiler_info,
            ).result()
            if req.divided_counts is None:
                raise RuntimeError("Expected divided_counts for list of circuits.")
            return req.divided_counts

        batched_result: list[dict[int, MeasurementCounts]] = []
        # SamplerをThreadPoolExecutorで並列化して実行
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(submit_sample, batch) for batch in batched_circuits]
            c = 0
            for _ in as_completed(futures):
                c += 1
                print("\r", f"succeeded: {c}/{len(batched_circuits)}", end="")
            batched_result = [future.result() for future in futures]
            print()

        for divided_counts in batched_result:
            # dictのkey(int)でsortしてから格納
            counts.extend(v for _, v in sorted(divided_counts.items()))
        return counts

    def concurrent_sample(
        self,
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        """Deprecated alias of :meth:`sample`. Kept for backward compatibility."""
        warnings.warn(
            "OqtopusSampler.concurrent_sample is deprecated; use .sample(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sample(circuit_shots_tuples)


def create_oqtopus_sampler(device_id: str, config: Optional[OqtopusConfig] = None):
    """Oqtopus用のSamplerを生成する関数
    quri-partsのSamplerとして動作

    Returns:
        Sampler: Oqtopus用のSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, config)
    return oqtopusSampler.sample_one


def create_oqtopus_concurrent_sampler(device_id: str, config: Optional[OqtopusConfig] = None):
    """Oqtopus用のConcurrentSamplerを生成する関数
    quri-partsのConcurrentSamplerとして動作
    Returns:
        ConcurrentSampler: Oqtopus用のConcurrentSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, config)
    return oqtopusSampler.sample
