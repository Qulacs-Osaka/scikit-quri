from concurrent.futures import ThreadPoolExecutor, as_completed

from quri_parts.circuit import NonParametricQuantumCircuit
from typing import Optional, Iterable
from quri_parts.core.sampling import MeasurementCounts
from quri_parts_oqtopus.backend import (
    OqtopusConfig,
    OqtopusSamplingBackend,
    OqtopusDeviceBackend,
)


class OqtopusSampler:
    def __init__(
        self,
        device_id: str,
        is_combine: Optional[bool] = True,
        config: Optional[OqtopusConfig] = None,
    ) -> None:
        """OqtopusSamplerを初期化する

        Args:
            device_id (str): 使用するOqtopusデバイスのID
            is_combine (bool): 複数の量子回路をcombineして実行するかどうか
            config (Optional[OqtopusConfig]): Oqtopusの接続設定。Noneの場合はデフォルト設定を使用する
        """
        self.backend = OqtopusSamplingBackend(config)
        self.device_id = device_id
        self.is_combine = is_combine
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

    def sample(self, circuit: NonParametricQuantumCircuit, shots: int) -> MeasurementCounts:
        """
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        result = self.backend.sample(circuit, device_id=self.device_id, shots=shots).result()
        return result.counts

    def concurrent_sample(
        self, circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        """
        concurrentにsampleする関数
        quri-partsのConcurrentSamplerに合わせるため、shotsはtupleの最大値を使用する
        is_combineがTrueの場合、複数のbatchに分けて実行する際に、
        devicesの量子ビット数を超える量子回路の組み合わせに対しては、複数回に分けて実行する
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        circuit_shots_list = list(circuit_shots_tuples)
        max_shots = max(shots for _, shots in circuit_shots_list)
        circuits = [circuit for circuit, _ in circuit_shots_list]
        batched_circuits = [[]]
        #! qiskitのtranspilerで物理layoutを考慮してしまうので、combineすると相互作用でoverlapが1.0じゃなくなる
        #! qulacsのsimでも物理layoutが考慮されているっぽい？
        if self.is_combine:
            device_qubits = self.get_device_qubit_count()
            prefix_qubits = 0
            for circuit in circuits:
                prefix_qubits += circuit.qubit_count
                if prefix_qubits > device_qubits:
                    prefix_qubits = circuit.qubit_count
                    batched_circuits.append([circuit])
                else:
                    batched_circuits[-1].append(circuit)
        else:
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


def create_oqtopus_sampler(
    device_id: str, is_combine: bool = True, config: Optional[OqtopusConfig] = None
):
    """Oqtopus用のSamplerを生成する関数
    quri-partsのSamplerとして動作

    Args:
        device_id (str): 使用するOqtopusデバイスのID
        is_combine (bool): 複数の量子回路をcombineして実行するかどうか。デフォルトはTrue
        config (Optional[OqtopusConfig]): Oqtopusの接続設定。Noneの場合はデフォルト設定を使用する

    Returns:
        Sampler: Oqtopus用のSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, is_combine, config)
    return oqtopusSampler.sample


def create_oqtopus_concurrent_sampler(
    device_id: str, is_combine: bool = True, config: Optional[OqtopusConfig] = None
):
    """Oqtopus用のConcurrentSamplerを生成する関数
    quri-partsのConcurrentSamplerとして動作

    Args:
        device_id (str): 使用するOqtopusデバイスのID
        is_combine (bool): 複数の量子回路をcombineして実行するかどうか。デフォルトはTrue
        config (Optional[OqtopusConfig]): Oqtopusの接続設定。Noneの場合はデフォルト設定を使用する

    Returns:
        ConcurrentSampler: Oqtopus用のConcurrentSampler
    """
    oqtopusSampler = OqtopusSampler(device_id, is_combine, config)
    return oqtopusSampler.concurrent_sample
