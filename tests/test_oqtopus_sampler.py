import pytest
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from scikit_quri.backend.oqtopus_sampler import (
    OqtopusSampler,
    create_oqtopus_sampler,
    create_oqtopus_concurrent_sampler,
)


def create_simple_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_CNOT_gate(0, 1)
    return circuit


def test_oqtopus_sampler_init() -> None:
    """OqtopusSamplerの初期化テスト"""
    sampler = OqtopusSampler(device_id="qulacs", config=None)
    assert sampler.device_id == "qulacs"
    assert sampler.config is None
    assert sampler.backend is not None


@pytest.mark.oqtopus
def test_create_oqtopus_sampler() -> None:
    """create_oqtopus_sampler関数のテスト"""
    sampler = create_oqtopus_sampler(device_id="qulacs")
    assert callable(sampler)


@pytest.mark.oqtopus
def test_oqtopus_sampler() -> None:
    """create_oqtopus_samplerで生成したSamplerの単一回路サンプリングテスト"""
    circuit: NonParametricQuantumCircuit = create_simple_circuit()
    sampler = create_oqtopus_sampler(device_id="qulacs")
    counts = sampler(circuit, 500)
    assert counts is not None
    assert sum(counts.values()) == 500


@pytest.mark.oqtopus
def test_oqtopus_concurrent_sampler() -> None:
    """create_oqtopus_concurrent_sampler関数のテスト"""
    circuits: list[NonParametricQuantumCircuit] = [create_simple_circuit(), create_simple_circuit()]
    concurrent_sampler = create_oqtopus_concurrent_sampler(device_id="qulacs", is_combine=False)
    counts_list = list(concurrent_sampler(zip(circuits, [500, 500])))
    assert len(counts_list) == 2
    for counts in counts_list:
        assert counts is not None
        assert sum(counts.values()) == 500


@pytest.mark.oqtopus
def test_svc_oqtopus_sampler() -> None:
    """30個のデータポイントを使用して、OqtopusSamplerを用いたQSVCのテスト"""
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from scikit_quri.circuit import create_ibm_embedding_circuit
    from scikit_quri.qsvm import QSVC
    import pandas as pd

    N_train = 5
    N_test = 5
    iris = datasets.load_iris(as_frame=True)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        iris.target,
        test_size=0.25,
        random_state=0,
    )
    x_train = x_train.to_numpy()[:N_train]
    x_test = x_test.to_numpy()[:N_test]
    y_train = y_train[:N_train]
    y_test = y_test[:N_test]
    n_qubit = 4  # x_train の次元数以上必要。あまり小さいと結果が悪くなる。
    circuit = create_ibm_embedding_circuit(n_qubit)
    sampler = create_oqtopus_concurrent_sampler(device_id="qulacs", is_combine=False)
    qsvm = QSVC(circuit)
    qsvm.fit(x_train, y_train, sampler, n_shots=1024)
    y_pred = qsvm.predict(x_test)
    score = f1_score(y_test, y_pred, average="weighted")
    assert score > 0.9
