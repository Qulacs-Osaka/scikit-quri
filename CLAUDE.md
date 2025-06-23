# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 開発環境のセットアップ

このプロジェクトは量子ニューラルネットワークライブラリで、quri-partsをベースにしています。Python 3.9.8以上、3.12未満が必要です。

```bash
poetry install
```

## 開発コマンド

### テスト実行
```bash
make test           # 全テスト実行
make tests/test_file.py  # 単一テストファイル実行
poetry run pytest tests/test_file.py::test_function  # 特定のテスト関数実行
```

### コード品質チェック
```bash
make check          # フォーマットとリントのチェック（差分表示）
make fix            # フォーマットとリントの自動修正
poetry run ruff format  # フォーマット実行
poetry run ruff check   # リントチェック
```

### カバレッジ
```bash
make cov            # カバレッジレポート生成（HTML）
make serve_cov      # カバレッジレポートをHTTPサーバーで提供
```

### ドキュメント
```bash
make html           # Sphinxドキュメント生成
```

## アーキテクチャ

scikit-quriは4つの主要モジュールで構成されています：

### circuit/
- `LearningCircuit`: 学習可能な量子回路の基底クラス
- `create_qcl_ansatz()`, `create_farhi_neven_ansatz()`: 事前定義された量子回路アンサッツ

### qnn/ (Quantum Neural Networks)
- `QNNClassifier`: 量子ニューラルネットワーク分類器
- `QNNRegressor`: 量子ニューラルネットワーク回帰器  
- `QNNGenerator`: 量子ニューラルネットワーク生成器

### qsvm/ (Quantum Support Vector Machines)
- `QSVC`: 量子サポートベクター分類器
- `QSVR`: 量子サポートベクター回帰器

### qkrr/ (Quantum Kernel Ridge Regression)
- `QKRR`: 量子カーネルリッジ回帰

### state/
- `overlap_estimator.py`: 量子状態重複推定器
- `overlap_estimator_real_device.py`: 実デバイス用重複推定器

## 依存関係

- quri-parts: 量子計算フレームワーク（Qulacsエクストラ付き）
- scikit-learn: 機械学習インターフェース互換性
- numpy, matplotlib, pandas: データ処理と可視化

典型的な使用例では、quri-partsのestimatorとoptimizer、scikit-quriの回路とモデルクラスを組み合わせて量子機械学習アルゴリズムを実装します。