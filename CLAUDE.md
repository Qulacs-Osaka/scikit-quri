# CLAUDE.md

このファイルはClaude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

scikit-quriは[quri-parts](https://quri-parts.qunasys.com/)をベースにした量子機械学習ライブラリです。量子ニューラルネットワーク（QNN）、量子サポートベクターマシン（QSVM）、量子カーネルリッジ回帰（QKRR）の実装を提供し、scikit-learn風のインターフェースを採用しています。

## 開発コマンド

```bash
# 依存関係のインストール
uv sync

# 全テストの実行
make test

# 単一テストファイルの実行
make tests/test_qnn_classifier.py

# フォーマットとリントの修正
make fix

# フォーマットとリントのチェック（CI検証用）
make check

# カバレッジレポートの生成
make cov

# ドキュメントのビルド
make html
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│           MLアルゴリズム層 (qnn/, qsvm/, qkrr/)         │
│  QNNClassifier, QNNRegressor, QSVC, QSVR, QKRR        │
├─────────────────────────────────────────────────────────┤
│              回路層 (circuit/)                          │
│  LearningCircuit + 事前定義アンサッツ                   │
├─────────────────────────────────────────────────────────┤
│              バックエンド層 (backend/)                  │
│  推定器 (Sim/Oqtopus) + 勾配推定器                     │
├─────────────────────────────────────────────────────────┤
│        外部ライブラリ: quri-parts, scikit-learn        │
└─────────────────────────────────────────────────────────┘
```

**主要モジュール:**
- `backend/`: 期待値と勾配の抽象・具象推定器実装。シミュレーション用の`SimEstimator`、量子ハードウェア用の`OqtopusEstimator`。
- `circuit/`: `LearningCircuit`はパラメトリック量子回路クラス。事前定義回路にはQCLアンサッツ、Farhi-Neven、IBM埋め込みが含まれる。
- `qnn/`: パウリZ期待値を使用した量子ニューラルネットワークモデル（`QNNClassifier`、`QNNRegressor`、`QNNGenerator`）。
- `qsvm/`: 量子カーネルサポートベクターマシン（`QSVC`、`QSVR`）。
- `qkrr/`: 量子カーネルリッジ回帰。
- `state/`: 量子状態計算用のオーバーラップ推定器。

## コードスタイル

- 行の長さ: 100文字
- フォーマッター: Ruff
- 型チェック: MyPy（厳格設定）
- Pythonバージョン: 3.10-3.11
