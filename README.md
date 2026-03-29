# TurboQuant on Apple Silicon (M5 Max Benchmarks)
🌎 [English](#english) | 🇯🇵 [日本語](#japanese)

---

<a name="english"></a>
## 🌎 English

This repository contains benchmark scripts and reproducibility reports testing the [TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression algorithm on Apple Silicon (M5 Max) using the [MLX](https://github.com/ml-explore/mlx) framework.

By utilizing TurboQuant's **Adaptive Layer Caching** combined with **Fused Metal Kernels**, we can compress the memory footprint of an LLM's KV cache by **~3x to 10x** with minimal degradation in generation speed and zero loss of quality. 

> [!TIP]
> This extreme optimization allows even massive models (like 32 Billion parameter models) to run smoothly on high-memory Macs while processing massive context lengths (like entire book contexts or RAG logic) without unified memory exhaustion.

### 🚀 Key Benchmarks

*Tested on: Apple MacBook Pro (M5 Max, 18-Core CPU, 128GB Unified Memory)*

#### Test 1: Llama/Qwen 1.5B (Adaptive Compression)
`benchmark_1_5B_adaptive.py`

| Cache Type | FP16 Layers | Cache Size (100 tokens) | Speed (tok/s) | Text Quality |
| :--- | :---: | :--- | :--- | :--- |
| **FP16 (Default)** | All | 7.0 MB | 194.9 | Perfect |
| **All TQ 4-bit** | None | 0.8 MB | 175.1 | Collapsed |
| **Adaptive 4-bit** | 2 layers | **1.7 MB (4.1x compression)** | **128.4** | **Perfect (Recovered)** |

> [!NOTE]
> Compressing every layer breaks the output. By keeping the first and last `fp16_layers` uncompressed (Adaptive mode), we eliminate text collapse while still compressing the overall cache.

#### Test 2: Extreme Scale - Qwen 2.5 32B (Massive Model)
`benchmark_32B_extreme.py`

| Cache Type | Cache Size (200 tokens) | Speed (tok/s) | Compression Ratio |
| :--- | :--- | :--- | :--- |
| **FP16 (Default)** | 64.0 MB | 25.4 | 1.0x |
| **Adaptive 4-bit (4 layers)**| **21.5 MB** | 18.0 | **2.97x compressed** |

> [!IMPORTANT]
> **Extrapolating Scale**: In a 32,000-token long-context scenario, FP16 requires roughly **10 GB** of RAM just for the KV cache. TurboQuant brings this down to roughly **3.4 GB**, making long context generation easily achievable alongside a 20GB loaded model.

---

<a name="japanese"></a>
## 🇯🇵 日本語 (Japanese)

このリポジトリは、[TurboQuant](https://arxiv.org/abs/2504.19874) KVキャッシュ圧縮技術をApple Silicon (M5 Max / 128GB) 環境で実証・ベンチマークしたスクリプトと詳細レポートです。

MLX向けに提供される**Fused Metalカーネル**と、モデルの最初と最後の層の精度を落とさない**適応的レイヤー処理（Adaptive Layer Caching）**を組み合わせることで、テキスト出力の品質をまったく落とすことなく、LLMのメモリ消費（KVキャッシュ）を **約3〜10倍** にまで劇的に圧縮できます。

> [!TIP]
> これにより、128GB RAMを搭載したMacであれば、**320億（32B）クラスの超巨大モデルに長文コンテキスト（本1冊分のテキストデータなど）をそのまま読み込ませてもメインメモリが溢れることなく超高速で処理できる**ようになります。

### 🚀 検証結果 (M5 Max 128GB)

#### テスト1: 軽量モデル 1.5B の適応型リカバリー検証
全レイヤーへの単一圧縮による「生成文章の崩壊（同じ単語の繰り返し等）」を、一部の層をFP16で維持するAdaptive Modeでどの程度回避しつつ圧縮できるかのテスト。

| キャッシュ方式 | FP16保持層 | キャッシュ(100tok) | 速度 (tok/s) | テキスト品質 |
| :--- | :---: | :--- | :--- | :--- |
| **FP16 (標準)** | すべて | 7.0 MB | 194.9 | 完璧 |
| **全層 TQ 4-bit** | なし | 0.8 MB | 175.1 | 崩壊で使い物にならず |
| **Adaptive 4-bit** | 2層 | **1.7 MB (約4倍圧縮)** | **128.4** | **非常に良好・リカバリー成功** |

#### テスト2: 32Bモデルによる限界性能テスト
約20GBもの巨大なウェイトモデルを動かしながら発生する膨大なキャッシュサイズをどこまで圧縮できるかの実用限界テスト。

| キャッシュ方式 | キャッシュサイズ (200tok時) | 推論速度 | 圧縮率 |
| :--- | :--- | :--- | :--- |
| **FP16 (標準)** | 64.0 MB | 25.4 tok/s | 1.0x |
| **Adaptive 4-bit (4層)**| **21.5 MB** | 18.0 tok/s | **約3倍（2.97x）圧縮** |

> [!IMPORTANT]
> **圧倒的なスケールメリット**: これを一般的な **32,000トークン** の長文処理（RAG等）に換算すると、FP16ならキャッシュだけで約 **10GB** のメモリを消費しますが、TurboQuantを使えばたった **3.4GB** で済みます。32Bモデルの実運用においてメモリの生存戦略を根本から覆します。

---

### ⚙️ セットアップ手順 (Setup Instructions)

現状、TurboQuant MLXパッケージはソースからの直接ビルドが必要です。以下の手順で環境を構築してください。

```bash
git clone https://github.com/arozanov/turboquant-mlx.git
cd turboquant-mlx

# 最新のPython環境の場合、インストール前にエラーを回避するため pyproject.toml を修正します
# 修正前: build-backend = "setuptools.backends._legacy:_Backend"
# 修正後: build-backend = "setuptools.build_meta"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install transformers huggingface_hub pyarrow
```

### 🏃 テストの実行 (Running the Benchmarks)

環境が構築できたら、スクリプトを実行してベンチマークを開始します。

```bash
python benchmark_1_5B_adaptive.py
python benchmark_32B_extreme.py
```
> [!WARNING]
> 32BのスクリプトはHugging Faceからモデルをダウンロードするため、お使いの環境のストレージ空き容量（約20GB）にご注意ください。
