# TurboQuant on Apple Silicon (M5 Max Benchmarks)

This repository contains benchmark scripts and reproducibility reports testing the [TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression algorithm on Apple Silicon (M5 Max) using the [MLX](https://github.com/ml-explore/mlx) framework.

By utilizing TurboQuant's **Adaptive Layer caching** along with **Fused Metal Kernels**, we can compress the memory footprint of an LLM's KV cache by **~3x to 10x** with minimal degradation in generation speed and zero loss of quality. This allows even massive models (like 32 Billion parameter models) to run smoothly on high-memory Macs while processing massive context lengths.

## 🚀 Key Benchmarks

*Tested on: Apple MacBook Pro (M5 Max, 18-Core CPU, 128GB Unified Memory)*

### Test 1: Llama/Qwen 1.5B (Adaptive Compression)
`benchmark_1_5B_adaptive.py`

| Cache Type | FP16 Layers | Cache Size (100 tokens) | Speed | Text Quality |
| :--- | :---: | :--- | :--- | :--- |
| **FP16 (Default)** | All | 7.0 MB | 194.9 tok/s | Perfect |
| **All TQ 4-bit** | None | 0.8 MB | 175.1 tok/s | Collapsed (Repetition) |
| **Adaptive 4-bit** | 2 layers | **1.7 MB (4.1x compression)** | **128.4 tok/s (66% speed)** | **Perfect (Recovered)** |

**Takeaway:** Compressing every layer breaks the output. By keeping just the first and last `fp16_layers=2` uncompressed (Adaptive mode), we eliminate text collapse while still compressing the overall cache by over 4x.

### Test 2: Extreme Scale - Qwen 2.5 32B (Massive Model)
`benchmark_32B_extreme.py`

Testing a massive 32B parameter model (roughly 19GB in size) tracking cache per 200 tokens.
| Cache Type | Cache Size (200 tokens) | Speed | Compression Ratio |
| :--- | :--- | :--- | :--- |
| **FP16 (Default)** | 64.0 MB | 25.4 tok/s | 1.0x |
| **Adaptive 4-bit (4 layers)**| **21.5 MB** | 18.0 tok/s | **2.97x compressed** |

**Takeaway:** Extrapolating to a 32,000 token long-context scenario (e.g. RAG), FP16 would require roughly **10 GB** of RAM just for the KV cache. TurboQuant brings this down to roughly **3.4 GB**, making long context generation easily achievable alongside the loaded model.

## ⚙️ Setup and Installation

Currently, TurboQuant for MLX requires cloning the community repository and installing from source.

```bash
# 1. Clone the MLX TurboQuant kernels repository
git clone https://github.com/arozanov/turboquant-mlx.git
cd turboquant-mlx

# ⚠️ Quick Fix for Setuptools Error on newer Python versions
# Before piping install, open `turboquant-mlx/pyproject.toml`
# Change line 3 from:
# build-backend = "setuptools.backends._legacy:_Backend"
# To:
# build-backend = "setuptools.build_meta"

# 2. Install inside virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install transformers huggingface_hub pyarrow
```

## 🏃 Running the Benchmarks

Once installed, simply copy the benchmark scripts into your environment and run them:
```bash
python benchmark_1_5B_adaptive.py
python benchmark_32B_extreme.py
```
> Note: The 32B benchmark script requires downloading around 19GB of weights. Ensure you have the time and storage available before running.
