#!/usr/bin/env python3
"""
TurboQuant Extreme Scale Benchmark (32B Model)
Tests massive memory reduction for local long-context generation.
Requirements: Mac with 64GB+ unified memory (e.g. M-series Max/Ultra).
"""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from turboquant_mlx.adaptive import make_adaptive_cache
from turboquant_mlx.patch import apply_patch

def run_bench(model, tokenizer, prompt, cache_list, max_tokens=200):
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    
    token = mx.argmax(logits[:, -1, :], axis=-1)
    tokens = [token.item()]
    
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache_list)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        tok_id = token.item()
        tokens.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
    decode_time = time.perf_counter() - t0
    
    cache_mb = sum([c.nbytes for c in cache_list]) / (1024 * 1024)
    return {
        "text": tokenizer.decode(tokens),
        "prefill_ms": prefill_time * 1000,
        "tok_s": len(tokens) / decode_time if decode_time > 0 else 0,
        "cache_mb": cache_mb,
    }

def main():
    apply_patch()
    
    model_name = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    print(f"Loading Massimo-scale dataset ({model_name})...")
    model, tokenizer = load(model_name)
    num_layers = len(model.layers)
    
    prompt = "Write a comprehensive 3-paragraph essay on the history of Macintosh computers and their architectural transitions (PowerPC to Intel to Apple Silicon) highlighting the performance leaps."
    
    configs = [
        ("FP16 (Default)", lambda: [KVCache() for _ in range(num_layers)]),
        ("Adaptive 4-bit (4 fp16 layers)", lambda: make_adaptive_cache(num_layers, bits=4, fp16_layers=4))
    ]
    
    for name, init_cache in configs:
        print(f"\n[{name}]")
        cache = init_cache()
        r = run_bench(model, tokenizer, prompt, cache)
        print(f"  Speed: {r['tok_s']:.1f} tok/s | Cache: {r['cache_mb']:.1f} MB")
        print(f"  Output: {r['text'][:150]}...\n")

if __name__ == "__main__":
    main()
