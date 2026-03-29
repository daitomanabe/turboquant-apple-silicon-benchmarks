#!/usr/bin/env python3
"""
TurboQuant Adaptive Layer Benchmark (1.5B Model)
Tests quality recovery by keeping first/last layers in FP16.
"""
import time
import mlx.core as mx
from mlx_lm import load

from turboquant_mlx.adaptive import make_adaptive_cache
from turboquant_mlx.patch import apply_patch

def run_bench(model, tokenizer, prompt, cache_list, max_tokens=100):
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_ms = (time.perf_counter() - t0) * 1000
    
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
        "prefill_ms": prefill_ms,
        "tok_s": len(tokens) / decode_time if decode_time > 0 else 0,
        "cache_mb": cache_mb,
    }

def main():
    apply_patch()
    
    model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    num_layers = len(model.layers)
    
    prompt = "Explain the concept of KV cache compression for large language models in three sentences."
    configs = [
        ("Adaptive 3-bit (4 fp16 layers)", 3, 4),
        ("Adaptive 4-bit (2 fp16 layers)", 4, 2)
    ]
    
    for name, bits, fp16 in configs:
        print(f"\n[{name}]")
        cache = make_adaptive_cache(num_layers, bits=bits, fp16_layers=fp16)
        r = run_bench(model, tokenizer, prompt, cache)
        print(f"  Speed: {r['tok_s']:.1f} tok/s | Cache: {r['cache_mb']:.1f} MB")
        print(f"  Output: {r['text'][:150]}...\n")

if __name__ == "__main__":
    main()
