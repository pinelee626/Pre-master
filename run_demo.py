"""
Demo script: Inject HopfieldCompressedAttention into a Llama model and run text generation.

Usage:
    python run_demo.py                           # Use a small Llama model from HF
    python run_demo.py --model meta-llama/Llama-3.2-1B
    python run_demo.py --prompt "The future of AI is"
    python run_demo.py --beta 2.0 --chunk_size 4 --top_k_ratio 0.3
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hopfield_compressed_attention import HopfieldCompressedAttention


def inject_hopfield_attention(model, config):
    """
    Replace every LlamaAttention layer in the model with HopfieldCompressedAttention,
    copying the original weights so the model behaves identically except for KV compression.
    """
    replaced = 0
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        # Create new HopfieldCompressedAttention with same config
        new_attn = HopfieldCompressedAttention(model.config, layer_idx=layer_idx)

        # Copy all parameters from the original attention module
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)

        # Move to same device / dtype
        new_attn = new_attn.to(
            device=next(old_attn.parameters()).device,
            dtype=next(old_attn.parameters()).dtype,
        )

        # Inject
        layer.self_attn = new_attn
        replaced += 1

    print(f"[✓] Replaced {replaced} attention layers with HopfieldCompressedAttention")
    print(f"    β={model.config.hopfield_beta}, chunk_size={model.config.chunk_size}, "
          f"top_k_ratio={model.config.top_k_ratio}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Hopfield-Compressed Attention Demo")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str,
                        default="The key innovation in modern transformer architectures is",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Number of tokens to generate")

    # Hopfield compression hyperparameters
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Hopfield inverse temperature")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Tokens per chunk for compression")
    parser.add_argument("--top_k_ratio", type=float, default=0.5,
                        help="Fraction of chunks to keep uncompressed")
    parser.add_argument("--compress_threshold", type=int, default=32,
                        help="Min sequence length to trigger compression")
    parser.add_argument("--rank_iterations", type=int, default=20,
                        help="Power-iteration steps for TokenRank")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if omitted)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Loading model: {args.model} ...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        attn_implementation="eager",  # We need explicit attention weights
    )
    model.eval()

    # Set compression hyperparameters on config (so layers can read them)
    model.config.hopfield_beta = args.beta
    model.config.chunk_size = args.chunk_size
    model.config.top_k_ratio = args.top_k_ratio
    model.config.compress_threshold = args.compress_threshold
    model.config.rank_iterations = args.rank_iterations

    # --- Baseline generation (standard attention) ---
    print("\n" + "=" * 60)
    print("BASELINE (standard LlamaAttention)")
    print("=" * 60)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        baseline_cache = DynamicCache()
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=baseline_cache,
        )
    baseline_time = time.time() - t0
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)

    # Measure baseline KV cache size
    baseline_kv_size = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for k, v in zip(baseline_cache.key_cache, baseline_cache.value_cache)
        if k.numel() > 0
    )

    print(f"Prompt: {args.prompt}")
    print(f"Output: {baseline_text}")
    print(f"Time:   {baseline_time:.2f}s")
    print(f"KV cache size: {baseline_kv_size / 1024:.1f} KB")

    # --- Inject Hopfield-compressed attention ---
    model = inject_hopfield_attention(model, model.config)

    # --- Compressed generation ---
    print("\n" + "=" * 60)
    print("HOPFIELD-COMPRESSED ATTENTION")
    print("=" * 60)

    t0 = time.time()
    with torch.no_grad():
        compressed_cache = DynamicCache()
        compressed_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=compressed_cache,
        )
    compressed_time = time.time() - t0
    compressed_text = tokenizer.decode(compressed_out[0], skip_special_tokens=True)

    # Measure compressed KV cache size
    compressed_kv_size = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for k, v in zip(compressed_cache.key_cache, compressed_cache.value_cache)
        if k.numel() > 0
    )

    print(f"Prompt: {args.prompt}")
    print(f"Output: {compressed_text}")
    print(f"Time:   {compressed_time:.2f}s")
    print(f"KV cache size: {compressed_kv_size / 1024:.1f} KB")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    if baseline_kv_size > 0:
        ratio = compressed_kv_size / baseline_kv_size
        print(f"KV cache compression ratio: {ratio:.2%} of baseline")
        print(f"Memory saved: {(1 - ratio) * 100:.1f}%")
    print(f"Speed: baseline {baseline_time:.2f}s vs compressed {compressed_time:.2f}s")


if __name__ == "__main__":
    main()
