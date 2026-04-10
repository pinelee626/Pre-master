"""
Hopfield-Compressed Attention: KV Cache Compression via TokenRank & Modern Hopfield Networks

This module implements a novel KV cache compression algorithm that:
1. Treats the attention matrix as a Discrete-Time Markov Chain (DTMC) transition matrix
2. Computes TokenRank (steady-state distribution) to identify metastable clusters
3. Uses Modern Hopfield Network energy-based update rules to fuse clusters into prototype tensors
4. Stores only compressed prototypes in the KV cache

Reference equations:
- TokenRank: π = π P  (steady-state of DTMC)
- Hopfield prototype: ξ_new = X^T softmax(β X ξ)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
)


def compute_token_rank(
    attn_weights: torch.Tensor,
    num_iterations: int = 20,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute TokenRank: the steady-state distribution of the attention matrix
    treated as a DTMC transition probability matrix.

    Args:
        attn_weights: (batch, num_heads, seq_len, seq_len) — post-softmax attention.
                      Each row already sums to 1, so it IS a valid transition matrix.
        num_iterations: power-iteration steps to approximate the stationary distribution.
        epsilon: small constant for numerical stability.

    Returns:
        token_rank: (batch, num_heads, seq_len) — importance score per token per head.
    """
    # P = attn_weights  — shape (B, H, S, S), rows sum to 1
    # We want the LEFT eigenvector π such that π P = π, i.e. P^T π = π.
    # Power iteration on P^T:
    P_T = attn_weights.transpose(-2, -1)  # (B, H, S, S)
    S = attn_weights.shape[-1]

    # Uniform initialisation
    pi = torch.ones(*attn_weights.shape[:-1], device=attn_weights.device) / S  # (B, H, S)

    for _ in range(num_iterations):
        # pi_new = P^T @ pi  (batch matmul)
        pi = torch.einsum("bhij,bhj->bhi", P_T, pi)
        # Re-normalise to stay on the simplex
        pi = pi / (pi.sum(dim=-1, keepdim=True) + epsilon)

    return pi  # (B, H, S)


def identify_chunks(
    token_rank: torch.Tensor,
    chunk_size: int = 8,
    top_k_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Partition the sequence into fixed-size chunks and decide which chunks to compress
    based on aggregate TokenRank mass.

    Chunks whose total TokenRank is in the BOTTOM (1 - top_k_ratio) are compressed
    (they are 'metastable' — tokens attend to them moderately but they are not critical sinks).
    Top-ranked chunks are kept verbatim.

    Args:
        token_rank: (B, H, S)
        chunk_size: number of tokens per chunk.
        top_k_ratio: fraction of chunks to KEEP uncompressed.

    Returns:
        compress_mask: (B, H, num_chunks) bool — True for chunks to compress.
        num_chunks: int
    """
    B, H, S = token_rank.shape
    # Pad to make S divisible by chunk_size
    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad > 0:
        token_rank = F.pad(token_rank, (0, pad), value=0.0)

    num_chunks = token_rank.shape[-1] // chunk_size
    # (B, H, num_chunks) — sum of TokenRank per chunk
    chunk_scores = token_rank.view(B, H, num_chunks, chunk_size).sum(dim=-1)

    # Determine threshold: keep the top-k chunks, compress the rest
    k = max(1, int(num_chunks * top_k_ratio))
    # Top-k scores → threshold is the k-th largest value
    topk_vals, _ = chunk_scores.topk(k, dim=-1)
    threshold = topk_vals[..., -1:]  # (B, H, 1)

    compress_mask = chunk_scores < threshold  # True = compress this chunk

    return compress_mask, num_chunks


def hopfield_prototype(
    X: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Modern Hopfield Network energy-based update to fuse a set of stored patterns
    into a single prototype (fixed-point attractor).

    Update rule:  ξ_new = X^T softmax(β X ξ)
    We initialise ξ as the mean of the patterns and run one update step
    (sufficient for a soft-max retrieval in the large-β regime).

    Args:
        X: (N, D) — N stored patterns of dimension D (e.g. key/value vectors in a chunk).
        beta: inverse temperature. Higher β → sharper selection (approaches argmax).

    Returns:
        prototype: (D,) — single fused prototype vector.
    """
    # Initialise query as mean pattern
    xi = X.mean(dim=0)  # (D,)

    # One Hopfield update step: ξ = X^T softmax(β X ξ)
    logits = beta * (X @ xi)        # (N,)
    weights = F.softmax(logits, dim=0)  # (N,)
    prototype = X.t() @ weights     # (D,)

    return prototype


def compress_kv_with_hopfield(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    compress_mask: torch.Tensor,
    chunk_size: int,
    beta: float,
    original_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Hopfield prototype merging to compress selected chunks in the KV cache.

    Args:
        key_states:   (B, H, S, D)
        value_states: (B, H, S, D)
        compress_mask: (B, H, num_chunks) — True for chunks to compress.
        chunk_size: tokens per chunk.
        beta: Hopfield inverse temperature.
        original_seq_len: original (unpadded) sequence length.

    Returns:
        compressed_keys:   (B, H, S', D) — S' <= S
        compressed_values: (B, H, S', D)
    """
    B, H, S, D = key_states.shape

    # Pad KV to match chunk alignment
    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad))
        value_states = F.pad(value_states, (0, 0, 0, pad))

    S_padded = key_states.shape[2]
    num_chunks = S_padded // chunk_size

    # Reshape into chunks: (B, H, num_chunks, chunk_size, D)
    k_chunks = key_states.view(B, H, num_chunks, chunk_size, D)
    v_chunks = value_states.view(B, H, num_chunks, chunk_size, D)

    new_keys = []
    new_values = []

    for b in range(B):
        head_keys = []
        head_values = []
        for h in range(H):
            tokens_k = []
            tokens_v = []
            for c in range(num_chunks):
                if compress_mask[b, h, c].item():
                    # Compress this chunk → single prototype
                    k_proto = hopfield_prototype(k_chunks[b, h, c], beta=beta)  # (D,)
                    v_proto = hopfield_prototype(v_chunks[b, h, c], beta=beta)  # (D,)
                    tokens_k.append(k_proto.unsqueeze(0))  # (1, D)
                    tokens_v.append(v_proto.unsqueeze(0))
                else:
                    # Keep verbatim
                    tokens_k.append(k_chunks[b, h, c])  # (chunk_size, D)
                    tokens_v.append(v_chunks[b, h, c])

            head_keys.append(torch.cat(tokens_k, dim=0))   # (S', D)
            head_values.append(torch.cat(tokens_v, dim=0))

        # All heads may have different S' — pad to max
        max_len = max(t.shape[0] for t in head_keys)
        for i in range(H):
            diff = max_len - head_keys[i].shape[0]
            if diff > 0:
                head_keys[i] = F.pad(head_keys[i], (0, 0, 0, diff))
                head_values[i] = F.pad(head_values[i], (0, 0, 0, diff))

        new_keys.append(torch.stack(head_keys, dim=0))     # (H, S', D)
        new_values.append(torch.stack(head_values, dim=0))

    compressed_k = torch.stack(new_keys, dim=0)   # (B, H, S', D)
    compressed_v = torch.stack(new_values, dim=0)

    return compressed_k, compressed_v


class HopfieldCompressedAttention(LlamaAttention):
    """
    Drop-in replacement for LlamaAttention that compresses the KV cache using
    TokenRank-based clustering and Modern Hopfield Network prototype merging.

    Hyperparameters (set via config or constructor):
        hopfield_beta:       Inverse temperature for Hopfield update (default: 1.0)
        chunk_size:          Tokens per chunk for compression (default: 8)
        top_k_ratio:         Fraction of chunks to keep uncompressed (default: 0.5)
        rank_iterations:     Power-iteration steps for TokenRank (default: 20)
        compress_threshold:  Minimum sequence length to trigger compression (default: 32)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # Hopfield / compression hyperparameters (can be overridden via config)
        self.hopfield_beta = getattr(config, "hopfield_beta", 1.0)
        self.chunk_size = getattr(config, "chunk_size", 8)
        self.top_k_ratio = getattr(config, "top_k_ratio", 0.5)
        self.rank_iterations = getattr(config, "rank_iterations", 20)
        self.compress_threshold = getattr(config, "compress_threshold", 32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Hopfield-compressed KV cache.

        Differences from standard LlamaAttention.forward:
        1. We compute attention weights EXPLICITLY (no SDPA / Flash shortcut)
           so that we can extract the attention matrix for TokenRank.
        2. After attention, we compress the KV states before storing in cache.
        """
        B, S, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # --- Projections ---
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # query/key/value: (B, H, S, D)

        # --- Rotary position embeddings ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- Prepend cached KV if available ---
        if past_key_values is not None:
            cached_k = past_key_values.key_cache
            cached_v = past_key_values.value_cache
            if self.layer_idx < len(cached_k) and cached_k[self.layer_idx].numel() > 0:
                key_states = torch.cat([cached_k[self.layer_idx], key_states], dim=2)
                value_states = torch.cat([cached_v[self.layer_idx], value_states], dim=2)

        total_seq_len = key_states.shape[2]

        # --- Explicit attention computation ---
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        # (B, H, S_q, S_kv)

        # Causal mask
        if attention_mask is not None:
            # Expand or slice mask to match current shapes
            causal_len = attn_weights.shape[-1]
            if attention_mask.dim() == 2:
                # (B, S_kv) → (B, 1, 1, S_kv)
                mask = attention_mask[:, None, None, :causal_len]
                attn_weights = attn_weights + (1.0 - mask) * torch.finfo(attn_weights.dtype).min
            elif attention_mask.dim() == 4:
                mask = attention_mask[..., :attn_weights.shape[-2], :causal_len]
                attn_weights = attn_weights + mask
        else:
            # Build causal mask manually
            S_q = query_states.shape[2]
            S_kv = key_states.shape[2]
            causal = torch.triu(
                torch.full((S_q, S_kv), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
                diagonal=S_kv - S_q + 1,
            )
            attn_weights = attn_weights + causal

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.training:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)

        # --- Attention output ---
        attn_output = torch.matmul(attn_probs, value_states)  # (B, H, S_q, D)

        # --- TokenRank-based KV compression ---
        if total_seq_len >= self.compress_threshold:
            # Build a square attention matrix over all KV positions for TokenRank
            # Use the last S_q rows but we need a full (S_kv x S_kv) transition matrix.
            # Approximation: use the available attn_probs rows and average over query positions.
            # For a proper DTMC we need P[i,j] = "how much token i attends to token j".
            # During prefill (S_q == S_kv) this is exact; during decode (S_q == 1) we
            # accumulate a running estimate.  For the PoC we use a symmetric fallback:
            #   P_ij ∝ exp(k_i · k_j / sqrt(d))
            with torch.no_grad():
                kk = torch.matmul(key_states, key_states.transpose(-2, -1)) * self.scaling
                # Apply causal mask to the KV-KV similarity
                causal_kk = torch.triu(
                    torch.full((total_seq_len, total_seq_len), float("-inf"),
                               device=kk.device, dtype=kk.dtype),
                    diagonal=1,
                )
                kk = kk + causal_kk
                P = F.softmax(kk, dim=-1)  # (B, H, S_kv, S_kv) — valid transition matrix

                token_rank = compute_token_rank(P, num_iterations=self.rank_iterations)
                compress_mask, num_chunks = identify_chunks(
                    token_rank,
                    chunk_size=self.chunk_size,
                    top_k_ratio=self.top_k_ratio,
                )

            compressed_k, compressed_v = compress_kv_with_hopfield(
                key_states, value_states,
                compress_mask,
                chunk_size=self.chunk_size,
                beta=self.hopfield_beta,
                original_seq_len=total_seq_len,
            )
        else:
            compressed_k = key_states
            compressed_v = value_states

        # --- Update KV cache with compressed states ---
        if past_key_values is not None:
            # Directly write compressed KV into cache (bypass .update() to avoid double concat)
            while len(past_key_values.key_cache) <= self.layer_idx:
                past_key_values.key_cache.append(torch.tensor([]))
                past_key_values.value_cache.append(torch.tensor([]))
            past_key_values.key_cache[self.layer_idx] = compressed_k
            past_key_values.value_cache[self.layer_idx] = compressed_v

        # --- Output projection ---
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs
