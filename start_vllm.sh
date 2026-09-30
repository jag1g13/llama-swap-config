#!/bin/bash
set -uo pipefail

REPO=/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4
REV="$(cat "$REPO/refs/main" 2>/dev/null || true)"
SNAP="${REV:+$REPO/snapshots/$REV}"
[ -d "$SNAP" ] || SNAP="$(ls -dt "$REPO"/snapshots/*/ | grep -v -- -fp8hybrid | head -1)"
# prefer the hybrid layout (fp8 side layers, ~7 GiB lighter + 20% faster) from scripts/prepare-hybrid.sh
HYBRID="${SNAP%/}-fp8hybrid"
[ -f "$HYBRID/.prepared" ] && SNAP="$HYBRID"

if [ -z "$SNAP" ] || [ ! -d "$SNAP" ]; then
  echo "checkpoint not found under $REPO - run scripts/download-weights.sh first"
  exit 1
fi

exec vllm serve "$SNAP" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8000 --load-format safetensors \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update","vllm::ple_mmap_lookup_ids"]}' \
  --no-enable-flashinfer-autotune \
  --kv-cache-dtype auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
