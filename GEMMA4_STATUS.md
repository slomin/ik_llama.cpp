# Gemma 4 Support - Work in Progress

> **TEMPORARY FILE** - Remove once Gemma 4 support is complete.

## Branch: `slomin/gemma4_support`

## What Works

- **Architecture registration**: `LLM_ARCH_GEMMA3N` and `LLM_ARCH_GEMMA4` recognized
- **Tensor loading**: All 601 tensors load from E2B GGUF
- **Hyperparameters**: Per-layer SWA flags, dual head dims (256/512), shared KV count
- **KV cache**: Per-layer sizing works (SWA=256, global=512) — 168 MiB vs 8960 MiB before fix
- **Graph building**: Attention, FFN, MoE path, norms, V normalization, proportional RoPE
- **Chat template**: Works via `--jinja --cnv --chat-template-kwargs '{"enable_thinking": false}'`
- **Inference runs**: ~16 t/s gen, ~52 t/s prompt on M1 Pro
- **Tokenizer**: SPM-style BPE with raw UTF-8, `[^\n]+|[\n]+` regex, space escaping — matches upstream
- **Shared KV cache**: Layers >= n_layer_kv_from_start read from source layer's cache (no redundant K/V)
- **Per-layer embeddings**: Enabled (projection + norm + injection at each layer)
- **Output scaling**: Uses ggml_mul broadcast for per-layer scalar scaling
- **E2B produces correct output**: "What is 2+2?" → "4", "Capital of France?" → "Paris"

## Remaining Work

### 1. Gemma3n Architecture (not yet runnable)

GEMMA3N is registered but the decoder graph is not implemented. Gemma3n requires
altup/laurel support which is distinct from the Gemma4 PLE path. Loading a gemma3n
GGUF will error clearly. No gemma3n GGUFs exist in the wild yet (E2B/E4B use
gemma4 arch despite the name).

### 2. MoE Per-Expert Scales (Unverified Quality)

Per-expert scale tensors are loaded and applied at the correct point (before
gate/up split + activation for up_exps_s, after down matmul for down_exps_s).
All four models load and produce correct output. Quality comparison against
upstream with quantized MoE has not been done yet.

### 5. Vision / Multimodal

Gemma 4 supports image input via MobileNetV5 projector. Files to modify:
- `examples/mtmd/clip-impl.h` — add projector types
- `examples/mtmd/clip.cpp` — tensor loading and forward pass
- `examples/mtmd/mtmd.cpp` — handle new projectors

### 6. Known Upstream Issues

- Flash attention may segfault on context >16K (#21336) — use `--flash-attn off`
- `<unused24>` token generation bug (#21321)
- Audio support not implemented (#21325)

## Files Changed

| File | What |
|------|------|
| `src/llama-arch.h/cpp` | New enums, KV keys, tensor types for gemma3n/gemma4 |
| `src/llama-hparams.h/cpp` | New fields (SWA head dims, altup, PLE), loading cases |
| `src/llama-model.h/cpp` | Layer fields (altup, laurel, PLE, norms), tensor name maps |
| `src/llama-load-tensors.cpp` | `create_gemma3n_tensors()`, `create_gemma4_tensors()` |
| `src/llama-build-context.h/cpp` | `build_gemma4()` graph builder, shared KV via il_kv param |
| `src/llama-vocab.h/cpp` | `gemma4` tokenizer, GEMMA4 pre-type, SPM-style BPE |
| `src/unicode.h/cpp` | `byte_encode` parameter for raw UTF-8 BPE |
| `src/llama.cpp` | RoPE type for gemma3n/gemma4, per-layer KV cache alloc |
| `tests/test-gemma4-arch.cpp` | Architecture registration tests |

## Quick Test Commands

```bash
# Build
cmake --build build --target llama-cli -j$(sysctl -n hw.ncpu)

# Test inference (produces correct output)
echo "What is 2+2?" | ./build/bin/llama-cli -m /Users/jan/0Projects/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
  -cnv --jinja --chat-template-kwargs '{"enable_thinking": false}' -n 64 -c 4096 --no-mmap 2>/dev/null

# Test tokenizer (should produce 9259)
./build/bin/llama-tokenize -m /Users/jan/0Projects/gemma/gemma-4-E2B-it-Q4_K_M.gguf -p "Hello"

# Run arch tests
./build/bin/test-gemma4-arch
```

## Upstream References

- **Main PR**: ggml-org/llama.cpp#21309 (merged 2026-04-02, commit `63f8fe0`)
- **Tokenizer fix**: ggml-org/llama.cpp#21343 + comment patch for `[\n]+`
- **Template fix**: ggml-org/llama.cpp#21326
- **Upstream clone**: `/Users/jan/0Projects/gemma/llama.cpp` (checked out at merge commit)

## Test Model

E2B Q4_K_M at `/Users/jan/0Projects/gemma/gemma-4-E2B-it-Q4_K_M.gguf` (2.9GB)
