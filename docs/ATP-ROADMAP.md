# ATP Runtime — mistral.rs scaffold

Fork of [`EricLBuehler/mistral.rs`](https://github.com/EricLBuehler/mistral.rs) as the
Rust substrate for the **Activation-Trajectory Persistence (ATP)** runtime
(paper draft: `inference-cluster/research/activation-trajectory-persistence.md`,
paper repo: `corzogac/remora`).

## Why mistral.rs

- Pure Rust (no C/C++ toolchain), Apache-2.0, built on HF `candle`.
- MoE + GGUF/UQFF/ISQ quantization, paged attention, Metal + CUDA,
  multi-GPU/distributed, OpenAI-compatible server.
- v0.8.2 CUDA benchmarks: ~1.8–2.3× llama.cpp prefill TPS (Gemma-4-E4B:
  27,706 vs 11,992 TPS on B200), decode on par — a Rust engine is not a
  compromise.
- Rust ownership model suits the ATP runtime's explicit memory pools, VRAM
  residency control and GPU-switching logic.

## ATP instrumentation hooks (port from llama-cpp-remora)

1. **Router / expert traces** — REMORA_TRACE_FILE equivalent: per-layer
   `R_l` (expert IDs + logits) dumps to JSONL/NdJSON during decode, exposed via
   the server API (OpenTelemetry-style span hooks), not just a file.
2. **Content-channel persistence** — cross-turn cosine of the router-blind
   hidden stream `h_blind` at matched positions (the §10.8 predictor), and the
   `ΔC → ΔR/ΔA/ΔH` sensitivity curve by layer (E2 perturbation fan).
3. **Trajectory-driven expert prefetch** — predict next experts from the
   content channel (not raw router logits) and overlap expert weight loading
   with compute — the ATP "PREPARE NOW → continue wave" runtime.
4. **VRAM residency / GPU switching** — explicit layer/expert pinning and
   residency policy hooks (which GPU owns which weights, when to migrate).

## Redundancy vs llama.cpp (adopting this fork)

| Component | Status with mistral.rs |
|---|---|
| `corzogac/llama-cpp-remora` fork + REMORA_TRACE_FILE | **Redundant** — traces reimplemented in Rust (hooks 1–2) |
| `~/remora/bin` llama-server build | **Redundant** (also currently broken: missing libllama-server-impl.dylib) |
| Ollama (packaged serving, MLX on AS) | **Keep** — user-facing packaged layer, cloud-model stubs |
| `corzogac/colibri` (disk-streaming frontier MoE, M3/MSA) | **Keep** — research vehicle; mistral.rs does not stream ~20k experts from disk at Colibri scale |

## Upstream contribution candidates (EricLBuehler/mistral.rs)

- Expert prefetching from hidden-state/content-channel prediction (ATP §10.8).
- MSA sparse attention for MiniMax-M3 GGUF (llama.cpp has dense-fallback only;
  our Colibri port is bit-exact — knowledge transfers).
- Router-logit trace hooks exposed via the server API.
- VRAM residency / pinning API.

## Watch

- Release watcher cron: `check_mistral_release.py` (weekly, alerts on new tag).

## Quickstart

```bash
cd ~/inference-cluster/mistral-rs
cargo build --release --features metal   # or cuda / cuda-pagedattn
./target/release/mistralrs-server --port 8080 gguf -f /path/to/model.gguf
```
