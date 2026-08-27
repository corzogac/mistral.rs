# mistral.rs — ATP scaffold validation (2026-08-27)

Build + smoke test of the Rust substrate for the ATP runtime
(see `ATP-ROADMAP.md`).

## Build

- Repo: `~/inference-cluster/mistral-rs` (fork `corzogac/mistral.rs`, upstream
  `EricLBuehler/mistral.rs`)
- Version: **v0.9.2**, `cargo build --release --features metal` → **6m22s**, exit 0
- Host: Mac mini (Apple M4, 32 GB unified), macOS

## Model

- `ornith:9b` (ollama tag) — `ornith-ai/Ornith-1.0-9B`, Q4_K_M GGUF, 5.6 GB
- Architecture (HF config): Qwen3.5 (`qwen3_5`), 32 layers mixing
  linear-attention (GLA-style) and full attention every 4th layer, MTP head,
  vocab 248320, 256K context
- Weights loaded **directly from the Ollama store on GC_SDD1**
  (`/Volumes/GC_SDD1/Models/ollama/blobs/sha256-af6336…`) via a local model dir
  (`~/inference-cluster/models/ornith-9b` = HF config/tokenizer + GGUF symlink —
  no duplicate weights)

## Bench (3 iterations, Metal)

| Metric | Value |
|---|---|
| TTFT (512 tok prompt) | 125.0 ± 1.3 T/s · 4098 ms |
| Decode (128 tok) | **15.6 ± 0.0 T/s** · 64.2 ms TPOT |

## One-shot generation

Prompt: "In one sentence, what is activation-trajectory persistence in
Mixture-of-Experts inference?"

- TTFT: 0.22 s · prefill 127.9 T/s · decode **16.4 T/s** (1434 tokens)
- Output coherent and on-topic (router-stability interpretation of ATP — the
  naive reading; the content-channel refinement is the paper's contribution)

## Comparison vs Ollama/llama.cpp (same model, same Mac)

- Ollama ornith:9b ≈ 17.7 T/s (fleet measurement 2026-08-15)
- mistral.rs ≈ 15.6–16.4 T/s → within ~7–12%, Metal backend fully functional

## Verdict

- ✅ Metal build loads and serves real GGUFs from the SSD
- ✅ Exotic Qwen3.5 (linear-attention) arch auto-detected and runs
- ✅ Base for ATP hooks 1–4 (router traces, content-channel persistence,
  trajectory-driven prefetch, VRAM residency) — see `ATP-ROADMAP.md`

## CLI gotchas (documented for reuse)

- `-m` is mandatory and must resolve (HF id or **local dir**); with an HF id,
  `-f` is interpreted **repo-relative** — use a local model dir for local GGUFs:
  `mistralrs run auto -m <dir> -f model.gguf --format gguf`
- HF downloads need `curl -L` (redirects to CDN)
