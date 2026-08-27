# Windows CUDA Build Fixes & Qwen3.8-27B Validation (2026-08-27)

Fork: `corzogac/mistral.rs` branch `atp-scaffold`, v0.9.2.
Machine: Janus (UN-IHE Zbook), Windows 11, i9-11900H, 34 GB RAM,
NVIDIA RTX A2000 Laptop 4 GB (driver 595.95), CUDA 12.4 toolkit,
Visual Studio Build Tools 2022 (MSVC), cargo 1.x.

## Why this document

The `atp-scaffold` fork previously built and validated only on macOS/Metal
(see `ATP-VALIDATION.md`, ornith-9b, M4). This run proves the same fork
builds and serves on **Windows + CUDA**, with three small build-system
fixes, and serves the **Qwen3.8-27B** GGUF (dense 27B, Gated DeltaNet
linear attention, qwen35 arch).

## The three fixes (all in this branch)

### 1. `mistralrs-quant/build.rs` — skip Marlin kernels on MSVC

`nvcc` on MSVC fails to compile `kernels/marlin/marlin_matmul_awq_bf16.cu`
(empty nvcc error; `-fPIC` warnings are harmless noise). Marlin is a
fast-path GEMM for GPTQ/AWQ quantized models only; GGUF inference does not
use it. The Rust side already guards all Marlin calls behind
`#[cfg(has_marlin_kernels)]` (`mistralrs-quant/src/lib.rs:122,1578,1801...`),
so emitting the cfg only when the kernels actually compiled is sufficient:

```rust
let has_marlin = cc_over_80 && !target.contains("msvc");
if has_marlin { println!("cargo:rustc-cfg=has_marlin_kernels"); }
...
if cc_over_80 && !has_marlin { excluded_files.push("marlin_*.cu"); }
```

Model on the device is sm_86 (`cc_over_80 == true`), so this only affects
the MSVC toolchain branch. On Linux/Windows-GNU Marlin still compiles.

### 2. `mistralrs-core/src/paged_attention/mod.rs` — widen `plan` module gate

The `plan` module was gated `#[cfg(any(all(feature = "cuda", target_family =
"unix"), feature = "metal"))]`, but the module body already contains full
Windows fallbacks (`#[cfg(not(all(feature = "cuda", target_family =
"unix")))]` at lines 583, 673, 694 — `fa3_pool_bytes = 0`, `GatherSdpa`
fallback, etc.). The gate was narrower than the code. Widened to:

```rust
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) mod plan;
```

### 3. `mistralrs-core/src/pipeline/cuda_graph.rs` — gate FA3 schedule prep

`prepare_fa3_decode_schedules` (and its single call site inside
`capture_cuda_decode_graph`) referenced `FlashInferMetadata::
for_each_fa3_decode_schedule`, `mistralrs_paged_attn::fa3_prepare_decode_metadata`
and `Fa3DecodeMetadata`, which only exist on
`all(feature = "cuda", target_family = "unix")` in the flash-attn crate.
Both the function definition and the call are now gated:

```rust
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn prepare_fa3_decode_schedules(...) -> candle_core::Result<()> { ... }
```

On Windows CUDA, decode-graph capture simply skips the FA3 schedule
preparation (FlashInfer/FA3 backend is unix-only; the decode graphs
themselves still work via the standard path).

### 4. `.cargo/config.toml` — static CRT on Windows MSVC

nvcc-compiled kernel archives (`mistralrsquant.lib` from
`mistralrs-quant/build.rs`, and `libmoe.a` from the candle-kernels git
dependency) default to **static CRT (/MT)**. Rust crates (notably mimalloc)
default to **dynamic (/MD)**, producing `LNK2038: RuntimeLibrary mismatch`
(47 errors) at the final link. Fix: force the whole Rust build to static
CRT on Windows:

```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+crt-static"]
```

This matches nvcc's /MT default everywhere and needs no changes to
candle-kernels (a git dependency we don't control).

## Build procedure (Windows)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd C:\Users\gco\mistral-rs
cargo build --release --features cuda --bin mistralrs
```

Prerequisite: MSVC host toolchain in PATH for nvcc (`vcvars64.bat`), and
Python 3.x with `python3.lib` (pyo3) — the `mistralrs` CLI bin links the
pyo3 layer. Install via `winget install Python.Python.3.12`.

## Validation: Qwen3.8-27B on Janus (LM Studio baseline)

- Model: `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_M.gguf`
  (17.39 GB, Apache-2.0, unsloth Dynamic V3.0 quant).
- Served via LM Studio (`:1234`, model id `qwen3.8-27b`) as baseline.
- Cold load 31.5 s, resident 16.2 GiB.
- One-shot: "Reply with exactly: LOCAL_OK and the number 42"
  → `LOCAL_OK 42` ✓
- 233 completion tokens in 123 s ≈ **1.9 tok/s**, of which **225 were
  reasoning tokens** (xhigh reasoning effort is the model's default).
- Conclusion: dense 27B on this box is memory-bandwidth-bound
  (~45 GB/s effective ÷ 17.5 GB weights ≈ 1.5–3 tok/s, matching
  physics estimate). Suitable as an overnight/batch lane, not a daily
  interactive agent. Set reasoning effort low/medium for agent loops.

## GGUF quant compatibility finding

**Unsloth Dynamic V3.0 quants (UD-Q4_K_M etc.) fail to load in mistral.rs
v0.9.2** on any platform: the Dynamic repack uses `IQ4_XS` tensors
internally, and the native GGUF binder supports only F32/F16/BF16,
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and Q2_K–Q8_K (MXFP4 only via the explicit
GPT-OSS binding). Error:

```
GGUF tensor `blk.0.ffn_down.weight` uses dtype IQ4_XS (23) for native binding
```

Use the **plain Q4_K_M from `lmstudio-community/Qwen3.8-27B-GGUF`**
(16.8 GB) with mistral.rs instead.

## Validation: same GGUF via mistral.rs on Windows CUDA (`--port 8080`)

- Model: `lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`
  (15.656 GiB, plain Q4_K_M — required, see quant finding above).
- Command:
  `mistralrs.exe serve -m C:\Users\gco\models\lmstudio-community\Qwen3.8-27B-GGUF
   -f Qwen3.8-27B-Q4_K_M.gguf --port 8080 --max-model-len 8192 --cpu`
- **`LOCAL_OK 42` ✓** (with `enable_thinking=false`): 21 prompt tok in
  7.86 s (2.67 tok/s prefill), 6 completion tok in 15.39 s
  (**0.39 tok/s decode**), 23.2 s total, avg 1.16 tok/s.
- Same model, LM Studio baseline: ~1.9 tok/s (incl. 225 reasoning tokens
  of 233). **LM Studio remains faster on this box for the 27B** — it
  offloads what fits onto the A2000 and uses its own CPU threading.

### Finding: auto device map fails when only ~2 layers fit on the A2000

With default auto mapping (no `--cpu`), mistral.rs placed layers 0-1 on
`cuda[0]` and 2-63 on CPU, then the prompt step died:

```
ERROR mistralrs_core::engine: prompt step - Model failed with error:
device mismatch in matmul, lhs: Cpu, rhs: Cuda { gpu_id: 0 }
```

i.e. a cross-device matmul between the CPU-resident embedding output and
the CUDA-resident first layer. Only 2/65 layers fit in 4 GB, so the
partial-offload gain would be ~3% anyway; `--cpu` is the correct lane on
this box. (Upstream has no CUDA-on-Windows CI — we are the pioneer here.)

### Finding: the Intel UHD "16 GB" is shared system RAM, not VRAM

`dxdiag` on Janus reports:
- Intel(R) UHD Graphics: **Dedicated Memory 128 MB**, Shared 16,215 MB
- NVIDIA RTX A2000 Laptop GPU: Dedicated 3,965 MB, Shared 16,215 MB

The iGPU has no real VRAM; both GPUs draw from the same 32 GB system
pool, and no inference backend in this stack (mistral.rs, llama.cpp,
candle) targets Intel iGPUs. The A2000's 3,965 MB is the only usable
accelerator; for a 27B it holds ~2/65 layers → CPU-bound regardless.

### Operational notes (Windows)

- The auto device mapper reads **free RAM** at startup. LM Studio holding
  a resident model (16+ GiB) starves mistral.rs (`cpu (avail: 1630MB)` →
  "does not fit"). Run `lms unload --all` before starting mistral.rs —
  the launcher must enforce this handoff.
- A killed SSH session does **not** kill the remote `mistralrs.exe`
  (13+ GiB zombie observed). Clean up with
  `taskkill /F /IM mistralrs.exe` before restarting.
- CPU-only load takes ~2.5 min to reach the listener (15.97 GiB
  resident); the f16 CPU KV cache is selected by default
  (`MISTRALRS_CPU_KV_F32=1` for f32).

## TODO after this doc

- [x] Serve the same GGUF via `mistralrs serve` (`--port 8080`), compare
      tok/s vs LM Studio → **done: 0.39 tok/s decode CPU-only vs 1.9
      LM Studio**; keep LM Studio for the 27B lane on Janus, use
      mistral.rs where it shines (smaller models that fit VRAM, e.g.
      ornith-9b, and ISQ/paged-attention experiments).
- [ ] Re-run `bench auto` with ISQ + paged attention on a model that fits
      the A2000 (e.g. ornith-1.0-9b) to measure the Rust engine's
      memory/token efficiency on Windows CUDA vs LM Studio.
- [ ] Launcher script for Janus: unload LM Studio (`lms unload --all`) →
      start mistral.rs (or vice versa); never both resident.
