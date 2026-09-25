# Real LoRA Training Roadmap

## Definition of done

An architecture is shown as available only after the application can locate a
matching local trainer and base model, validate its dataset contract, run that
trainer without substitution, validate the saved adapter, and render a fixed
seed base-versus-LoRA comparison with the matching inference backend.

## Phase 1 — Integrity foundation (completed 2026-09-24)

1. Reject unreadable reference images and empty captions during every export.
2. Record image dimensions, colour mode, caption hash, output hash, trainer
   command, base-model hash, exit code, and parsed loss points in provenance.
3. Make the UI and supervisor consume the runtime capability registry rather
   than the aspirational architecture catalogue.
4. Add regression tests proving an invalid image, random safetensors file,
   missing provenance, or unavailable runtime cannot be called real.

Acceptance: no export contains an unreadable image; no UI/API label says
"available", "trained", or "tested" without matching evidence.

Evidence: dataset exports reject corrupt inputs and record source/export/caption
SHA-256 values plus dimensions and colour mode. Training provenance binds the
model output to its command, checkpoint, exit code, and output SHA-256. The
supervisor/UI use runtime capabilities; regression suite: 39 passing tests.

## Phase 2 — SDXL production baseline

1. Validate Kohya `sdxl_train_network.py` CLI compatibility before launch.
2. Add a controlled 2–4 image, 1–2 step smoke-run command and preserve its
   logs/provenance separately from production output.
3. Add real SDXL base/LoRA A/B rendering at fixed seed, prompt, resolution,
   and sampler; save both images and settings.

Acceptance: a verified SDXL artifact loads in both Diffusers and ComfyUI, with
an auditable production record and real A/B images.

## Phase 3 — FLUX.1-dev production baseline

1. Validate FLUX UNet, CLIP-L, T5XXL, AE, and `flux_train_network.py` as one
compatible installation.
2. Implement a FLUX-specific command/profile and parser; do not reuse SDXL
arguments, metadata, or inference code.
3. Add FLUX-specific adapter validation and fixed-seed FLUX A/B inference.

Acceptance: an independently loadable FLUX LoRA with a completed local smoke
run and saved comparison output.

## Phase 4 — Image-model adapters, one at a time

Implement Qwen Image, then Z-Image. For each: select an upstream supported
trainer, pin its version and model format, add a dedicated adapter command,
dataset contract, metadata validator, inference loader, and smoke/e2e test.

Acceptance per model: no shared SDXL/FLUX fallback; successful training and
matching-model inference proven on the target installation.

## Phase 5 — Video LoRA adapters, one at a time

Implement Wan, LTX-Video, and MiniMax/Hunyuan only after their installed
trainer and open model terms are confirmed. Each adapter needs frame-sequence
dataset export (fps, frame count, clip duration, captions), temporal LoRA
validation, and a short real video generation comparison.

Acceptance per model: the generated LoRA is loaded by its intended video
pipeline and changes a fixed input/prompt video run.

## Phase 6 — Portable runtime layer (in progress)

1. Replace hard-coded `D:/ComfyUI/...` locations with a persisted configurable
runtime profile plus discovery.
2. Support CUDA first, then MPS/Apple Silicon and CPU only where a trainer
actually supports them. Unsupported devices remain explicitly unavailable.
3. Add per-backend memory estimates and preflight checks; never claim GPU
agnostic behavior merely because a device was detected.

Acceptance: paths and backend are configurable; capability output accurately
states support for the active machine.

## Phase 7 — Quality evaluation and release gates

1. Dataset diversity/caption-quality checks.
2. Repeatable held-out prompts, fixed-seed A/B renders, and human review
records.
3. Compatibility load tests in Diffusers and the configured ComfyUI install.
4. Integration and mutation tests for blocked fallbacks and false provenance.

Acceptance: a release artifact has reproducible training, validation, and
comparison evidence—not only a non-zero tensor norm or file size.
