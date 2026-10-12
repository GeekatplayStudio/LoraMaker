# Changelog

All notable changes to Geekatplay LoRA Maker. The version number in `app/core/config.py`
has not been bumped for these changes yet.

## Unreleased

### Added
- **Built-in SDXL LoRA trainer** (Diffusers + PEFT) that performs real gradient descent with
  cached latents and text embeddings, and writes Kohya/ComfyUI-format LoRAs.
- **Step-budgeted presets**: Smoke test, Fast character, Balanced, Quality, and Custom, resolved
  per GPU tier and dataset size, with a preset selector and plan badge in the UI.
- **Learning check**: loss on a fixed batch before and after every run, stored in `.metrics.json`.
- `verify_training.py` / `verify_training.bat` to prove training works on a given machine.
- End-to-end CPU training test, preset tests, and a hardening test module.
- `GET /api/training/presets`, `GET /api/captions/batch/status`, `POST /api/captions/batch/cancel`.
- GitHub Actions workflow running the suite on Python 3.11 and 3.12.
- `peft`, `bitsandbytes` (CUDA), `numpy`, `huggingface_hub`, `pytest-timeout` dependencies.

### Changed
- Training engine selection: Kohya sd-scripts when found (run with its own venv interpreter),
  built-in engine otherwise. Architectures without a verified trainer are reported unavailable.
- Kohya command now caches latents to disk, uses SDPA, constant LR, a hard step cap, and enables
  8-bit AdamW only when `bitsandbytes` imports.
- Batch captioning is a background job; the gallery polls its progress.
- Installers use the CUDA 12.6 PyTorch index (12.8 fallback) and warn on Python 3.14+.
- Dependency floors raised in `requirements.txt`.
- README rewritten: support matrix, presets, configuration, API overview, troubleshooting.

### Fixed
- The previous non-Kohya worker produced random tensors and a scripted loss curve; it is gone.
- LoRA verification trusted file headers; it now requires trained (non-zero) up-projections.
- Captioning wrote a generic caption when Ollama failed; it now reports the error instead.
- Keyframe suggestions invented pose labels; they are now ranked by measured frame change.
- Stale concept folders from earlier exports silently multiplied dataset size and run time.
- A user stop was reported as a failure; job state is now lock-protected and snapshot on read.
- Saving settings created a `D:` folder in the working directory on macOS/Linux and redirected
  the Hugging Face cache into it; Windows drive paths are no longer resolved or created off Windows.
- Tests no longer read or write the real `user_settings.json`, global settings, or cache env vars.

### Security
- CORS limited to the Studio's local origins (was `*` with credentials).
- Media endpoints serve media files only; caption edits and frame deletes are limited to files
  inside a Studio project; uploads use a sanitised filename and require a project.
