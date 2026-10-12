# ⚡ Geekatplay LoRA Maker: Universal Multi-Agent Studio

**Created by Vladimir Chopine** ([Geekatplay Studio](https://github.com/GeekatplayStudio))  
**Official Repository**: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)  
**Version**: 2.0.0 • **License**: MIT

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Local_Vision-Ollama-black.svg?style=flat)](https://ollama.ai)
[![PyTorch](https://img.shields.io/badge/Hardware-Auto--Adaptive-76B900.svg?style=flat&logo=nvidia)](https://pytorch.org)
[![ComfyUI](https://img.shields.io/badge/Deploy-ComfyUI_Ready-orange.svg)](https://comfy.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🌟 Overview

**Geekatplay LoRA Maker** is a desktop AI workstation engineered for creators, animators, digital artists, and developers. It bridges the gap between raw video footage/image collections and production-ready Low-Rank Adaptation (LoRA) models across all visual genres:
- **Photorealistic Portraits & Likeness**: Natural skin texture, candid focal depth, studio lighting.
- **Anime & Manga Illustration**: Sharp cel shading, expressive linework, vibrant aesthetic keyframes.
- **Stylized 3D Character Features**: Subsurface scattering, Pixar/DreamWorks clay aesthetic, soft three-point lighting.
- **Vintage & Classic Animation**: 1930s rubber-hose, classic ink and paint, Fleischer cel aesthetics.
- **Sci-Fi & Fantasy Concept Art**: Monumental architectural vistas, matte paintings, atmospheric worldbuilding.
- **Commercial Product Design**: Clean studio hero backdrops, softbox rim lighting, luxury industrial materials.

---

## 🖥️ Universal Hardware Auto-Detection & Adaptive Scaling

Geekatplay LoRA Maker probes your hardware on startup and resolves each training preset against your GPU tier:

| Hardware Tier | GPU VRAM | Can train | Preset resolution cap | Precision | Optimizer |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Ultra Studio** | **24 GB+** (RTX 3090, 4090, A100) | SDXL 1.0, FLUX.1 [dev] | 1024 | bf16 on Ampere+, else fp16 | AdamW |
| **High Performance** | **16 – 23 GB** (RTX 4080, 3080 16GB) | SDXL 1.0, FLUX.1 [dev] (fp8 base, block swap) | 1024 (FLUX 768) | bf16 on Ampere+, else fp16 | AdamW8bit¹ |
| **Mid-Range** | **10 – 15 GB** (RTX 3060 12GB, 4070) | SDXL 1.0 (FLUX.1 [dev] not recommended) | 1024 | bf16 on Ampere+, else fp16 | AdamW8bit¹ |
| **Entry / Budget** | **6 – 9 GB** (RTX 2060, 3060 8GB) | SDXL 1.0 | 768 | bf16 on Ampere+, else fp16 | AdamW8bit¹ |
| **No CUDA GPU** | CPU / Apple Silicon / under 6 GB | nothing (tagging, dataset prep, inspection) | – | – | – |

¹ Used when `bitsandbytes` imports in the trainer's Python; otherwise the run uses plain AdamW.

> `GET /api/system/hardware` also reports batch-size and gradient-accumulation hints. Training presets do not apply gradient accumulation; they rely on batch size 1 and a hard step cap. Whether an architecture can run is decided by the [support matrix](#6-supported-architectures) (trainer and weights present), not by tier alone.

---

## 🚀 Key Features

### 1. Zero-Distortion Aspect Ratio Bucketing
Never squeeze or stretch your training subjects:
- **Bucket Mode**: Automatically categorizes keyframes into native aspect-ratio buckets (`16:9`, `9:16`, `2:3`, `3:2`, `4:3`, `3:4`, `1:1`) snapped to optimal multiples of 64.
- **Pad & Crop Modes**: Neutral background letterboxing or uniform center-cropping without geometric distortion.
- Bucketing is executed by the Kohya engine. The built-in SDXL engine trains square images and treats *bucket* as *pad*.

### 2. Precision Timeline Scrubber & Frame Extraction
- Real-time video player with sub-frame accuracy (`←` / `→` for 1 frame, `Shift + ←` / `Shift + →` for 10 frames).
- Press `C` to cut any frame directly into your dataset with clean, sequential naming.

### 3. Local Multimodal Vision Captioning (Ollama)
- Automatic discovery of local vision models (`qwen2.5vl:7b`, `llama3.2-vision`, `llava`, `minicpm-v`).
- Automatically prepends your clean, unique trigger token to all training captions.

### 4. Autonomous Supervisor & Specialist Agents
- **Supervisor Agent**: Central creative orchestrator guiding you through dataset preparation and model selection.
- **Visual Agent**: Recommends style tokens, character turnaround sheets, and pose variety.
- **Story Agent**: Formulates multi-scene screenplay storyboards (JSON) formatted with camera and motion cues.
- **Trainer Agent**: Audits training readiness and formats Kohya_ss & ComfyUI configurations.

### 5. Verified, Step-Budgeted Training Presets
Every run uses a preset that fixes the number of optimizer steps up-front, caches latents and text-encoder outputs once, trains the UNet only, and caps resolution to what your GPU tier can hold. Run time is therefore predictable no matter how many keyframes you have.

| Preset | Steps | Resolution | Rank / Alpha | LR | 24 GB+ | 16-23 GB | 10-15 GB | 6-9 GB |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| 🧪 **Smoke test** | 20 | 512 | 4 / 4 | 1e-4 | 1 min | 2 min | 3 min | 4 min |
| 🚀 **Fast character** (default) | 400 | 768 | 8 / 8 | 1.5e-4 | 5 min | 8 min | 14 min | 25 min |
| ⚖️ **Balanced** | 900 | 1024 | 16 / 8 | 1e-4 | 12 min | 20 min | 35 min | 60 min |
| 💎 **Quality** | 1600 | 1024 | 32 / 16 | 8e-5 | 22 min | 38 min | 65 min | 110 min |
| 🛠️ **Custom** | your fields | your fields | your fields | your fields | – | – | – | – |

The per-tier times are **planning estimates, not benchmarks**. During a run the Studio shows an ETA computed from the real step rate. Resolution is capped by tier (768 px on 6-9 GB cards; FLUX is capped at 768 px under 24 GB). Any preset field can be overridden through the API without switching to *Custom*.

Two engines are used, and the Studio tells you which one it picked:
- **Kohya sd-scripts** (SDXL, FLUX.1) when a checkout is found via `KOHYA_ROOT`, `KOHYA_SD_SCRIPTS_DIR`, or the usual `D:\kohya_ss\sd-scripts` locations. The trainer runs with Kohya's own `venv` interpreter when present.
- **Built-in Diffusers/PEFT trainer** (SDXL) when Kohya is not installed. Real gradient descent on the UNet attention projections (`to_q`, `to_k`, `to_v`, `to_out`), text encoders frozen, saved in Kohya/ComfyUI key format. It trains square images (*pad* or *crop*; *bucket* framing is treated as *pad*). Aspect-ratio bucketing requires the Kohya engine.

Every finished run is verified before it is reported as complete: the safetensors file must contain LoRA pairs for the right architecture **and** non-zero up-projection tensors (they start at zero and only training changes them). Architectures without a verified trainer (Qwen, Z-Image, Wan, LTX, Hunyuan) are shown as unavailable rather than "trained" by a stand-in.

**Prove it works on your machine** (after `install.bat`):
```bash
verify_training.bat            # 20 real steps on your SDXL checkpoint, or a tiny test model
python verify_training.py --tiny   # CPU-only check of the pipeline (~10 s)
pytest tests/test_training_e2e.py  # full pipeline on CPU: trains, verifies, loads, checks learning
```

Every run also records a **learning check**: the loss on a fixed batch is measured before and after training and stored in `<model>.metrics.json` as `fixed_batch_loss_before/after`. The end-to-end test asserts that it drops, that the adapter changes the UNet output, and that scale 0 reproduces the base model.

### 6. Supported Architectures
The Studio only offers a model for training when a verified trainer **and** its weights are present. Everything else is shown as unavailable with the reason.

| Architecture | Trainer | Status |
| :--- | :--- | :--- |
| **SDXL 1.0** | Kohya sd-scripts if found, otherwise the built-in Diffusers/PEFT engine | ✅ Supported. The built-in engine is exercised end-to-end by the automated tests (tiny SDXL, CPU). |
| **FLUX.1 [dev]** | Kohya sd-scripts only; needs `flux1-dev`, `clip_l`, `t5xxl` and `ae` | ✅ Supported when those files are found. Command construction is tested; a full FLUX run is not part of CI. |
| Qwen-Image, Z-Image, Wan 2.1, MiniMax / Hunyuan Video, LTX-Video Turbo | none | ⛔ Registered, but no verified trainer is installed, so they are reported unavailable. |
| FLUX.1 [schnell], Wan 2.1 Turbo, LTX-Video, CogVideoX-5B | none | ⛔ Cards are shown for reference only; there is no trainer integration. |

Training itself needs an **NVIDIA CUDA GPU**. Apple Silicon and CPU-only machines can use every other feature (video scrubbing, captioning, dataset export, inspection).

### 7. Interactive Model Diagnostics & Test Bench
- **Frobenius Norm & Weight Telemetry**: Inspect raw adapter tensors ($\|W_{\text{down}}\|_F$, $\|W_{\text{up}}\|_F$) and effective weight deltas.
- **LoRA Weight Scale Slider (`0.00` to `1.50`)**:
  - `0.00`: Completely unloads LoRA adapter weights for a 100% clean baseline render.
  - `0.85`: Standard balanced stylization.
  - `1.20+`: High-intensity artistic stylization.
- **1-Click ComfyUI Deployment**: Instantly copy trained weights to ComfyUI `models/loras/`.

### 8. Multi-Drive Storage & Visual Path Browser
- **Drive Protection**: Redirect heavy base model downloads to external secondary drives (e.g., `O:\ComfyUI\models`) to keep `C:` drive clean.
- **Multiple Search Roots**: Add and remove multiple model directories across `C:`, `D:`, `F:`, `O:`, etc.
- **Visual Directory Browser**: Navigate folders interactively without manual typing.

### 9. System Error & Diagnostics Telemetry Console
- **Live Exception Tracking**: Captures Python stack traces, HTTP status, and request payloads.
- **Context-Aware Suggestions**: Intelligent recommendations for missing checkpoints, uncaptioned frames, and VRAM limits.
- **1-Click Diagnostic Report**: Copies a complete, formatted Markdown snapshot of system state and errors directly to the clipboard.

---

## 🛡️ Reliability & Honesty Guarantees

### Engineering safeguards
- **Local-only API**: CORS allows only the Studio's own `127.0.0.1` / `localhost` origins, so other pages open in your browser cannot drive the API.
- **Typed, scoped file endpoints**: image and video serving return media files only; caption edits and frame deletes touch only files inside a Studio project (a folder with `project.json`); uploads are renamed to a safe basename.
- **One training run per project**: a second start is refused while a run is active, and a user stop is reported as *stopped*, never *failed*.
- **Live telemetry**: trainer output is read as it arrives, so sparse log lines and tqdm `\r` updates show up immediately.
- **Background batch captioning**: `POST /api/captions/batch` returns at once; progress comes from `GET /api/captions/batch/status`.
- **Caching**: settings and LoRA verification results are memoised until the underlying files change.
- **No stray folders**: Windows drive paths such as `D:/ComfyUI/...` are never created or resolved on macOS/Linux.
- **CI**: GitHub Actions runs the full suite on Python 3.11 and 3.12, including the real CPU training test.

### No simulations policy
Nothing in the Studio fabricates a result:
- Training telemetry comes only from the trainer's output; the UI says "telemetry not verified" otherwise.
- A LoRA is "verified" only when its up-projection tensors are non-zero (trained) and, when a provenance file exists, its hashes match.
- Every run records a learning check: the loss on a fixed batch before and after training, stored in `<model>.metrics.json`.
- Captioning returns an error when Ollama is unavailable instead of writing a generic caption; failed frames stay uncaptioned and block training.
- Keyframe suggestions are ranked by measured frame-to-frame change, with the score shown.
- Storyboards from the built-in template are labelled `"source": "template"`.
- Test renders require a real CUDA SDXL pipeline; there is no placeholder image.

---

## 📦 Installation & Setup

### Prerequisites
1. **Python 3.10 – 3.13** (3.14 is not yet validated for the CUDA wheels or Kohya; the installer warns)
2. **NVIDIA GPU** with a recent driver (560+ for the CUDA 12.6 wheels). Apple Silicon / CPU machines can do everything except training
3. **[Ollama](https://ollama.ai)** (Optional, recommended for automated local vision captions):
   ```bash
   ollama pull qwen2.5vl:7b
   ```

### Quick Installation

#### Windows (1-Click Automated Setup)
1. Double-click **`install.bat`** (or run `.\install.ps1` in PowerShell). This automatically:
   - Verifies Python 3.10+
   - Creates and activates a dedicated virtual environment (`venv`)
   - Detects your NVIDIA GPU and installs PyTorch with CUDA 12.6 (falls back to 12.8)
   - Installs all dependencies from `requirements.txt`
2. Double-click **`start.bat`** (or run `.\start.ps1` in PowerShell) to launch the Studio and automatically open [http://127.0.0.1:7860](http://127.0.0.1:7860).

#### Manual Setup
```bash
# Clone the repository
git clone https://github.com/GeekatplayStudio/LoraMaker.git
cd LoraMaker

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA 12.6 & dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Launch application
python run_app.py
```

---

## ⚙️ Configuration

Settings come from three places, highest priority first:

1. The **Settings** tab, saved to `user_settings.json` (git-ignored).
2. Environment variables, read at startup. A `.env` file is **not** loaded automatically; set them in your shell or in `start.bat`.
3. Built-in defaults, which assume the original Windows layout under `D:\`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `COMFYUI_ROOT` | `D:/ComfyUI/ComfyUI` | ComfyUI folder that contains `models/` (checkpoints, loras) |
| `KOHYA_ROOT` | `D:/kohya_ss/sd-scripts` | Kohya checkout or its `sd-scripts` folder |
| `KOHYA_FALLBACK_ROOT` | `D:/ComfyUI/lora-training/fluxgym/sd-scripts` | Second Kohya location |
| `KOHYA_SD_SCRIPTS_DIR` | unset | Checked before the two above when set |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama endpoint for captioning and chat |
| `HF_TOKEN` | unset | Read by `huggingface_hub` for gated models such as FLUX.1 [dev] |
| `LORAMAKER_ALLOW_CPU_TRAINING` | unset | Test/CI only: lets the built-in trainer run on CPU |

Host and port are command-line flags: `python run_app.py --host 127.0.0.1 --port 7860 --no-browser`.

---

## 🔌 API Overview

Interactive docs are served at `/docs`. The training and captioning endpoints most integrations use:

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| GET | `/api/training/capabilities` | Per-architecture availability, engine, and missing requirements |
| GET | `/api/training/presets?project_dir=&base_model=` | Presets with the plan each resolves to on this GPU and dataset |
| POST | `/api/training/start` | Start a run (`preset`, optional overrides, `execution_mode`: `real_gpu` or `validate_only`) |
| GET | `/api/training/status?project_dir=` | Snapshot: step, loss history, ETA, engine, plan, learning check, log |
| POST | `/api/training/stop` | Stop the active run |
| POST | `/api/captions/batch` | Start background captioning (`409` if one is already running) |
| GET | `/api/captions/batch/status?project_dir=` | Progress, failure count, per-frame results |
| POST | `/api/captions/batch/cancel` | Stop after the current frame |

```json
POST /api/training/start
{
  "project_dir": "C:/LoraMaker/projects/Zorb",
  "base_model": "sdxl-1.0",
  "preset": "fast",
  "execution_mode": "real_gpu",
  "framing_mode": "pad",
  "max_train_steps": 600
}
```

Fields left out or `null` take the preset value. Only the `custom` preset also reads `epochs` and `repeats`.

---

## 🩺 Troubleshooting

| You see | Meaning | What to do |
| :--- | :--- | :--- |
| `CUDA GPU unavailable` | PyTorch cannot see an NVIDIA GPU | Check `nvidia-smi`, rerun `install.bat` so the CUDA build of PyTorch is installed |
| `missing base weights: SDXL 1.0` | No SDXL checkpoint in any configured model folder | Add the folder in **Settings**, or place `sd_xl_base_1.0.safetensors` under a `checkpoints/` folder |
| `no verified trainer installed for ...` | The architecture has no trainer integration | Use SDXL, or FLUX.1 [dev] with Kohya sd-scripts installed |
| `Trainer exited with code N: ... out of memory` | Not enough VRAM | Use *Fast* or *Smoke test*, lower the resolution or rank |
| `Output adapter is not trained` | The run finished but the adapter weights never moved | Do not use the file; check the log and learning rate, then rerun |
| `A training run is already in progress` | One run per project at a time | Stop the active run first |
| `bitsandbytes not importable` | 8-bit AdamW is unavailable | Informational: the run uses standard AdamW and a little more VRAM |
| Frames show no caption after batch captioning | Ollama failed for those frames; nothing was written | Start Ollama and `ollama pull qwen2.5vl:7b`, or caption by hand, then rerun |

To confirm that training works at all on a machine, run `verify_training.bat` (see [Verified training presets](#5-verified-step-budgeted-training-presets)).

---

## 📂 Project Structure

```
LoraMaker/
├── app/
│   ├── agents/          # Multi-agent orchestrators (Supervisor, Visual, Story, Video, Trainer)
│   ├── api/             # FastAPI routers (projects, video, captions, datasets, training, evaluation, system)
│   ├── core/            # Settings (config.py) and path guards (paths.py)
│   ├── services/
│   │   ├── training_service.py        # Engine selection, preflight, subprocess, verification
│   │   ├── training_presets.py        # Step-budgeted presets resolved per GPU tier
│   │   ├── diffusers_lora_worker.py   # Built-in SDXL LoRA trainer (Diffusers + PEFT)
│   │   └── ...                        # hardware, dataset, evaluation, vision, video, settings, diagnostics
│   └── static/          # Web UI (HTML, CSS, JS controllers)
├── tests/               # Pytest suite, including real end-to-end training
├── .github/workflows/   # CI: pytest on Python 3.11 and 3.12
├── projects/            # Your projects (Keyframes_Out, Training/output, ...)
├── install.bat / .ps1   # Windows installers (venv, CUDA PyTorch, packages)
├── start.bat / .ps1     # Launchers
├── verify_training.bat  # Proves real LoRA training works on this machine
├── verify_training.py   # Same check, any OS (--tiny for a CPU-only run)
├── run_app.py           # Application entrypoint
├── pytest.ini           # Test configuration (timeout)
├── requirements.txt     # Python dependencies
├── CHANGELOG.md         # Notable changes
└── README.md            # This file
```

---

## 🧪 Testing

```bash
pytest                              # whole suite, about 10 seconds
pytest tests/test_training_e2e.py   # real CPU training on a tiny SDXL test model
```

| File | What it proves |
| :--- | :--- |
| `test_training_e2e.py` | Full pipeline trains: one loss line per step, fixed-batch loss drops, provenance hashes match, tensors are non-zero, the LoRA loads in diffusers and changes the UNet output, scale 0 reproduces the base model |
| `test_training_presets.py` | Preset resolution, tier caps, optimizer fallback, Kohya command flags, engine selection, rejection of untrained adapters |
| `test_hardening.py` | Path guards, CORS, upload sanitising, concurrent-start rejection, stop versus fail, caches, background captioning, no stray drive folders |
| `test_training.py`, `test_evaluation.py`, `test_dataset_service.py`, `test_api.py`, ... | Services and API behaviour |

Notes:
- The end-to-end test downloads a small public test model (about 6 MB, `hf-internal-testing/tiny-stable-diffusion-xl-pipe`) once and is skipped when it cannot be fetched.
- Tests are isolated: they never read or write your `user_settings.json`, global settings, or cache environment variables.
- The drive-letter browse test runs on Windows only.
- Kohya runs and CUDA are not exercised in CI; use `verify_training.bat` on a GPU machine.

---

## 👤 Author & Credits

Created by **Vladimir Chopine**  
**Geekatplay Studio**: [https://github.com/GeekatplayStudio](https://github.com/GeekatplayStudio)  
Official Repository: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
