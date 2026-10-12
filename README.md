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

Geekatplay LoRA Maker automatically probes your system's hardware on startup and dynamically tunes training parameters according to your detected GPU tier:

| Hardware Tier | GPU VRAM | Recommended Models | Batch Size | Grad Accum | Precision | Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultra Studio Tier** | **24 GB+** (RTX 3090, 4090, A100) | FLUX.1 [dev], SDXL 1.0, Wan2.1, Hunyuan | 2 | 1 | `bf16` / `fp16` | AdamW |
| **High Performance Tier** | **16 GB – 23 GB** (RTX 4080, 3080 16GB) | FLUX.1 [schnell], SDXL 1.0, Wan2.1 Turbo | 1 | 2 | `bf16` / `fp16` | AdamW8bit |
| **Mid-Range Tier** | **10 GB – 15 GB** (RTX 3060 12GB, 4070) | SDXL 1.0, LTX-Video Turbo, Qwen | 1 | 4 | `fp16` | AdamW8bit |
| **Entry / Budget Tier** | **6 GB – 9 GB** (RTX 3060 6GB/8GB, 2060) | SDXL 1.0 (768px), SD 1.5 | 1 | 4 | `fp16` | AdamW8bit |
| **CPU / Minimal Fallback**| **<6 GB or CPU / Apple Silicon** | SDXL base inference & dataset tagging | 1 | 8 | `no` | AdamW |

---

## 🚀 Key Features

### 1. Zero-Distortion Aspect Ratio Bucketing
Never squeeze or stretch your training subjects:
- **Bucket Mode**: Automatically categorizes keyframes into native aspect-ratio buckets (`16:9`, `9:16`, `2:3`, `3:2`, `4:3`, `3:4`, `1:1`) snapped to optimal multiples of 64.
- **Pad & Crop Modes**: Neutral background letterboxing or uniform center-cropping without geometric distortion.

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

| Preset | Steps | Resolution | Rank / Alpha | LR | Typical time (12 GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🧪 **Smoke test** | 20 | 512 | 4 / 4 | 1e-4 | ~3 min |
| 🚀 **Fast character** (default) | 400 | 768 | 8 / 8 | 1.5e-4 | ~14 min |
| ⚖️ **Balanced** | 900 | 1024 | 16 / 8 | 1e-4 | ~35 min |
| 💎 **Quality** | 1600 | 1024 | 32 / 16 | 8e-5 | ~65 min |
| 🛠️ **Custom** | your fields | your fields | your fields | your fields | – |

Two engines are used, and the Studio tells you which one it picked:
- **Kohya sd-scripts** (SDXL, FLUX.1) when a checkout is found via `KOHYA_ROOT`, `KOHYA_SD_SCRIPTS_DIR`, or the usual `D:\kohya_ss\sd-scripts` locations. The trainer runs with Kohya's own `venv` interpreter when present.
- **Built-in Diffusers/PEFT trainer** (SDXL) when Kohya is not installed. Real gradient descent on the UNet attention projections, saved in Kohya/ComfyUI key format.

Every finished run is verified before it is reported as complete: the safetensors file must contain LoRA pairs for the right architecture **and** non-zero up-projection tensors (they start at zero and only training changes them). Architectures without a verified trainer (Qwen, Z-Image, Wan, LTX, Hunyuan) are shown as unavailable rather than "trained" by a stand-in.

**Prove it works on your machine** (after `install.bat`):
```bash
verify_training.bat            # 20 real steps on your SDXL checkpoint, or a tiny test model
python verify_training.py --tiny   # CPU-only check of the pipeline (~10 s)
pytest tests/test_training_e2e.py  # full pipeline on CPU: trains, verifies, loads, checks learning
```

Every run also records a **learning check**: the loss on a fixed batch is measured before and after training and stored in `<model>.metrics.json` as `fixed_batch_loss_before/after`. The end-to-end test asserts that it drops, that the adapter changes the UNet output, and that scale 0 reproduces the base model.

### No simulations policy
Nothing in the Studio fabricates a result. Concretely:
- Training telemetry comes only from the trainer's stdout; the UI shows "telemetry not verified" otherwise.
- A LoRA is "verified" only when its tensors are trained (non-zero up-projections) and, when present, its provenance file hashes match.
- Captioning returns an error when Ollama is unavailable instead of writing a generic caption.
- Keyframe suggestions are ranked by measured frame-to-frame change, with the score shown.
- Storyboards generated from the built-in template are labelled `"source": "template"`.
- Test renders require a real CUDA SDXL pipeline; there is no placeholder image.

### 6. Multi-Architecture LoRA Training
- **SDXL 1.0**: Native Kohya format with 640-dim and 1280-dim UNet cross-attention adapters.
- **FLUX.1 [dev/schnell]**: 3072-dim single and double transformer DiT blocks.
- **Qwen & Z-Image Support**: HuggingFace cache discovery and model auto-tuning.
- **Video Diffusion**: Wan2.1, Hunyuan Video, Cosmos, LTX-Video spatiotemporal modules.
- **Dual Mode**: Authentic PyTorch GPU gradient descent or dry-run validation.

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

## 📦 Installation & Setup

### Prerequisites
1. **Python 3.10 – 3.12**
2. **NVIDIA GPU** with CUDA support (or Apple Silicon / CPU for lightweight tagging)
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

## 📂 Project Structure

```
LoraMaker/
├── app/
│   ├── agents/          # Multi-agent orchestrators (Supervisor, Visual, Story, Video)
│   ├── api/             # FastAPI REST endpoints (Projects, Training, Evaluation, System)
│   ├── core/            # Configuration and constants
│   ├── services/        # Hardware auto-detection, video scrubbing, dataset formatting, training, diagnostics
│   └── static/          # Modern Web UI (CSS glassmorphism, responsive JS controllers)
├── projects/            # Project directories with Keyframes_Out and training datasets
├── tests/               # Automated test suite (Pytest)
├── install.bat          # Automated Windows 1-click installer (venv + PyTorch CUDA + packages)
├── start.bat            # Windows 1-click startup script (auto browser launch)
├── install.ps1          # PowerShell automated installer
├── start.ps1            # PowerShell launcher
├── run_app.py           # Application entrypoint launcher
├── requirements.txt     # Python dependencies
└── README.md            # Studio documentation
```

---

## 🧪 Testing

Run the automated test suite:
```bash
pytest tests/
```

---

## 👤 Author & Credits

Created by **Vladimir Chopine**  
**Geekatplay Studio**: [https://github.com/GeekatplayStudio](https://github.com/GeekatplayStudio)  
Official Repository: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
