# 📘 Geekatplay LoRA Maker: Comprehensive User Guide

**Author**: Vladimir Chopine ([Geekatplay Studio](https://github.com/GeekatplayStudio))  
**Repository**: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)  
**Version**: 2.0.0

---

## 🌟 Introduction

**Geekatplay LoRA Maker** is a universal, multi-agent AI desktop studio engineered for digital artists, animators, photographers, and developers. It allows you to:
1. Extract high-quality keyframes from video footage or import image sets.
2. Fit and bucket any aspect ratio (16:9, 9:16, 2:3, 3:2, 4:3, 1:1) **without distortion, stretching, or squeezing**.
3. Auto-generate rich training captions using local multimodal vision models (via Ollama).
4. Auto-detect your GPU hardware and automatically tune batch sizes, gradient accumulation, and precision.
5. Train and evaluate authentic Low-Rank Adaptation (LoRA) models for **SDXL 1.0**, **FLUX.1 [dev/schnell]**, and **Video Diffusion (Wan2.1, Hunyuan, LTX-Video)**.

---

## 🖥️ Hardware Requirements & Auto-Adaptive Scaling

Geekatplay LoRA Maker automatically probes your system's hardware on startup and dynamically tunes training parameters according to your GPU tier:

| Hardware Tier | GPU VRAM | Recommended Models | Batch Size | Grad Accum | Precision | Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultra Studio Tier** | **24 GB+** (RTX 3090, 4090, A100) | FLUX.1 [dev], SDXL 1.0, Wan2.1, Hunyuan | 2 | 1 | `bf16` / `fp16` | AdamW |
| **High Performance Tier** | **16 GB – 23 GB** (RTX 4080, 3080 16GB) | FLUX.1 [schnell], SDXL 1.0, Wan2.1 Turbo | 1 | 2 | `bf16` / `fp16` | AdamW8bit |
| **Mid-Range Tier** | **10 GB – 15 GB** (RTX 3060 12GB, 4070) | SDXL 1.0, LTX-Video Turbo | 1 | 4 | `fp16` | AdamW8bit |
| **Entry / Budget Tier** | **6 GB – 9 GB** (RTX 3060 6GB/8GB, 2060) | SDXL 1.0 (768px), SD 1.5 | 1 | 4 | `fp16` | AdamW8bit |
| **CPU / Minimal Fallback**| **<6 GB or CPU / Apple Silicon** | SDXL base inference & dataset tagging | 1 | 8 | `no` | AdamW |

You can click the **Hardware Status Badge** in the top navigation bar at any time to inspect your detected VRAM, CUDA compute capability, and recommended training limits.

---

## 🚀 Quickstart Guide

### 1. Launching the Studio
Run the launcher:
```bash
# On Windows:
start.bat

# Or via Python:
python run_app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 2. Initializing a Project
Click **`+ New Project`** in the top navigation. You can select any of our 1-click presets:
- **👤 Photorealism**: Human portraits, candid photography, 85mm cinematic stills.
- **✨ Anime & Manga**: Cel shading, expressive linework, vibrant keyframe colors.
- **🧸 Stylized 3D**: Subsurface scattering, Pixar/DreamWorks character features.
- **🎞️ Vintage Animation**: 1930s rubber-hose, classic ink and paint, Fleischer style.
- **🪐 Concept Art**: Sci-fi cities, alien vistas, architectural matte paintings.
- **⌚ Product Design**: Commercial product studio hero shots, softbox lighting.

Each preset auto-fills the project folder, subject name, unique trigger token, and descriptive prompt.

---

## 🎬 Workflow Walkthrough

### Step 1: Ingesting Footage & Cutting Keyframes
1. In the **Timeline Scrubber** tab, load a video file or paste a local filepath.
2. Scrub through the timeline or use playback controls:
   - `Arrow Left` / `Arrow Right`: Step 1 frame.
   - `Shift + Arrow`: Step 10 frames.
   - `C`: Cut the current frame.
3. Frames are automatically saved to `Keyframes_Out/` preserving native aspect ratios.

### Step 2: Auto-Captioning with Local Vision AI
1. In the **Keyframe Gallery** tab, select your preferred local Ollama vision model (`qwen2.5vl:7b`, `llama3.2-vision`, `llava`).
2. Click **`⚡ Auto-Caption All Keyframes`**.
3. The local vision AI inspects each frame, generating rich detailed descriptions prefixed by your unique trigger token (e.g. `elena_portrait, woman standing in warm sunlight...`).

### Step 3: Interactive Supervisor Guidance
Switch to the **Supervisor Studio** tab to interact with the multi-agent orchestrator:
- Ask for prompt improvements or character turnaround guidance.
- Generate structured, scene-by-scene animation scripts (`Prompts/animation_script.json`).
- Run a pre-training dataset audit to ensure image-caption parity.

### Step 4: LoRA Training
In the **LoRA Training** tab:
1. Select your target base architecture (SDXL 1.0, FLUX.1, or Wan2.1 Video).
2. Choose your framing strategy:
   - **Bucket**: Snaps to native aspect-ratio buckets (zero distortion).
   - **Pad**: Letterboxes/pillarboxes onto square canvas.
   - **Crop**: Center-crops excess canvas.
3. Click **`Start Training Pipeline`**.
4. Real-time telemetry streams training loss, epoch progression, step count, and GPU VRAM utilization.
5. The serialized `.safetensors` file is saved to `Training/output/`.

### Step 5: Test Bench & Diagnostics
In the **Diagnostics & Testing** tab:
1. View detailed Frobenius norms, tensor key shapes, and adapter parameter breakdown.
2. Select any suggested prompt or enter a custom prompt.
3. Adjust the **LoRA Weight Slider** (`0.00` to `1.50`):
   - **`0.00`**: Disables LoRA completely and runs the pure, unadapted base model.
   - **`0.85`**: Applies balanced LoRA stylization.
   - **`1.20+`**: Strong LoRA emphasis.
4. Click **`⚡ Render Real Test Sample`** to inspect the live denoised output directly on GPU.

---

## 🔌 Exporting to ComfyUI & External Tools
- Click **`🚀 1-Click Deploy to ComfyUI`** to copy trained `.safetensors` directly to your local ComfyUI models directory (`models/loras/`).
- An executable training workflow JSON is always exported to `Training/comfyui_training_workflow.json`.
- A standard Kohya config file is exported to `Training/kohya_lora_config.toml`.

---

## 📄 License & Attribution
Created by **Vladimir Chopine** at **Geekatplay Studio**.  
Licensed under the open-source **MIT License**.  
Visit the project repository: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)
