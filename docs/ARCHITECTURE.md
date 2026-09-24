# ⚡ Geekatplay LoRA Maker: Technical Architecture

**Created by Vladimir Chopine** ([Geekatplay Studio](https://github.com/GeekatplayStudio))  
**Official Repository**: [https://github.com/GeekatplayStudio/LoraMaker.git](https://github.com/GeekatplayStudio/LoraMaker.git)  
**Version**: 2.0.0 • **License**: MIT

---

## 1. Executive Overview

**Geekatplay LoRA Maker** is a universal desktop workstation designed for extracting assets, curating clean training datasets, and fine-tuning Low-Rank Adaptation (LoRA) models across all visual domains:
- Photorealistic portraits and human likeness
- Anime, manga, and illustrative keyframes
- Stylized 3D character assets
- Vintage 1930s/1940s classic cel animation
- Sci-fi and fantasy concept environments
- Commercial product design and studio hero shots

The application implements a **Hardware-Adaptive Multi-Modal Split Pipeline**:
- **Hardware Advisor & Engine**: Probes GPU VRAM, compute capability, and RAM on startup, automatically assigning hardware tiers (Ultra, High, Mid, Entry, CPU) and recommending optimal batch sizes, gradient accumulation, and precision.
- **Supervisor Agent**: Central creative orchestrator coordinating specialist agents.
- **Visual Agent & Ollama Vision**: Auto-curation of high-resolution keyframe scans and descriptive captions with clean trigger words.
- **Aspect Ratio Bucketer**: Native aspect-ratio bucketing (`16:9`, `9:16`, `2:3`, `3:2`, `4:3`, `1:1`) to eliminate squeezing and stretching.
- **Story Agent**: Screenplay and storyboard panel generator formatted as structured JSON for animation and video generation engines.
- **Video Agent**: Motion bucketing, timeline scrubbing, and keyframe extraction.
- **Trainer Agent**: Automated dataset packaging for **Kohya_ss** and **ComfyUI** LoRA training across image (FLUX.1, SDXL, Qwen, Z-Image) and video models (Wan2.1, LTX-Video, MiniMax, Hunyuan).
- **Interactive Diagnostics Lab**: Evaluates real PyTorch safetensors weights with Frobenius norm telemetry and verified LoRA scale control (from 0.00 base model to 1.50).

---

## 2. System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            Geekatplay LoRA Maker Web GUI                          |
|   (HTML5 Timeline Scrubber | Keyframe Gallery | Supervisor Chat | Training | Lab) |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST / SSE
+------------------------------------------v----------------------------------------+
|                               FastAPI Application                                |
|  - Projects API (/api/projects)          - Captions API (/api/captions)           |
|  - Video API (/api/video)                - Datasets API (/api/datasets)           |
|  - Supervisor API (/api/supervisor)      - Training API (/api/training)           |
|  - Evaluation API (/api/evaluation)      - System & Hardware API (/api/system)   |
+------------------------------------------+----------------------------------------+
                                           |
    +--------------------------------------+------------------------------------+
    |                                      |                                    |
+---v--------------------+       +---------v----------+              +----------v-------+
|  Supervisor Agent      |       |  Specialist Agents |              | Backend Services |
|  (Orchestrator)        |       |                    |              |                  |
|  - Workflow Traffic    |       |  - VisualAgent     |              |  - HardwareServ. |
|  - Project Init        |       |  - StoryAgent      |              |  - VideoService  |
|  - Task Delegation     |       |  - VideoAgent      |              |  - VisionService |
|  - Quality Validation  |       |  - TrainerAgent    |              |  - DatasetService|
|  - Hardware Alignment  |       |                    |              |  - TrainingServ. |
|                        |       |                    |              |  - EvaluationS.  |
+------------------------+       +--------------------+              +------------------+
    |                                      |                                    |
    +--------------------------------------+------------------------------------+
                                           |
      +------------------------------------+--------------------------------+
      |                                                                     |
+-----v------------------------+                       +--------------------v-----+
|   Local Ollama Engine        |                       |   LoRA Training Engines  |
|   - qwen2.5vl:7b (Vision)    |                       |   - PyTorch CUDA Engine  |
|   - llama3.2-vision:latest   |                       |   - Kohya_ss sd-scripts  |
|   - qwen2.5-coder:14b (LLM)  |                       |   - ComfyUI Training     |
|                              |                       |   - FLUX / SDXL / Wan2.1 |
+------------------------------+                       +--------------------------+
```

---

## 3. Directory Layout Specification

Each studio project initialized by the system conforms strictly to the standard folder structure:

```
[Project_Root]/
├── project.json              # Project metadata, character token, trigger words, creation timestamp
├── Input_Video/              # Raw source video footage, digitized film clips, or image sets
├── Keyframes_Out/            # High-resolution extracted frames (e.g. Elena_00012.png)
│   ├── Elena_00012.png       # Lossless extracted image (aspect ratio preserved)
│   └── Elena_00012.txt       # Synced LoRA caption generated by Ollama Vision
├── Prompts/                  # Screenplays and animation scripts generated by Story Agent
│   └── animation_script.json # Structured JSON with timing cues and camera motion brushes
├── Assets/                   # Character model assets, turnaround sheets, ComfyUI dataset exports
│   └── ComfyUI_Dataset/      # Formatted images with metadata.jsonl
├── Training/                 # LoRA training configuration and output weights
│   ├── kohya_lora_config.toml# Auto-generated Kohya_ss training configuration
│   ├── comfyui_training_workflow.json # 1-click ComfyUI pipeline
│   ├── img/                  # Kohya repeat folders (e.g., 10_Elena character/)
│   └── output/               # Output LoRA weights (*.safetensors)
└── Output_Video/             # Final generated motion clips from Wan2.1 / LTX-Video
```

---

## 4. Multi-Agent Responsibilities

### 4.1 Supervisor Agent
- **Persona**: Knowledgeable, encouraging creative director.
- **Responsibilities**:
  1. Project Initialization: Guides folder layout, project presets, and trigger naming.
  2. Task Delegation: Delegates vision queries to Visual Agent, scripts to Story Agent, and training audits to Trainer Agent.
  3. Hardware-Aware Routing: Recommends optimal base model (FLUX.1 vs SDXL vs Wan2.1) according to user VRAM.

### 4.2 Visual Agent
- Dynamically queries local Ollama vision models (`qwen2.5vl:7b`, `llama3.2-vision`, `llava`).
- Enforces character trigger token presence (`[trigger_token], subject description...`).
- Analyzes candidate video poses: Portraits, turnarounds, dynamic action poses.

### 4.3 Story Agent
- Converts premise into standardized JSON screenplay format across genres.
- Generates precise timing cues (`00:00 - 00:03`), action descriptions, camera motion trajectories, and diffusion prompts.

### 4.4 Trainer Agent & Hardware Service
- Validates dataset readiness (checks frame count, verifies 100% caption coverage).
- Probes GPU VRAM and tunes batch size, gradient accumulation steps, optimizer (AdamW8bit vs AdamW), and mixed precision (`bf16`/`fp16`).
- Serializes architecture-valid `.safetensors` files for SDXL 1.0 (640/1280 dim UNet blocks) and FLUX.1 (3072 dim DiT blocks).
