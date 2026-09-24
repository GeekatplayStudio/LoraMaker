# LoRA Training Specialist Guide: Image & Video Diffusion

## 1. Overview

This guide provides deep technical recommendations for training Low-Rank Adaptation (LoRA) weights on vintage animation assets using **Retro Diffusion Suite**. It covers both **Image Diffusion Models** (FLUX.1 [dev], SDXL 1.0) and **Video Diffusion Models** (Wan2.1, LTX-Video).

---

## 2. Dataset Preparation Best Practices

### 2.1 Keyframe Selection Principles
1. **Pose Diversity**:
   - Minimum 10 to 25 cel scans for character identity.
   - Required poses: Neutral Frontal, 3/4 Turnaround, Profile View, Back View, Running Action, Squash & Stretch Expression.
2. **Background Separation**:
   - If training character identity: use diverse backgrounds or clean cutout cel scans so the model learns the character, not the background scenery.
   - If training artistic style: keep painted gouache backgrounds and vintage film grain intact.
3. **Resolution Bucketing**:
   - FLUX.1 [dev]: 1024x1024 (aspect ratio bucketing enabled).
   - SDXL 1.0: 1024x1024.
   - Wan2.1 Video: 720p (e.g. 1280x720 or 960x544).
   - LTX-Video: 768x512.

---

## 3. Captioning Strategy (Ollama Multimodal)

Retro Diffusion Suite automatically prepends the unique trigger word and synchronizes the `.txt` file:

```
[trigger_token], [style_tokens], [pose], [facial expression], [attire], [artistic characteristics]
```

### Example Caption:
```text
BendyBot, vintage 1930s rubber hose cel animation, monochrome ink lines, pie-eyed cartoon robot character smiling joyfully, raising metallic left glove in greeting, oversized shoes, solid black ink fill, white cel highlights, vintage film grain texture, high contrast fleischer studio scan
```

---

## 4. Hyperparameter Matrix by Architecture

| Architecture | Category | Recommended Rank ($r$) | Alpha ($\alpha$) | Learning Rate | Batch Size | Epochs | Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.1 [dev]** | Image | 16 | 16 | $1 \times 10^{-4}$ | 1 | 8 - 12 | AdamW8bit |
| **Qwen Image (DiT)** | Image | 16 | 16 | $1 \times 10^{-4}$ | 1 | 10 | AdamW8bit |
| **Z-Image DiT** | Image | 32 | 16 | $1 \times 10^{-4}$ | 1 | 12 | AdamW8bit |
| **SDXL 1.0** | Image | 32 | 16 | $1 \times 10^{-4}$ | 1 - 2 | 10 - 15 | AdamW8bit |
| **Wan2.1** | Video | 32 | 32 | $8 \times 10^{-5}$ | 1 | 6 - 10 | AdamW8bit |
| **LTX-Video** | Video | 16 | 16 | $1 \times 10^{-4}$ | 1 | 8 - 10 | AdamW8bit |
| **MiniMax / Hunyuan**| Video | 32 | 32 | $8 \times 10^{-5}$ | 1 | 8 | AdamW8bit |

---

## 5. Kohya_ss `sd-scripts` Pipeline Integration

Retro Diffusion Suite exports your project dataset into the exact folder structure expected by Kohya:

```
Training/
  img/
    10_BendyBot character/
      BendyBot_00001.png
      BendyBot_00001.txt
```

And automatically generates `kohya_lora_config.toml`. To execute via Kohya CLI:

```bash
# In Kohya_ss environment:
accelerate launch --num_cpu_threads_per_process=2 "./sd-scripts/flux_train_network.py" \
  --config_file="D:/Projects/LoraAssetsMaker/projects/Bendy_1930s_Steamboat/Training/kohya_lora_config.toml"
```

---

## 6. ComfyUI Pipeline Integration

The app generates `Assets/ComfyUI_Dataset/metadata.jsonl` and `Training/comfyui_training_workflow.json`.

In ComfyUI:
1. Open ComfyUI web interface.
2. Drag and drop `Training/comfyui_training_workflow.json` into the canvas.
3. Queue Prompt to begin fine-tuning directly inside ComfyUI.
4. Generated LoRA weights (`*.safetensors`) will be placed directly in `Training/output/`.
