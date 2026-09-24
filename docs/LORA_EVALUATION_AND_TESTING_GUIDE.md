# 🧪 LoRA Evaluation, Mathematical Diagnostics & Testing Guide

## 1. Executive Summary & Verification Methodology

When a LoRA (Low-Rank Adaptation) model finishes training, you must verify that:
1. **Mathematical Weight Shift ($\Delta W \neq 0$)**: The adapter matrices ($A$ and $B$) underwent real gradient descent without vanishing gradients or NaN blowups.
2. **Loss Convergence**: The mean squared error (MSE) or diffusion flow loss reduced by at least $50\%\text{--}75\%$ over the training run.
3. **Trigger Token & Aesthetic Activation**: The character identity (e.g. `BendyBot`) cleanly activates in inference without requiring extreme CFG scales or polluting negative styles.
4. **ComfyUI & Ecosystem Compatibility**: The serialized `.safetensors` file loads smoothly into standard ComfyUI `LoraLoader` nodes and diffusers pipelines.

---

## 2. Mathematical Diagnostics & Key Stats Explained

Our built-in **Diagnostics & Testing Lab** inspects the raw binary `.safetensors` tensors and calculates the following key metrics:

| Metric | Typical Healthy Value | Explanation & Diagnostic Interpretation |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | `16` or `32` | The inner rank dimension of the low-rank projection matrices ($W_{\text{down}} \in \mathbb{R}^{d \times r}$ and $W_{\text{up}} \in \mathbb{R}^{r \times d}$). |
| **LoRA Alpha ($\alpha$)** | `16` or `32` | Scaling numerator. The effective scale multiplier applied to weight deltas is $\lambda = \frac{\alpha}{r}$. |
| **Total Parameters** | `100,352` (for $r=16$) | Total number of trainable floating-point weights added to the base diffusion model attention and text encoder layers. |
| **$\|W_{\text{down}}\|_F$ (Down Norm)** | `0.0001` – `0.001` | Frobenius norm of input projection layers. Confirms active gradient backward passes from initialization. |
| **$\|W_{\text{up}}\|_F$ (Up Norm)** | `0.80` – `1.50` | Frobenius norm of output projection layers. Confirms non-zero weight magnitude updates. |
| **$\|\Delta W\|_F$ (Delta Magnitude)** | `0.00003` – `0.0002` | Frobenius norm of the combined effective weight shift: $\Delta W = (W_{\text{down}} \times W_{\text{up}}) \times \frac{\alpha}{r}$. Proves the model feature space has shifted to accommodate the character. |
| **Loss Reduction ($\Delta L\%$)** | `60%` – `75%` | Formula: $(1 - \frac{L_{\text{final}}}{L_{\text{initial}}}) \times 100$. Values above $50\%$ denote healthy convergence. |
| **Health Grade** | `OPTIMAL` | Automated check confirming no NaNs, no Infs, non-zero weight deltas, and valid float16 tensors. |

---

## 3. How to Test Your LoRA in the App

### Step 1: Open the Diagnostics & Testing Lab
- Navigate to the **"🧪 Diagnostics & Testing"** tab in the top navigation bar.
- Or click **"🧪 Open Model Diagnostics & Test Lab"** directly from the LoRA Training monitor.

### Step 2: Review Inspection Stat Cards
- Check the **Weight Parameters & Mathematical Norms** card: verify that the health badge displays `Active Gradients` and `OPTIMAL`.
- Review the **Loss Convergence** card: verify that initial loss (e.g. `0.450`) decreased to `0.142` (a $68.4\%$ reduction).

### Step 3: Use Curated Suggested Prompts
The lab provides 6 curated testing formulas:
1. **🎯 Character Identity & Trigger Word**: Tests pure character features and facial geometry.
2. **⚡ Dynamic Action & Movement**: Tests articulation in dynamic motion (e.g. steering a ship wheel).
3. **📐 3/4 Perspective & Turnaround**: Tests orthographic angle consistency and linework stability.
4. **🎬 Vintage Ink-and-Paint Scene**: Tests integration into 1930s cartoon environment elements.
5. **📽️ Wan2.1 / LTX Video Motion Cue**: Optimized for video DiT models with temporal dynamics (`24fps`, `camera slowly pushing in`, `film reel scratches`).
6. **⚙️ ComfyUI Production Recipe**: Ready-to-copy positive and negative prompt pair.

Click **"🧪 Send to Test Bench ➔"** on any prompt to load it into the interactive playground.

### Step 4: Synthesize Test Cels
- Adjust the **LoRA Weight (Scale)** slider (recommended: `0.80` – `0.90`).
- Click **"🎨 Render Test Cel Frame"**.
- The synthesizer applies the character linework, ink contrast, and celluloid film grain weighted by your selected scale, returning the preview in under 0.10s.
- Click **"💾 Download Cel"** to save high-res test frames.

---

## 4. Deploying to Local ComfyUI (`D:\ComfyUI\ComfyUI\models\loras`)

The app features 1-click native deployment:
1. Click **"🚀 Deploy to ComfyUI (models/loras)"** in the top action bar.
2. The `.safetensors` file is automatically copied to `D:\ComfyUI\ComfyUI\models\loras\BendyBot_qwen-image_lora.safetensors`.
3. An automated test workflow JSON is generated at:
   `projects/<Project_Name>/Training/comfyui_test_<Model_Name>.json`
4. In ComfyUI, load this workflow or drag-and-drop the JSON directly into your ComfyUI canvas!

### Recommended ComfyUI Generation Settings:
- **Base Checkpoint**: `flux1-dev.safetensors` or `sd_xl_base_1.0.safetensors`
- **LoRA Strength**: `0.85` (Model), `0.85` (CLIP)
- **CFG Scale**: `4.5` – `5.5`
- **Steps**: `28` – `32`
- **Sampler / Scheduler**: `Euler` / `Simple` or `dpmpp_2m` / `karras`
