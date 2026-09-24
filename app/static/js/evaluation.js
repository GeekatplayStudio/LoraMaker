// LoRA Model Evaluation, Diagnostics, Suggested Prompts, and Interactive Test Bench Controller

window.Evaluation = {
    selectedModel: null,
    currentPrompts: [],
    currentInspection: null,

    init() {
        // Event Listeners
        const modelSelect = document.getElementById("eval-model-select");
        if (modelSelect) {
            modelSelect.addEventListener("change", (e) => {
                this.selectedModel = e.target.value;
                this.inspect();
            });
        }

        const btnRefresh = document.getElementById("btn-eval-refresh");
        if (btnRefresh) {
            btnRefresh.addEventListener("click", () => this.load());
        }

        const btnDeploy = document.getElementById("btn-deploy-comfyui");
        if (btnDeploy) {
            btnDeploy.addEventListener("click", () => this.deployToComfyUI());
        }

        // LoRA Strength Slider with dynamic Base Model vs Active LoRA feedback
        const slider = document.getElementById("test-lora-scale");
        const sliderVal = document.getElementById("test-lora-scale-val");
        if (slider && sliderVal) {
            slider.addEventListener("input", (e) => {
                const val = parseFloat(e.target.value);
                if (val <= 0.001) {
                    sliderVal.textContent = "0.00 (Pure Base Model)";
                    sliderVal.style.color = "#38bdf8";
                } else if (val >= 0.99 && val <= 1.01) {
                    sliderVal.textContent = "1.00 (Full LoRA Effect)";
                    sliderVal.style.color = "#fbbf24";
                } else {
                    sliderVal.textContent = `${val.toFixed(2)} (LoRA Scale)`;
                    sliderVal.style.color = "var(--amber-primary)";
                }
            });
        }

        // Random Seed Toggle
        const btnRandomSeed = document.getElementById("btn-random-seed");
        const seedInput = document.getElementById("test-seed-input");
        if (btnRandomSeed && seedInput) {
            btnRandomSeed.addEventListener("click", () => {
                seedInput.value = Math.floor(Math.random() * 999999);
            });
        }

        // Render Test Cel Button
        const btnRender = document.getElementById("btn-render-test-cel");
        if (btnRender) {
            btnRender.addEventListener("click", () => this.renderTestSample());
        }

        // A/B Comparison Button (0.0 vs 1.0)
        const btnCompare = document.getElementById("btn-compare-ab");
        if (btnCompare) {
            btnCompare.addEventListener("click", () => this.renderCompareAB());
        }

        // Switch to Testing tab from Training
        const btnGotoTesting = document.getElementById("btn-goto-testing");
        if (btnGotoTesting) {
            btnGotoTesting.addEventListener("click", () => {
                if (typeof switchTab === "function") switchTab("evaluation");
            });
        }
    },

    async loadBaseCheckpoints() {
        const select = document.getElementById("test-base-ckpt-select");
        if (!select) return;

        try {
            const res = await fetch("/api/evaluation/base_checkpoints");
            const data = await res.json();
            if (data.checkpoints && data.checkpoints.length > 0) {
                select.innerHTML = data.checkpoints.map(c => `
                    <option value="${c.path}" ${c.name.includes("sd_xl_base_1.0") ? "selected" : ""}>
                        ${c.is_recommended ? '⭐ ' : ''}${c.name} (${c.size_gb} GB)
                    </option>
                `).join("");
            } else {
                select.innerHTML = `<option value="">sd_xl_base_1.0 (Default)</option>`;
            }
        } catch (e) {
            console.warn("Failed to load base checkpoints:", e);
            select.innerHTML = `<option value="">sd_xl_base_1.0 (Default)</option>`;
        }
    },

    async load() {
        if (!AppState.projectDir) return;

        try {
            await this.loadBaseCheckpoints();

            // 1. Fetch available models
            const res = await fetch(`/api/evaluation/models?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            const modelSelect = document.getElementById("eval-model-select");
            if (modelSelect) {
                modelSelect.innerHTML = "";
                if (data.models && data.models.length > 0) {
                    let firstGenuine = null;
                    data.models.forEach((m, idx) => {
                        const opt = document.createElement("option");
                        opt.value = m.filename;
                        const isGenuine = m.is_genuine || m.size_mb > 5.0;
                        if (isGenuine && !firstGenuine) firstGenuine = m.filename;

                        opt.textContent = isGenuine
                            ? `✅ ${m.filename} [Genuine LoRA - ${m.size_mb} MB]`
                            : `⚠️ ${m.filename} [Mock Stub - ${m.size_kb} KB]`;
                        
                        modelSelect.appendChild(opt);
                    });

                    // Auto-select genuine LoRA if present
                    if (firstGenuine) {
                        modelSelect.value = firstGenuine;
                        this.selectedModel = firstGenuine;
                    } else {
                        modelSelect.selectedIndex = 0;
                        this.selectedModel = data.models[0].filename;
                    }
                } else {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No trained models found in Training/output/";
                    modelSelect.appendChild(opt);
                    this.selectedModel = null;
                }
            }

            // 2. Run inspection, analytics, and prompts load
            await Promise.all([
                this.inspect(),
                this.loadAnalytics(),
                this.loadPrompts()
            ]);
        } catch (err) {
            console.error("Evaluation load error:", err);
            showToast("Failed to load evaluation data", "error");
        }
    },

    selectAndGoToBench(filename) {
        this.selectedModel = filename;
        const select = document.getElementById("eval-model-select");
        if (select) select.value = filename;

        if (typeof switchTab === "function") {
            switchTab("evaluation");
        }
        this.inspect();

        const bench = document.getElementById("test-bench-viewport");
        if (bench) bench.scrollIntoView({ behavior: "smooth", block: "center" });
        showToast(`Selected "${filename}" in Interactive Test Bench`, "success");
    },

    async inspect() {
        if (!AppState.projectDir) return;
        const modelParam = this.selectedModel ? `&model_filename=${encodeURIComponent(this.selectedModel)}` : "";

        try {
            const res = await fetch(`/api/evaluation/inspect?project_dir=${encodeURIComponent(AppState.projectDir)}${modelParam}`);
            const data = await res.json();
            this.currentInspection = data;

            // Update UI Stat Cards
            const elModelName = document.getElementById("eval-stat-model-name");
            const elRankAlpha = document.getElementById("eval-stat-rank-alpha");
            const elParams = document.getElementById("eval-stat-params");
            const elDeltaNorm = document.getElementById("eval-stat-delta-norm");
            const elHealthBadge = document.getElementById("eval-stat-health-badge");
            const elComfyStatus = document.getElementById("eval-comfy-status");

            if (elModelName) elModelName.textContent = data.model_name || "N/A";
            if (elRankAlpha) elRankAlpha.textContent = `Rank: ${data.rank || 16} | Alpha: ${data.alpha || 16} (Scale: ${data.scale || 1.0})`;
            if (elParams) elParams.textContent = `${(data.total_parameters || 0).toLocaleString()} weights`;
            if (elDeltaNorm) elDeltaNorm.textContent = `ΔW Norm: ${data.delta_weight_magnitude || 0.000000}`;
            
            if (elHealthBadge) {
                if (data.health_grade === "OPTIMAL") {
                    elHealthBadge.className = "eval-badge badge-optimal";
                    elHealthBadge.textContent = "✅ Active Gradient Updates";
                } else if (data.health_grade === "LOW_DELTA") {
                    elHealthBadge.className = "eval-badge badge-warning";
                    elHealthBadge.textContent = "⚠️ Low Delta Magnitude";
                } else {
                    elHealthBadge.className = "eval-badge badge-warning";
                    elHealthBadge.textContent = data.health_grade || "Pending Run";
                }
            }

            if (elComfyStatus) {
                if (data.comfyui_deployed) {
                    elComfyStatus.className = "eval-badge badge-optimal";
                    elComfyStatus.textContent = "Deployed in ComfyUI";
                } else {
                    elComfyStatus.className = "eval-badge";
                    elComfyStatus.style.background = "rgba(56, 189, 248, 0.15)";
                    elComfyStatus.style.color = "#38bdf8";
                    elComfyStatus.textContent = "ComfyUI Ready";
                }
            }

            // Populate Tensor Table
            const tbody = document.getElementById("eval-tensor-tbody");
            if (tbody && data.tensors) {
                tbody.innerHTML = "";
                data.tensors.forEach(t => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td class="mono" style="color: #fde68a;">${t.key}</td>
                        <td class="mono">[${t.shape.join(", ")}]</td>
                        <td>${t.dtype}</td>
                        <td class="mono">${t.parameters.toLocaleString()}</td>
                        <td class="mono" style="color: var(--cyan-accent);">${t.norm}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (err) {
            console.error("Inspect error:", err);
        }
    },

    async loadAnalytics() {
        if (!AppState.projectDir) return;
        const modelParam = this.selectedModel ? `&model_filename=${encodeURIComponent(this.selectedModel)}` : "";

        try {
            const res = await fetch(`/api/evaluation/analytics?project_dir=${encodeURIComponent(AppState.projectDir)}${modelParam}`);
            const data = await res.json();

            const elLossRed = document.getElementById("eval-stat-loss-reduction");
            const elLossCurve = document.getElementById("eval-stat-loss-curve");
            const elHardware = document.getElementById("eval-stat-hardware");

            if (elLossRed) elLossRed.textContent = `-${data.loss_reduction_percent || 0}%`;
            if (elLossCurve) elLossCurve.textContent = `${data.initial_loss} ➔ ${data.final_loss} (${data.epochs || 10} epochs)`;
            if (elHardware && data.hardware) {
                elHardware.textContent = `${data.hardware.device} (${data.hardware.vram_total_gb} GB VRAM)`;
            }
        } catch (err) {
            console.error("Analytics error:", err);
        }
    },

    async loadPrompts() {
        if (!AppState.projectDir) return;

        try {
            const res = await fetch(`/api/evaluation/prompts?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();
            this.currentPrompts = data.prompts || [];

            const container = document.getElementById("suggested-prompts-list");
            if (!container) return;

            container.innerHTML = "";

            this.currentPrompts.forEach((p, idx) => {
                const card = document.createElement("div");
                card.className = "prompt-card";

                card.innerHTML = `
                    <div class="prompt-card-header">
                        <div class="prompt-category-title">
                            <span>${p.badge || '✨'}</span>
                            <span>${p.category}</span>
                        </div>
                        <span class="prompt-specs-meta mono">Scale: ${p.recommended_lora_scale} | CFG: ${p.cfg}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: var(--text-dim);">${p.description}</div>
                    <div class="prompt-text-box mono">${p.positive}</div>
                    <div class="prompt-card-actions">
                        <button class="btn-secondary" style="padding: 6px 12px; font-size: 0.78rem;" onclick="window.Evaluation.copyPrompt('${idx}')">
                            📋 Copy Prompt
                        </button>
                        <button class="btn-primary" style="padding: 6px 14px; font-size: 0.78rem;" onclick="window.Evaluation.sendToBench('${idx}')">
                            🧪 Send to Test Bench ➔
                        </button>
                    </div>
                `;
                container.appendChild(card);
            });

            // Pre-fill the test bench with the first prompt if empty
            const testPromptInput = document.getElementById("test-prompt-input");
            if (testPromptInput && !testPromptInput.value && this.currentPrompts.length > 0) {
                testPromptInput.value = this.currentPrompts[0].positive;
            }
        } catch (err) {
            console.error("Prompts load error:", err);
        }
    },

    copyPrompt(promptIdx) {
        const p = this.currentPrompts[promptIdx];
        if (!p) return;

        navigator.clipboard.writeText(p.positive).then(() => {
            showToast(`Copied prompt: ${p.category}`, "success");
        }).catch(() => {
            // Fallback
            const ta = document.createElement("textarea");
            ta.value = p.positive;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            showToast(`Copied prompt: ${p.category}`, "success");
        });
    },

    sendToBench(promptIdx) {
        const p = this.currentPrompts[promptIdx];
        if (!p) return;

        const testPromptInput = document.getElementById("test-prompt-input");
        const testNegInput = document.getElementById("test-neg-prompt-input");
        const slider = document.getElementById("test-lora-scale");
        const sliderVal = document.getElementById("test-lora-scale-val");

        if (testPromptInput) testPromptInput.value = p.positive;
        if (testNegInput && p.negative) testNegInput.value = p.negative;
        if (slider && p.recommended_lora_scale) {
            slider.value = p.recommended_lora_scale;
            if (sliderVal) sliderVal.textContent = parseFloat(p.recommended_lora_scale).toFixed(2);
        }

        const aspectSelect = document.getElementById("test-aspect-ratio");
        if (aspectSelect) {
            if (p.id === "video" || p.id === "scene") {
                aspectSelect.value = "16:9";
            } else if (p.id === "action") {
                aspectSelect.value = "3:2";
            } else {
                aspectSelect.value = "1:1";
            }
        }

        showToast(`Sent "${p.badge}" to Interactive Test Bench!`, "success");

        // Smooth scroll to test bench viewport
        const bench = document.getElementById("test-bench-viewport");
        if (bench) bench.scrollIntoView({ behavior: "smooth", block: "center" });
    },

    async renderTestSample() {
        if (!AppState.projectDir) {
            showToast("No active project loaded", "error");
            return;
        }

        const promptInput = document.getElementById("test-prompt-input");
        const negPromptInput = document.getElementById("test-neg-prompt-input");
        const scaleInput = document.getElementById("test-lora-scale");
        const seedInput = document.getElementById("test-seed-input");
        const aspectInput = document.getElementById("test-aspect-ratio");
        const framingInput = document.getElementById("test-framing-mode");
        const btnRender = document.getElementById("btn-render-test-cel");
        const viewport = document.getElementById("test-bench-viewport");

        const prompt = promptInput ? promptInput.value.trim() : "";
        if (!prompt) {
            showToast("Please enter or select a test prompt", "error");
            return;
        }

        const loraScale = scaleInput ? parseFloat(scaleInput.value) : 0.85;
        const seed = seedInput ? parseInt(seedInput.value, 10) : 42;
        const negPrompt = negPromptInput ? negPromptInput.value.trim() : "";
        const aspectRatio = aspectInput ? aspectInput.value : "1:1";
        const framingMode = framingInput ? framingInput.value : "pad";

        const baseCkptInput = document.getElementById("test-base-ckpt-select");
        const baseCkpt = baseCkptInput ? baseCkptInput.value : "";

        if (btnRender) {
            btnRender.disabled = true;
            btnRender.textContent = `⏳ Synthesizing LoRA Test Cel [${aspectRatio}]...`;
        }

        try {
            const res = await fetch("/api/evaluation/test_render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    prompt: prompt,
                    negative_prompt: negPrompt,
                    lora_scale: loraScale,
                    seed: seed,
                    steps: 30,
                    model_filename: this.selectedModel,
                    base_checkpoint: baseCkpt,
                    aspect_ratio: aspectRatio,
                    framing_mode: framingMode
                })
            });

            const data = await res.json();

            if (data.success && data.image_base64) {
                if (viewport) {
                    const loraBadgeText = data.lora_applied ? `🔥 LoRA (${data.lora_scale})` : `🎯 Pure Base Model`;
                    const loraBadgeColor = data.lora_applied ? '#fbbf24' : '#38bdf8';
                    const genuineTag = data.is_genuine_lora ? ` [Genuine ${data.lora_size_mb} MB]` : '';

                    viewport.innerHTML = `
                        <img src="${data.image_base64}" alt="LoRA Test Cel" id="test-cel-img" />
                        <div style="position: absolute; bottom: 8px; right: 12px; background: rgba(10, 12, 16, 0.88); padding: 5px 10px; border-radius: 4px; font-size: 0.72rem; color: #fde68a; display: flex; gap: 8px; align-items: center;" class="mono">
                            <span>⚡ ${data.render_time_seconds}s</span>
                            <span>|</span>
                            <span>${data.resolution} [${aspectRatio}]</span>
                            <span>|</span>
                            <span>${data.base_checkpoint || 'SDXL Base'}</span>
                            <span>|</span>
                            <span style="color: ${loraBadgeColor}; font-weight: 600;">${loraBadgeText}${genuineTag}</span>
                        </div>
                    `;
                }

                const dlBtn = document.getElementById("btn-download-test-cel");
                if (dlBtn) {
                    dlBtn.style.display = "inline-flex";
                    dlBtn.onclick = () => {
                        const a = document.createElement("a");
                        a.href = data.image_base64;
                        a.download = data.filename || `lora_test_${aspectRatio.replace(':', 'x')}.png`;
                        a.click();
                    };
                }

                showToast(`Test Cel frame synthesized in ${data.render_time_seconds}s!`, "success");
            } else {
                const errMsg = data.error || data.detail || "Unknown error";
                showToast("Render failed: " + errMsg, "error");
                if (viewport) {
                    viewport.innerHTML = `
                        <div style="padding: 24px; text-align: center; color: var(--rose-accent);">
                            <div style="font-size: 2rem; margin-bottom: 8px;">⚠️</div>
                            <h3>Inference Diagnostics Notice</h3>
                            <p style="font-size: 0.85rem; color: #fde68a; max-width: 500px; margin: 8px auto;">${errMsg}</p>
                            <p style="font-size: 0.76rem; color: var(--text-dim); margin-top: 12px;">Tip: Select a genuine LoRA file (e.g. 'test_real_kohya_fitted.safetensors', ~96.8 MB) or set LoRA Weight to 0.00 for pure base model generation.</p>
                        </div>
                    `;
                }
            }
        } catch (err) {
            console.error("Test render error:", err);
            showToast("Test render request failed: " + err.message, "error");
        } finally {
            if (btnRender) {
                btnRender.disabled = false;
                btnRender.textContent = "🎨 Render Test Cel Frame";
            }
        }
    },

    async renderCompareAB() {
        if (!AppState.projectDir) {
            showToast("No active project loaded", "error");
            return;
        }

        const promptInput = document.getElementById("test-prompt-input");
        const negPromptInput = document.getElementById("test-neg-prompt-input");
        const seedInput = document.getElementById("test-seed-input");
        const aspectInput = document.getElementById("test-aspect-ratio");
        const framingInput = document.getElementById("test-framing-mode");
        const baseCkptInput = document.getElementById("test-base-ckpt-select");
        const btnCompare = document.getElementById("btn-compare-ab");
        const viewport = document.getElementById("test-bench-viewport");

        const prompt = promptInput ? promptInput.value.trim() : "";
        if (!prompt) {
            showToast("Please enter a prompt to compare", "error");
            return;
        }

        const seed = seedInput ? parseInt(seedInput.value, 10) : 42;
        const negPrompt = negPromptInput ? negPromptInput.value.trim() : "";
        const aspectRatio = aspectInput ? aspectInput.value : "1:1";
        const framingMode = framingInput ? framingInput.value : "pad";
        const baseCkpt = baseCkptInput ? baseCkptInput.value : "";

        if (btnCompare) {
            btnCompare.disabled = true;
            btnCompare.textContent = "⏳ Running A/B Test (0.0 vs 1.0)...";
        }

        if (viewport) {
            viewport.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-dim); gap: 10px; min-height: 250px;">
                    <div class="pulse-dot" style="width: 20px; height: 20px; background: var(--amber-primary);"></div>
                    <div style="font-weight: 600; color: #fde68a;">Executing Dual Latent Diffusion Passes on CUDA...</div>
                    <div style="font-size: 0.8rem; color: var(--text-dim);">Pass 1: Pure SDXL Base (0.00) ➔ Pass 2: Full LoRA Style (1.00)</div>
                </div>
            `;
        }

        try {
            // Pass 1: Weight 0.00 (Pure Base Model)
            const res0 = await fetch("/api/evaluation/test_render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    prompt: prompt,
                    negative_prompt: negPrompt,
                    lora_scale: 0.0,
                    seed: seed,
                    steps: 25,
                    model_filename: this.selectedModel,
                    base_checkpoint: baseCkpt,
                    aspect_ratio: aspectRatio,
                    framing_mode: framingMode
                })
            });
            const data0 = await res0.json();

            // Pass 2: Weight 1.00 (Full LoRA Effect)
            const res1 = await fetch("/api/evaluation/test_render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    prompt: prompt,
                    negative_prompt: negPrompt,
                    lora_scale: 1.0,
                    seed: seed,
                    steps: 25,
                    model_filename: this.selectedModel,
                    base_checkpoint: baseCkpt,
                    aspect_ratio: aspectRatio,
                    framing_mode: framingMode
                })
            });
            const data1 = await res1.json();

            if (data0.success && data1.success) {
                if (viewport) {
                    viewport.innerHTML = `
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%; height: 100%; padding: 8px;">
                            <!-- Left: 0.0 Base Model -->
                            <div style="display: flex; flex-direction: column; background: rgba(10, 12, 16, 0.85); border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);">
                                <div style="padding: 6px 12px; background: rgba(56, 189, 248, 0.15); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color);">
                                    <span style="font-weight: 700; color: #38bdf8; font-size: 0.8rem;">🟢 PURE BASE MODEL (0.00 LoRA)</span>
                                    <span class="mono" style="font-size: 0.7rem; color: var(--text-dim);">${data0.render_time_seconds}s</span>
                                </div>
                                <div style="flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 6px;">
                                    <img src="${data0.image_base64}" alt="Base Model Output" style="max-width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px;" />
                                </div>
                                <div class="mono" style="padding: 4px 8px; font-size: 0.7rem; color: var(--text-dim); background: rgba(0,0,0,0.5); text-align: center;">
                                    ${data0.base_checkpoint || 'SDXL Base 1.0'} | Baseline Prompt
                                </div>
                            </div>

                            <!-- Right: 1.0 LoRA Model -->
                            <div style="display: flex; flex-direction: column; background: rgba(10, 12, 16, 0.85); border-radius: 8px; overflow: hidden; border: 1px solid var(--amber-primary);">
                                <div style="padding: 6px 12px; background: rgba(245, 158, 11, 0.2); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color);">
                                    <span style="font-weight: 700; color: #fbbf24; font-size: 0.8rem;">🟣 100% LORA ADAPTED (1.00 Weight)</span>
                                    <span class="mono" style="font-size: 0.7rem; color: var(--text-dim);">${data1.render_time_seconds}s</span>
                                </div>
                                <div style="flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 6px;">
                                    <img src="${data1.image_base64}" alt="LoRA Adapted Output" style="max-width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px;" />
                                </div>
                                <div class="mono" style="padding: 4px 8px; font-size: 0.7rem; color: var(--text-dim); background: rgba(0,0,0,0.5); text-align: center;">
                                    LoRA: ${this.selectedModel || 'Active'} | ${data1.lora_size_mb || 0} MB
                                </div>
                            </div>
                        </div>
                    `;
                }
                showToast("A/B Comparison Render Complete!", "success");
            } else {
                throw new Error(data0.error || data1.error || "Comparison render failed");
            }
        } catch (e) {
            console.error("A/B compare error:", e);
            showToast("Comparison error: " + e.message, "error");
            if (viewport) {
                viewport.innerHTML = `
                    <div style="padding: 24px; text-align: center; color: var(--rose-accent);">
                        <h3>⚠️ A/B Comparison Issue</h3>
                        <p style="font-size: 0.85rem; color: #fde68a; margin-top: 8px;">${e.message}</p>
                    </div>
                `;
            }
        } finally {
            if (btnCompare) {
                btnCompare.disabled = false;
                btnCompare.textContent = "⚖️ Compare 0.0 vs 1.0";
            }
        }
    },

    async deployToComfyUI() {
        if (!AppState.projectDir) return;
        const btnDeploy = document.getElementById("btn-deploy-comfyui");
        if (btnDeploy) {
            btnDeploy.disabled = true;
            btnDeploy.textContent = "🚀 Deploying...";
        }

        try {
            const res = await fetch("/api/evaluation/deploy_comfyui", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    model_filename: this.selectedModel
                })
            });

            const data = await res.json();
            if (data.success) {
                showToast(`Deployed to ComfyUI! (${data.file_size_kb} KB)`, "success");
                const elComfyStatus = document.getElementById("eval-comfy-status");
                if (elComfyStatus) {
                    elComfyStatus.className = "eval-badge badge-optimal";
                    elComfyStatus.textContent = "Deployed in ComfyUI";
                }
            } else {
                showToast("Deploy failed: " + (data.error || "Error"), "error");
            }
        } catch (err) {
            console.error("Deploy error:", err);
            showToast("Deployment request failed", "error");
        } finally {
            if (btnDeploy) {
                btnDeploy.disabled = false;
                btnDeploy.textContent = "🚀 Deploy to ComfyUI";
            }
        }
    }
};

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    window.Evaluation.init();
});
