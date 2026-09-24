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
                    sliderVal.textContent = "0.00 (Base Model Only)";
                    sliderVal.style.color = "#94a3b8";
                } else {
                    sliderVal.textContent = `${val.toFixed(2)}`;
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

        // Switch to Testing tab from Training
        const btnGotoTesting = document.getElementById("btn-goto-testing");
        if (btnGotoTesting) {
            btnGotoTesting.addEventListener("click", () => {
                if (typeof switchTab === "function") switchTab("evaluation");
            });
        }
    },

    async load() {
        if (!AppState.projectDir) return;

        try {
            // 1. Fetch available models
            const res = await fetch(`/api/evaluation/models?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            const modelSelect = document.getElementById("eval-model-select");
            if (modelSelect) {
                modelSelect.innerHTML = "";
                if (data.models && data.models.length > 0) {
                    data.models.forEach((m, idx) => {
                        const opt = document.createElement("option");
                        opt.value = m.filename;
                        opt.textContent = `${m.filename} (${m.size_kb} KB)${m.is_empty ? ' [Stub]' : ''}`;
                        if (idx === 0) opt.selected = true;
                        modelSelect.appendChild(opt);
                    });
                    this.selectedModel = data.models[0].filename;
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
                    aspect_ratio: aspectRatio,
                    framing_mode: framingMode
                })
            });

            const data = await res.json();

            if (data.success && data.image_base64) {
                if (viewport) {
                    const loraBadgeText = data.lora_applied ? `🔥 LoRA (${data.lora_scale})` : `🎯 Pure Base Model`;
                    const loraBadgeColor = data.lora_applied ? '#fbbf24' : '#38bdf8';
                    viewport.innerHTML = `
                        <img src="${data.image_base64}" alt="LoRA Test Cel" id="test-cel-img" />
                        <div style="position: absolute; bottom: 8px; right: 12px; background: rgba(10, 12, 16, 0.88); padding: 5px 10px; border-radius: 4px; font-size: 0.72rem; color: #fde68a; display: flex; gap: 8px; align-items: center;" class="mono">
                            <span>⚡ ${data.render_time_seconds}s</span>
                            <span>|</span>
                            <span>${data.resolution} [${aspectRatio}]</span>
                            <span>|</span>
                            <span style="color: ${loraBadgeColor}; font-weight: 600;">${loraBadgeText}</span>
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
                showToast("Render failed: " + (data.error || "Unknown error"), "error");
            }
        } catch (err) {
            console.error("Test render error:", err);
            showToast("Test render request failed", "error");
        } finally {
            if (btnRender) {
                btnRender.disabled = false;
                btnRender.textContent = "🎨 Render Test Cel Frame";
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
