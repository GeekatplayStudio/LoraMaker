// LoRA Training Pipeline Controller
const Training = {
    selectedModel: "flux-1-dev",
    pollInterval: null,

    init() {
        // Model card selection
        document.querySelectorAll(".model-card").forEach(card => {
            card.addEventListener("click", () => {
                document.querySelectorAll(".model-card").forEach(c => c.classList.remove("selected"));
                card.classList.add("selected");
                this.selectedModel = card.getAttribute("data-model-id");
                this.loadModelProfile(this.selectedModel);
            });
        });

        // Model category filters (All, Images, Video, Turbo)
        document.querySelectorAll(".model-filters button").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".model-filters button").forEach(b => {
                    b.classList.remove("active-filter");
                    b.style.borderColor = "var(--border-color)";
                    b.style.background = "rgba(255, 255, 255, 0.05)";
                });
                btn.classList.add("active-filter");
                btn.style.borderColor = "var(--amber-primary)";
                btn.style.background = "rgba(245, 158, 11, 0.2)";

                const filter = btn.getAttribute("data-filter");
                document.querySelectorAll(".model-card").forEach(card => {
                    const cat = card.getAttribute("data-category");
                    const isTurbo = card.getAttribute("data-turbo") === "true";

                    if (filter === "all") {
                        card.style.display = "block";
                    } else if (filter === "image") {
                        card.style.display = cat === "image" ? "block" : "none";
                    } else if (filter === "video") {
                        card.style.display = cat === "video" ? "block" : "none";
                    } else if (filter === "turbo") {
                        card.style.display = isTurbo ? "block" : "none";
                    }
                });
            });
        });

        document.getElementById("btn-start-training")?.addEventListener("click", () => this.startTraining());
        document.getElementById("btn-audit-dataset")?.addEventListener("click", () => this.audit());
    },

    async loadModelProfile(modelId) {
        try {
            const res = await fetch(`/api/training/profile?base_model_id=${encodeURIComponent(modelId)}`);
            const data = await res.json();
            const params = data.recommended_parameters;

            if (params) {
                document.getElementById("train-rank-input").value = params.rank;
                document.getElementById("train-alpha-input").value = params.alpha;
                document.getElementById("train-lr-input").value = params.lr;
                document.getElementById("train-epochs-input").value = params.epochs;
            }
        } catch (e) {
            console.warn("Could not load model profile:", e);
        }
    },

    async audit() {
        if (!AppState.projectDir) return;
        const auditBox = document.getElementById("audit-results-box");
        if (!auditBox) return;

        try {
            const res = await fetch(`/api/training/audit?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            let html = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong>Dataset Readiness Audit:</strong>
                    <span style="font-weight: 700; color: ${data.is_ready_for_training ? 'var(--emerald-accent)' : 'var(--amber-primary)'};">
                        ${data.is_ready_for_training ? '✅ READY TO TRAIN' : '⏳ NEEDS MORE CEL FRAMES'}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); display: flex; flex-wrap: wrap; gap: 10px;">
                    <span>Frames: <strong>${data.total_frames}</strong></span>
                    <span>Captioned: <strong>${data.captioned_frames}</strong></span>
                    <span>Pending: <strong>${data.uncaptioned_frames}</strong></span>
                    <span style="color: #38bdf8;">Format: <strong>${data.aspect_ratio || '1:1'} (${data.native_resolution ? data.native_resolution.join('×') : '512×512'})</strong></span>
                </div>
            `;

            if (data.issues && data.issues.length > 0) {
                html += '<ul style="margin-top: 8px; padding-left: 20px; font-size: 0.8rem; color: var(--amber-primary);">';
                data.issues.forEach(iss => html += `<li>${iss}</li>`);
                html += '</ul>';
            }

            if (data.recommendations && data.recommendations.length > 0) {
                html += '<ul style="margin-top: 6px; padding-left: 20px; font-size: 0.8rem; color: #38bdf8;">';
                data.recommendations.forEach(rec => html += `<li>${rec}</li>`);
                html += '</ul>';
            }

            auditBox.innerHTML = html;
        } catch (e) {
            auditBox.innerHTML = `<span style="color: var(--rose-accent);">Audit error: ${e.message}</span>`;
        }
    },

    async startTraining() {
        if (!AppState.projectDir) {
            showToast("Please select an active project first!", "error");
            return;
        }

        const rank = parseInt(document.getElementById("train-rank-input").value) || 16;
        const alpha = parseInt(document.getElementById("train-alpha-input").value) || 16;
        const epochs = parseInt(document.getElementById("train-epochs-input").value) || 5;
        const repeats = parseInt(document.getElementById("train-repeats-input").value) || 10;
        const batchSize = parseInt(document.getElementById("train-batch-input").value) || 1;
        const lr = parseFloat(document.getElementById("train-lr-input").value) || 0.0001;
        const execMode = document.getElementById("train-exec-mode")?.value || "real_gpu";
        const framingMode = document.getElementById("train-framing-mode")?.value || "bucket";

        const btn = document.getElementById("btn-start-training");
        btn.disabled = true;
        btn.innerHTML = execMode === "real_gpu" ? "<span>🔥 GPU Training Active...</span>" : "<span>⚡ Dry-Run Active...</span>";

        showToast(`Starting LoRA pipeline for ${this.selectedModel} (${execMode}, ${framingMode})...`, "info");

        try {
            const res = await fetch("/api/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    base_model: this.selectedModel,
                    lora_rank: rank,
                    lora_alpha: alpha,
                    epochs: epochs,
                    repeats: repeats,
                    batch_size: batchSize,
                    learning_rate: lr,
                    execution_mode: execMode,
                    framing_mode: framingMode
                })
            });

            const data = await res.json();
            if (res.ok) {
                showToast("LoRA training pipeline launched!", "success");
                this.startPollingStatus();
            } else {
                throw new Error(data.detail || "Training startup failed");
            }
        } catch (e) {
            showToast("Training launch failed: " + e.message, "error");
            btn.disabled = false;
            btn.innerHTML = "<span>🚀 Launch LoRA Training Run</span>";
        }
    },

    startPollingStatus() {
        if (this.pollInterval) clearInterval(this.pollInterval);

        const logViewer = document.getElementById("training-log-viewer");
        const progressBar = document.getElementById("training-progress-fill");
        const statusText = document.getElementById("training-status-text");
        const lossText = document.getElementById("training-loss-text");

        this.pollInterval = setInterval(async () => {
            if (!AppState.projectDir) return;

            try {
                const res = await fetch(`/api/training/status?project_dir=${encodeURIComponent(AppState.projectDir)}`);
                const data = await res.json();

                if (progressBar) progressBar.style.width = `${data.progress_percent}%`;
                if (statusText) statusText.textContent = `Status: ${data.status.toUpperCase()} (${data.current_step}/${data.total_steps})`;
                if (lossText) lossText.textContent = `Current Loss: ${data.current_loss}`;

                if (logViewer && data.log) {
                    logViewer.innerHTML = data.log.map(l => `<div>› ${l}</div>`).join("");
                    logViewer.scrollTop = logViewer.scrollHeight;
                }

                if (data.status === "completed" || data.status === "failed") {
                    clearInterval(this.pollInterval);
                    const btn = document.getElementById("btn-start-training");
                    if (btn) {
                        btn.disabled = false;
                        btn.innerHTML = "<span>🚀 Launch LoRA Training Run</span>";
                    }
                    if (data.status === "completed") {
                        showToast("LoRA Training Run Completed! Model weights ready in Training/output", "success");
                    } else {
                        showToast("LoRA Training Run encountered an issue.", "error");
                    }
                }
            } catch (e) {
                console.warn("Poll error:", e);
            }
        }, 1000);
    }
};

window.Training = Training;
document.addEventListener("DOMContentLoaded", () => Training.init());
