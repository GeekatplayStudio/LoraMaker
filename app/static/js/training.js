// LoRA Training Pipeline Controller
const Training = {
    selectedModel: "flux-1-dev",
    pollInterval: null,
    capabilities: new Map(),

    // A file's size is not evidence that it is a usable LoRA.  The API is the
    // authority for provenance/validation; older API responses intentionally
    // remain "unverified" instead of being promoted based on a heuristic.
    modelTruth(model) {
        const verified = model.verified === true || model.is_verified === true ||
            model.is_valid_lora === true || model.artifact_verified === true;
        const unavailable = model.available === false || model.training_supported === false ||
            model.render_supported === false || model.is_valid_lora === false ||
            model.artifact_valid === false;
        const label = model.verification_status || model.artifact_status;

        if (unavailable) return { unavailable: true, className: "badge-warning", text: `⚠️ ${label || "Unavailable / invalid artifact"}`, color: "#fca5a5" };
        if (verified) return { unavailable: false, className: "badge-optimal", text: `✓ ${label || "Verified LoRA artifact"}`, color: "#34d399" };
        return { unavailable: false, className: "badge-warning", text: `? ${label || "Unverified artifact"}`, color: "#fde68a" };
    },

    init() {
        // Model card selection
        document.querySelectorAll(".model-card").forEach(card => {
            card.addEventListener("click", () => {
                const capability = this.capabilities.get(card.getAttribute("data-model-id"));
                if (capability && capability.available === false) {
                    showToast(capability.reason || "No compatible local trainer is available for this model.", "error");
                    return;
                }
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
        document.getElementById("btn-refresh-manifest")?.addEventListener("click", () => this.loadDatasetManifest());
        document.getElementById("btn-refresh-vault")?.addEventListener("click", () => this.loadLoRAVault());

        // Initial loads
        this.loadCapabilities();
        this.loadDatasetManifest();
        this.loadLoRAVault();
    },

    async loadCapabilities() {
        try {
            const response = await fetch("/api/training/capabilities");
            const data = await response.json();
            [...(data.architectures || []), ...(data.unavailable_architectures || [])].forEach(item => this.capabilities.set(item.id, item));
            document.querySelectorAll(".model-card").forEach(card => {
                const capability = this.capabilities.get(card.getAttribute("data-model-id"));
                if (capability && capability.available === false) {
                    card.style.opacity = "0.45";
                    card.title = capability.reason || (capability.missing_requirements || []).join(", ");
                    card.setAttribute("aria-disabled", "true");
                }
            });
        } catch (error) {
            console.warn("Training capability check failed", error);
        }
    },

    async loadDatasetManifest() {
        if (!AppState.projectDir) return;
        const tbody = document.getElementById("manifest-table-tbody");
        const statTotal = document.getElementById("manifest-stat-total");
        const statUpscaled = document.getElementById("manifest-stat-upscaled");
        const statPadded = document.getElementById("manifest-stat-padded");

        try {
            const res = await fetch(`/api/training/manifest?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            if (statTotal) statTotal.textContent = data.total_images || 0;
            if (statUpscaled) statUpscaled.textContent = `${data.total_upscaled || 0} (${Math.round(((data.total_upscaled || 0) / Math.max(1, data.total_images || 1)) * 100)}%)`;
            if (statPadded) statPadded.textContent = `${data.total_padded || 0} (${data.framing_mode || 'pad'})`;

            if (tbody && data.images && data.images.length > 0) {
                tbody.innerHTML = data.images.map(img => `
                    <tr>
                        <td class="mono" style="font-weight: 600; color: #fde68a;">${img.filename}</td>
                        <td class="mono">${img.original_width} × ${img.original_height}</td>
                        <td><span class="eval-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">${img.original_aspect}</span></td>
                        <td class="mono">${img.fitted_width} × ${img.fitted_height} (${img.framing_mode})</td>
                        <td>${img.upscaled ? '<span style="color: #fbbf24; font-weight: 600;">Yes (Upscaled)</span>' : '<span style="color: #34d399;">Native / Down</span>'}</td>
                        <td style="color: var(--text-dim); max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${img.caption || ''}">${img.caption || '<em>(no caption)</em>'}</td>
                    </tr>
                `).join("");
            } else if (tbody) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-dim); padding: 14px;">No keyframes found in Keyframes_Out yet.</td></tr>`;
            }
        } catch (e) {
            console.warn("Failed to load dataset manifest:", e);
        }
    },

    async loadLoRAVault() {
        if (!AppState.projectDir) return;
        const tbody = document.getElementById("lora-vault-tbody");
        if (!tbody) return;

        try {
            const res = await fetch(`/api/evaluation/models?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            if (data.models && data.models.length > 0) {
                tbody.innerHTML = data.models.map(m => {
                    const truth = this.modelTruth(m);
                    const testButton = truth.unavailable
                        ? `<button class="btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" disabled title="The backend marked this artifact unavailable">Unavailable</button>`
                        : `<button class="btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" onclick="Evaluation.selectAndGoToBench('${m.filename}')">🧪 Test in Lab</button>`;

                    return `
                        <tr>
                            <td class="mono" style="font-weight: 600; color: ${truth.color};">${m.filename}</td>
                            <td class="mono" style="font-weight: 700;">${m.size_mb > 0 ? m.size_mb + ' MB' : m.size_kb + ' KB'}</td>
                            <td><span class="eval-badge ${truth.className}">${truth.text}</span></td>
                            <td class="mono">${m.base_model_hint || 'Unknown — not verified'}</td>
                            <td style="font-size: 0.76rem; color: var(--text-dim);">${m.training_engine || 'No engine provenance recorded'}</td>
                            <td class="mono">${m.total_steps ? `${m.total_steps} steps (loss: ${m.final_loss || 'N/A'})` : 'N/A'}</td>
                            <td style="font-size: 0.75rem; color: var(--text-dim);">${m.modified_datetime || 'Recently'}</td>
                            <td>
                                ${testButton}
                            </td>
                        </tr>
                    `;
                }).join("");
            } else {
                tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 16px;">No trained LoRA models found in Training/output/. Launch a training run to generate real weights.</td></tr>`;
            }
        } catch (e) {
            console.warn("Failed to load LoRA vault:", e);
        }
    },

    drawLossChart(lossHistory) {
        const canvas = document.getElementById("training-loss-chart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        if (!lossHistory || lossHistory.length === 0) {
            ctx.fillStyle = "#64748b";
            ctx.font = "12px JetBrains Mono";
            ctx.textAlign = "center";
            ctx.fillText("Awaiting loss telemetry points...", w / 2, h / 2);
            return;
        }

        const losses = lossHistory.map(p => p.loss);
        const minLoss = Math.min(...losses);
        const maxLoss = Math.max(...losses);
        const range = Math.max(0.001, maxLoss - minLoss);

        const minmaxEl = document.getElementById("loss-chart-minmax");
        if (minmaxEl) {
            minmaxEl.textContent = `Steps: ${lossHistory[lossHistory.length - 1].step} | Min: ${minLoss.toFixed(4)} | Max: ${maxLoss.toFixed(4)}`;
        }

        // Draw horizontal grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        for (let i = 1; i <= 3; i++) {
            const y = (h / 4) * i;
            ctx.beginPath();
            ctx.moveTo(30, y);
            ctx.lineTo(w - 10, y);
            ctx.stroke();
        }

        // Draw loss curve
        const paddingLeft = 35;
        const paddingRight = 15;
        const paddingTop = 15;
        const paddingBottom = 20;

        const plotW = w - paddingLeft - paddingRight;
        const plotH = h - paddingTop - paddingBottom;

        ctx.beginPath();
        lossHistory.forEach((pt, idx) => {
            const x = paddingLeft + (idx / Math.max(1, lossHistory.length - 1)) * plotW;
            const normY = (pt.loss - minLoss) / range;
            const y = h - paddingBottom - normY * plotH;

            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Fill area under curve
        ctx.lineTo(paddingLeft + plotW, h - paddingBottom);
        ctx.lineTo(paddingLeft, h - paddingBottom);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, paddingTop, 0, h - paddingBottom);
        grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // Draw min/max text labels
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px JetBrains Mono";
        ctx.textAlign = "right";
        ctx.fillText(maxLoss.toFixed(2), paddingLeft - 5, paddingTop + 8);
        ctx.fillText(minLoss.toFixed(2), paddingLeft - 5, h - paddingBottom);
    },

    async loadModelProfile(modelId) {
        if (!AppState.projectDir) return;
        try {
            const res = await fetch(`/api/training/auto_tune?project_dir=${encodeURIComponent(AppState.projectDir)}&base_model=${encodeURIComponent(modelId)}`);
            if (!res.ok) return;
            const params = await res.json();

            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el && val !== undefined) el.value = val;
            };

            setVal("train-rank-input", params.lora_rank);
            setVal("train-alpha-input", params.lora_alpha);
            setVal("train-lr-input", params.learning_rate);
            setVal("train-epochs-input", params.recommended_epochs);
            setVal("train-repeats-input", params.recommended_repeats);
            setVal("train-batch-input", params.batch_size);

            const badge = document.getElementById("auto-tune-badge");
            if (badge) {
                badge.innerHTML = `⚡ <strong>Hardware Auto-Tuned</strong> for ${params.device_name || 'GPU'} (${params.vram_tier}): Batch=${params.batch_size}, LR=${params.learning_rate}, ${params.target_total_steps} target steps (${params.recommended_epochs} epochs × ${params.recommended_repeats} repeats).`;
                badge.style.display = "block";
            }
        } catch (e) {
            console.warn("Auto-tune error:", e);
        }
    },

    async startTraining() {
        if (!AppState.projectDir) return showToast("Create or select a project first.", "error");
        const capability = this.capabilities.get(this.selectedModel);
        if (capability && capability.available === false) return showToast(capability.reason || "No matching local trainer is available.", "error");

        if (window.DiagnosticsConsole) {
            DiagnosticsConsole.clearInlineError('training-error-banner');
            DiagnosticsConsole.clearInlineError('training-preflight-error-banner');
        }

        const value = id => document.getElementById(id)?.value;
        const payload = {
            project_dir: AppState.projectDir,
            base_model: this.selectedModel,
            lora_rank: Number(value("train-rank-input")),
            lora_alpha: Number(value("train-alpha-input")),
            learning_rate: Number(value("train-lr-input")),
            epochs: Number(value("train-epochs-input")),
            repeats: Number(value("train-repeats-input")),
            batch_size: Number(value("train-batch-input")),
            execution_mode: value("train-exec-mode"),
            framing_mode: value("train-framing-mode")
        };
        const button = document.getElementById("btn-start-training");
        if (button) button.disabled = true;

        let response;
        try {
            response = await fetch("/api/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } catch (netErr) {
            if (button) button.disabled = false;
            const netErrorObj = {
                endpoint: "/api/training/start",
                method: "POST",
                status_code: 0,
                error_type: "NetworkConnectionError",
                message: "Network Error: Unable to connect to backend server at http://127.0.0.1:7860. The server process may have stopped or restarted.",
                traceback: netErr.stack || String(netErr),
                request_payload: payload,
                suggestion: "Verify that the Python backend process is active on port 7860, or check the terminal log."
            };
            if (window.DiagnosticsConsole) {
                DiagnosticsConsole.errors.unshift(netErrorObj);
                DiagnosticsConsole.updateBadge();
                DiagnosticsConsole.renderInlineError('training-error-banner', netErrorObj);
                DiagnosticsConsole.renderInlineError('training-preflight-error-banner', netErrorObj);
            }
            showToast(netErrorObj.message, "error");
            return;
        }

        let data;
        try {
            data = await response.json();
        } catch (jsonErr) {
            data = { detail: "Invalid non-JSON response returned from server" };
        }

        if (!response.ok) {
            if (button) button.disabled = false;
            let errDetail = data.detail || data;
            let errMsg = typeof errDetail === 'string' ? errDetail : (errDetail.message || errDetail.detail || "Training preflight failed");
            let errType = (typeof errDetail === 'object' && errDetail.error_type) ? errDetail.error_type : "TrainingPreflightError";
            let tb = (typeof errDetail === 'object' && errDetail.traceback) ? errDetail.traceback : "";
            let suggestion = (typeof errDetail === 'object' && errDetail.suggestion) ? errDetail.suggestion : "";

            const errorRecord = {
                endpoint: "/api/training/start",
                method: "POST",
                status_code: response.status,
                error_type: errType,
                message: errMsg,
                traceback: tb,
                request_payload: payload,
                suggestion: suggestion
            };

            if (window.DiagnosticsConsole) {
                DiagnosticsConsole.errors.unshift(errorRecord);
                DiagnosticsConsole.updateBadge();
                DiagnosticsConsole.renderInlineError('training-error-banner', errorRecord);
                DiagnosticsConsole.renderInlineError('training-preflight-error-banner', errorRecord);
            }
            showToast(`${errType}: ${errMsg}`, "error");
            return;
        }

        showToast(data.status === "validated" ? "Preflight passed; no weights were created." : "Real training started.", "success");
        if (data.status === "started") {
            this.startPollingStatus();
        } else if (button) {
            button.disabled = false;
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
                const telemetryVerified = data.telemetry_verified === true || data.metrics_verified === true;
                if (lossText) lossText.textContent = telemetryVerified
                    ? `Current Loss: ${data.current_loss ?? "N/A"}`
                    : "Current Loss: telemetry not verified";

                if (telemetryVerified && data.loss_history) {
                    this.drawLossChart(data.loss_history);
                } else if (!telemetryVerified) {
                    this.drawLossChart([]);
                }

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
                        showToast(data.artifact_verified === true || data.output_verified === true
                            ? "Training completed and the output artifact was verified."
                            : "Training completed, but output verification is still required.",
                            data.artifact_verified === true || data.output_verified === true ? "success" : "error");
                        this.loadLoRAVault();
                        this.loadDatasetManifest();
                    } else {
                        const failRecord = {
                            endpoint: `/api/training/status?project_dir=${encodeURIComponent(AppState.projectDir)}`,
                            method: "GET",
                            status_code: 200,
                            error_type: "TrainingExecutionError",
                            message: data.error || "Training run encountered an execution error.",
                            traceback: (data.log || []).slice(-20).join("\n"),
                            suggestion: "Review the training log stream above or click 'Copy Diagnostic Report' for full context."
                        };
                        if (window.DiagnosticsConsole) {
                            DiagnosticsConsole.errors.unshift(failRecord);
                            DiagnosticsConsole.updateBadge();
                            DiagnosticsConsole.renderInlineError('training-error-banner', failRecord);
                        }
                        showToast("LoRA Training Run encountered an issue. See Diagnostics Console for details.", "error");
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
