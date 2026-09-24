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
        document.getElementById("btn-refresh-manifest")?.addEventListener("click", () => this.loadDatasetManifest());
        document.getElementById("btn-refresh-vault")?.addEventListener("click", () => this.loadLoRAVault());

        // Initial loads
        this.loadDatasetManifest();
        this.loadLoRAVault();
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
                    const isGenuine = m.is_genuine || m.size_mb > 5.0;
                    const badgeClass = isGenuine ? 'badge-optimal' : 'badge-warning';
                    const badgeText = isGenuine ? `✅ Genuine LoRA (${m.size_mb} MB)` : `⚠️ Mock Stub (${m.size_kb} KB)`;

                    return `
                        <tr>
                            <td class="mono" style="font-weight: 600; color: ${isGenuine ? '#34d399' : '#fde68a'};">${m.filename}</td>
                            <td class="mono" style="font-weight: 700;">${m.size_mb > 0 ? m.size_mb + ' MB' : m.size_kb + ' KB'}</td>
                            <td><span class="eval-badge ${badgeClass}">${badgeText}</span></td>
                            <td class="mono">${m.base_model_hint || 'sdxl-1.0'}</td>
                            <td style="font-size: 0.76rem; color: var(--text-dim);">${m.training_engine || 'Kohya / PyTorch'}</td>
                            <td class="mono">${m.total_steps ? `${m.total_steps} steps (loss: ${m.final_loss || 'N/A'})` : 'N/A'}</td>
                            <td style="font-size: 0.75rem; color: var(--text-dim);">${m.modified_datetime || 'Recently'}</td>
                            <td>
                                <button class="btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" onclick="Evaluation.selectAndGoToBench('${m.filename}')">
                                    🧪 Test in Lab
                                </button>
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

                if (data.loss_history) {
                    this.drawLossChart(data.loss_history);
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
                        showToast("LoRA Training Run Completed! Real weights saved in Training/output", "success");
                        this.loadLoRAVault();
                        this.loadDatasetManifest();
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
