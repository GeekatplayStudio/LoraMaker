/**
 * Settings & Storage Management Module
 * Geekatplay LoRA Maker - Vladimir Chopine
 * Repository: https://github.com/GeekatplayStudio/LoraMaker.git
 */

const SettingsManager = {
    modal: null,

    init() {
        this.modal = document.getElementById("settings-modal");
        
        // Open button
        document.getElementById("btn-open-settings")?.addEventListener("click", () => this.openModal());
        
        // Close buttons
        document.getElementById("btn-close-settings")?.addEventListener("click", () => this.closeModal());
        document.getElementById("btn-cancel-settings")?.addEventListener("click", () => this.closeModal());

        // Scan button
        document.getElementById("btn-scan-models")?.addEventListener("click", () => this.scanModels());

        // Save button
        document.getElementById("btn-save-settings")?.addEventListener("click", () => this.saveSettings());

        // Auto-detect button
        document.getElementById("btn-autodetect-paths")?.addEventListener("click", () => this.autoDetectPaths());
    },

    async openModal() {
        if (!this.modal) this.modal = document.getElementById("settings-modal");
        if (this.modal) {
            this.modal.classList.add("active");
            this.modal.style.display = "flex";
        }
        await this.loadSettings();
    },

    closeModal() {
        if (this.modal) {
            this.modal.classList.remove("active");
            this.modal.style.display = "none";
        }
    },

    async loadSettings() {
        try {
            const res = await fetch("/api/system/settings");
            const data = await res.json();

            // Populate inputs
            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el && val) el.value = val;
            };

            setVal("cfg-models-dir", data.MODELS_DIR);
            setVal("cfg-download-dir", data.DOWNLOAD_DIR);
            setVal("cfg-comfyui-root", data.COMFYUI_ROOT);
            setVal("cfg-kohya-root", data.KOHYA_ROOT);
            setVal("cfg-ollama-host", data.OLLAMA_HOST);

            // Render drive space cards
            this.renderDrives(data.drives || {});

            // Update status indicators
            this.updatePathStatus(data.paths_status || {});
        } catch (e) {
            console.warn("Failed to load settings:", e);
        }
    },

    renderDrives(drives) {
        const container = document.getElementById("drives-container");
        if (!container) return;

        const driveList = Object.values(drives);
        if (driveList.length === 0) {
            container.innerHTML = `<div style="color: var(--text-dim); font-size: 0.8rem;">No drive telemetry available.</div>`;
            return;
        }

        container.innerHTML = driveList.map(d => {
            const isLow = d.is_low_space || d.free_gb < 20;
            const barColor = isLow ? "#ef4444" : (d.percent_free < 25 ? "#f59e0b" : "#10b981");
            const badgeText = isLow ? "⚠️ Low Storage" : "✅ Plenty Space";
            const badgeColor = isLow ? "#ef4444" : "#10b981";

            return `
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px; flex: 1; min-width: 180px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="mono" style="font-weight: 700; color: #fff;">Drive ${d.root}</span>
                        <span style="font-size: 0.72rem; color: ${badgeColor}; font-weight: 600;">${badgeText}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: var(--text-dim); margin-bottom: 6px;">
                        <strong style="color: #fff;">${d.free_gb} GB free</strong> of ${d.total_gb} GB (${d.percent_free}% free)
                    </div>
                    <div style="width: 100%; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden;">
                        <div style="width: ${100 - d.percent_free}%; height: 100%; background: ${barColor};"></div>
                    </div>
                </div>
            `;
        }).join("");
    },

    updatePathStatus(status) {
        const setIndicator = (id, exists) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.innerHTML = exists 
                ? `<span style="color: #10b981; font-weight: 600;">✅ Found</span>` 
                : `<span style="color: #ef4444; font-weight: 600;">❌ Directory Not Found</span>`;
        };

        setIndicator("status-models-dir", status.models_dir_exists);
        setIndicator("status-download-dir", status.download_dir_exists);
        setIndicator("status-comfyui", status.comfyui_exists);
        setIndicator("status-kohya", status.kohya_root_exists || status.kohya_fallback_exists);
    },

    async scanModels() {
        const input = document.getElementById("cfg-models-dir");
        const dir = input ? input.value : "";
        const resultEl = document.getElementById("models-scan-results");
        if (resultEl) {
            resultEl.style.display = "block";
            resultEl.innerHTML = `<div style="color: var(--amber-primary); padding: 10px;">🔍 Scanning models directory for checkpoints, UNets, and LoRAs...</div>`;
        }

        try {
            const res = await fetch("/api/system/settings/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ models_dir: dir })
            });
            const data = await res.json();

            if (!data.success) {
                if (resultEl) resultEl.innerHTML = `<div style="color: #ef4444; padding: 10px;">❌ ${data.error}</div>`;
                return;
            }

            const cats = data.categories || {};
            if (resultEl) {
                resultEl.innerHTML = `
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <strong style="color: #34d399;">Model Inventory Scan Complete: ${data.total_models} models found (${data.total_size_gb} GB total)</strong>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; font-size: 0.78rem;">
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">Checkpoints:</span> <strong style="color: #fff;">${cats.checkpoints?.count || 0}</strong>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">UNet (FLUX):</span> <strong style="color: #fff;">${cats.unet?.count || 0}</strong>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">Video Models:</span> <strong style="color: #fff;">${cats.diffusion_models?.count || 0}</strong>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">Text Encoders:</span> <strong style="color: #fff;">${cats.clip?.count || 0}</strong>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">VAEs:</span> <strong style="color: #fff;">${cats.vae?.count || 0}</strong>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px;">
                                <span style="color: var(--text-dim);">Trained LoRAs:</span> <strong style="color: #fff;">${cats.loras?.count || 0}</strong>
                            </div>
                        </div>
                    </div>
                `;
            }
            showToast(`Scanned ${data.total_models} models (${data.total_size_gb} GB) in configured folder.`, "success");
        } catch (e) {
            if (resultEl) resultEl.innerHTML = `<div style="color: #ef4444; padding: 10px;">❌ Scan failed: ${e.message}</div>`;
        }
    },

    async autoDetectPaths() {
        const candidates = ["D:/ComfyUI/ComfyUI", "D:/ComfyUI", "C:/ComfyUI", "D:/AI_Models"];
        showToast("Auto-probing local drives for ComfyUI and models...", "info");
        await this.loadSettings();
        await this.scanModels();
    },

    async saveSettings() {
        const val = id => document.getElementById(id)?.value?.trim();
        const payload = {
            MODELS_DIR: val("cfg-models-dir"),
            DOWNLOAD_DIR: val("cfg-download-dir"),
            COMFYUI_ROOT: val("cfg-comfyui-root"),
            KOHYA_ROOT: val("cfg-kohya-root"),
            OLLAMA_HOST: val("cfg-ollama-host")
        };

        const btn = document.getElementById("btn-save-settings");
        if (btn) btn.disabled = true;

        try {
            const res = await fetch("/api/system/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            showToast("Configuration saved! Downloads redirected to protect primary drive.", "success");
            this.closeModal();

            // Refresh model capabilities and base checkpoints
            if (window.Training?.loadCapabilities) window.Training.loadCapabilities();
            if (window.Evaluation?.loadBaseCheckpoints) window.Evaluation.loadBaseCheckpoints();
        } catch (e) {
            showToast(`Failed to save settings: ${e.message}`, "error");
        } finally {
            if (btn) btn.disabled = false;
        }
    }
};

window.SettingsManager = SettingsManager;
