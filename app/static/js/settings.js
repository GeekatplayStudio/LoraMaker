/**
 * Settings & Storage Management Module
 * Supports Multi-Location Model Directories Across Multiple Drives
 * Geekatplay LoRA Maker - Vladimir Chopine
 * Repository: https://github.com/GeekatplayStudio/LoraMaker.git
 */

const SettingsManager = {
    modal: null,

    init() {
        this.modal = document.getElementById("settings-modal");
        
        // Open button in header
        document.getElementById("btn-open-settings")?.addEventListener("click", () => this.openModal());
        
        // Close buttons
        document.getElementById("btn-close-settings")?.addEventListener("click", () => this.closeModal());
        document.getElementById("btn-cancel-settings")?.addEventListener("click", () => this.closeModal());

        // Add Model Folder button
        document.getElementById("btn-add-model-folder")?.addEventListener("click", () => {
            FolderBrowser.open((selectedFolder) => this.addModelPath(selectedFolder));
        });

        // Scan button (scans all configured locations)
        document.getElementById("btn-scan-models")?.addEventListener("click", () => this.scanModels());

        // Save button
        document.getElementById("btn-save-settings")?.addEventListener("click", () => this.saveSettings());

        // Auto-detect button
        document.getElementById("btn-autodetect-paths")?.addEventListener("click", () => this.autoDetectPaths());

        // Interactive folder browser
        FolderBrowser.init();
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

            // Populate form inputs
            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el && val) el.value = val;
            };

            setVal("cfg-models-dir", data.MODELS_DIR);
            setVal("cfg-download-dir", data.DOWNLOAD_DIR);
            setVal("cfg-comfyui-root", data.COMFYUI_ROOT);
            setVal("cfg-kohya-root", data.KOHYA_ROOT);
            setVal("cfg-ollama-host", data.OLLAMA_HOST);

            // Render multi-location model directories list
            this.renderModelPathsList(data.MODELS_DIR, data.EXTRA_MODEL_PATHS || [], data.drives || {});

            // Render drive space telemetry cards
            this.renderDrives(data.drives || {});

            // Update status indicators
            this.updatePathStatus(data.paths_status || {});
        } catch (e) {
            console.warn("Failed to load settings:", e);
        }
    },

    renderModelPathsList(primary, extras, drives) {
        const container = document.getElementById("model-paths-list");
        if (!container) return;

        let allLocations = [{ path: primary, isPrimary: true }];
        (extras || []).forEach(p => {
            if (p && p.toLowerCase() !== (primary || "").toLowerCase()) {
                allLocations.push({ path: p, isPrimary: false });
            }
        });

        container.innerHTML = allLocations.map((loc) => {
            const driveLetter = loc.path.slice(0, 2).toUpperCase();
            const driveKey = driveLetter + "\\";
            const driveData = drives[driveKey] || {};
            const driveFreeStr = driveData.free_gb !== undefined ? `${driveData.free_gb} GB free` : '';
            const drivePill = driveFreeStr 
                ? `<span style="font-size: 0.7rem; background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 4px; color: var(--text-dim); white-space: nowrap;">💽 Drive ${driveLetter} (${driveFreeStr})</span>` 
                : `<span style="font-size: 0.7rem; background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 4px; color: var(--text-dim); white-space: nowrap;">💽 Drive ${driveLetter}</span>`;

            const escapedPath = loc.path.replace(/\\/g, '\\\\');

            const badge = loc.isPrimary 
                ? `<span style="background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">⭐ Primary</span>`
                : `<button type="button" class="btn-secondary" style="padding: 3px 8px; font-size: 0.72rem;" onclick="SettingsManager.setPrimaryModelPath('${escapedPath}')" title="Make this folder the primary download destination">Set Primary</button>
                   <button type="button" class="btn-secondary" style="padding: 3px 8px; font-size: 0.72rem; color: #ef4444;" onclick="SettingsManager.removeModelPath('${escapedPath}')" title="Remove this search location">🗑️</button>`;

            return `
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.07); border-radius: 6px; padding: 8px 12px; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; flex: 1;">
                        <span style="font-size: 1.1rem;">📁</span>
                        <div style="overflow: hidden; flex: 1;">
                            <div class="mono" style="font-size: 0.82rem; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500;" title="${loc.path}">
                                ${loc.path}
                            </div>
                        </div>
                        ${drivePill}
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <button type="button" class="btn-secondary" style="padding: 3px 8px; font-size: 0.72rem;" onclick="SettingsManager.scanModels('${escapedPath}')" title="Scan models in this specific folder">🔍 Scan</button>
                        ${badge}
                    </div>
                </div>
            `;
        }).join("");
    },

    async addModelPath(path) {
        if (!path) return;
        try {
            const res = await fetch("/api/system/settings/add_model_path", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path })
            });
            const data = await res.json();
            showToast(`Added model search folder: ${path}`, "success");
            await this.loadSettings();
            await this.scanModels();
        } catch (e) {
            showToast(`Failed to add folder: ${e.message}`, "error");
        }
    },

    async removeModelPath(path) {
        if (!path) return;
        try {
            const res = await fetch("/api/system/settings/remove_model_path", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path })
            });
            const data = await res.json();
            showToast(`Removed model search folder: ${path}`, "info");
            await this.loadSettings();
            await this.scanModels();
        } catch (e) {
            showToast(`Failed to remove folder: ${e.message}`, "error");
        }
    },

    async setPrimaryModelPath(path) {
        if (!path) return;
        try {
            const res = await fetch("/api/system/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ MODELS_DIR: path })
            });
            const data = await res.json();
            showToast(`Set primary model folder to: ${path}`, "success");
            await this.loadSettings();
        } catch (e) {
            showToast(`Failed to set primary: ${e.message}`, "error");
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
            const badgeText = isLow ? "⚠️ Low Space" : "✅ Plenty Space";
            const badgeColor = isLow ? "#ef4444" : "#10b981";

            return `
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px; flex: 1; min-width: 170px;">
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

        setIndicator("status-download-dir", status.download_dir_exists);
        setIndicator("status-comfyui", status.comfyui_exists);
        setIndicator("status-kohya", status.kohya_root_exists || status.kohya_fallback_exists);
    },

    async scanModels(specificDir = null) {
        const resultEl = document.getElementById("models-scan-results");
        if (resultEl) {
            resultEl.style.display = "block";
            resultEl.innerHTML = `<div style="color: var(--amber-primary); padding: 10px;">🔍 Scanning model directories across all locations...</div>`;
        }

        try {
            const res = await fetch("/api/system/settings/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ models_dir: specificDir })
            });
            const data = await res.json();

            if (!data.success) {
                if (resultEl) resultEl.innerHTML = `<div style="color: #ef4444; padding: 10px;">❌ ${data.error}</div>`;
                return;
            }

            const cats = data.categories || {};
            const locs = data.locations_scanned || [];

            let locsHtml = "";
            if (locs.length > 0) {
                locsHtml = `
                    <div style="margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                        <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 6px; font-weight: 600;">Locations Scanned (${locs.length}):</div>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            ${locs.map(l => `
                                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; background: rgba(0,0,0,0.25); padding: 6px 10px; border-radius: 4px;">
                                    <span class="mono" style="color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%;" title="${l.path}">${l.path}</span>
                                    <span style="color: ${l.exists ? 'var(--amber-primary)' : '#ef4444'}; font-weight: 600;">
                                        ${l.exists ? `${l.models_count} models (${l.size_gb} GB)` : 'Not Found'}
                                    </span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                `;
            }

            if (resultEl) {
                resultEl.innerHTML = `
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <strong style="color: #34d399;">Model Inventory Scan: ${data.total_models} models found (${data.total_size_gb} GB total)</strong>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; font-size: 0.78rem;">
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
                        ${locsHtml}
                    </div>
                `;
            }
            showToast(`Scanned ${data.total_models} models (${data.total_size_gb} GB) across configured locations.`, "success");
        } catch (e) {
            if (resultEl) resultEl.innerHTML = `<div style="color: #ef4444; padding: 10px;">❌ Scan failed: ${e.message}</div>`;
        }
    },

    async autoDetectPaths() {
        showToast("Auto-probing local drives for model libraries and ComfyUI...", "info");
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

            showToast("Configuration saved! All model locations indexed and downloads redirected.", "success");
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

/**
 * Interactive Server-Side Directory Navigator
 * Supports selecting for input fields OR function callbacks.
 */
const FolderBrowser = {
    modal: null,
    targetInputId: null,
    onSelectCallback: null,
    currentPath: "",
    parentPath: null,
    selectedPath: "",
    drives: [],
    items: [],

    init() {
        this.modal = document.getElementById("dir-browser-modal");

        // Event delegation for browse buttons
        document.addEventListener("click", (e) => {
            const btn = e.target.closest(".btn-browse-folder");
            if (btn) {
                const targetId = btn.getAttribute("data-target");
                const currentVal = document.getElementById(targetId)?.value?.trim();
                this.open(targetId, currentVal);
            }
        });

        // Close buttons
        document.getElementById("btn-close-browser")?.addEventListener("click", () => this.close());
        document.getElementById("btn-browser-cancel")?.addEventListener("click", () => this.close());

        // Nav toolbar buttons
        document.getElementById("btn-browser-up")?.addEventListener("click", () => this.navigateUp());
        document.getElementById("btn-browser-drives")?.addEventListener("click", () => this.navigate(""));
        document.getElementById("btn-browser-refresh")?.addEventListener("click", () => this.navigate(this.currentPath));

        // Go button & Enter on path input
        document.getElementById("btn-browser-go")?.addEventListener("click", () => {
            const path = document.getElementById("browser-path-input")?.value?.trim();
            this.navigate(path);
        });
        document.getElementById("browser-path-input")?.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                this.navigate(e.target.value.trim());
            }
        });

        // Live folder filter search
        document.getElementById("browser-filter-input")?.addEventListener("input", (e) => {
            this.renderItems(e.target.value.trim());
        });

        // Confirm Selection button
        document.getElementById("btn-browser-select")?.addEventListener("click", () => this.confirmSelection());
    },

    open(targetInputIdOrCallback, initialPath = "") {
        if (typeof targetInputIdOrCallback === "function") {
            this.onSelectCallback = targetInputIdOrCallback;
            this.targetInputId = null;
            const label = document.getElementById("browser-target-label");
            if (label) label.textContent = "Selecting new model directory location to add";
        } else {
            this.onSelectCallback = null;
            this.targetInputId = targetInputIdOrCallback;
            const targetInput = document.getElementById(targetInputIdOrCallback);
            const label = document.getElementById("browser-target-label");
            if (label && targetInput) {
                const fieldLabel = targetInput.closest(".form-group")?.querySelector(".form-label")?.textContent || targetInputIdOrCallback;
                label.textContent = `Selecting folder for: ${fieldLabel}`;
            }
        }

        if (!this.modal) this.modal = document.getElementById("dir-browser-modal");
        if (this.modal) {
            this.modal.classList.add("active");
            this.modal.style.display = "flex";
        }

        const startPath = initialPath || (this.targetInputId ? document.getElementById(this.targetInputId)?.value?.trim() : "") || "";
        this.navigate(startPath);
    },

    close() {
        if (this.modal) {
            this.modal.classList.remove("active");
            this.modal.style.display = "none";
        }
    },

    async navigate(path) {
        const container = document.getElementById("browser-items-container");
        if (container) {
            container.innerHTML = `<div style="color: var(--amber-primary); padding: 25px; text-align: center;">📂 Scanning filesystem...</div>`;
        }

        try {
            const url = `/api/system/browse` + (path ? `?path=${encodeURIComponent(path)}` : "");
            const res = await fetch(url);
            const data = await res.json();

            if (!data.success) {
                if (container) {
                    container.innerHTML = `<div style="color: #ef4444; padding: 25px; text-align: center;">❌ ${data.error || 'Failed to read directory'}</div>`;
                }
                return;
            }

            this.currentPath = data.current_path || "";
            this.parentPath = data.parent_path !== undefined ? data.parent_path : null;
            this.items = data.items || [];

            // Default selectedPath to current directory if not root
            this.selectedPath = this.currentPath || "";
            this.updateSelectedPreview();

            // Update path input
            const pathInput = document.getElementById("browser-path-input");
            if (pathInput) pathInput.value = this.currentPath;

            // Render breadcrumbs
            this.renderBreadcrumbs(data.breadcrumbs || []);

            // If root or drives received, store drives list
            if (data.is_root) {
                this.drives = data.items || [];
                this.renderSidebar();
            } else if (this.drives.length === 0) {
                this.fetchDrivesSidebar();
            } else {
                this.renderSidebar();
            }

            // Reset filter input & render items
            const filterInput = document.getElementById("browser-filter-input");
            if (filterInput) filterInput.value = "";
            this.renderItems("");

        } catch (e) {
            if (container) {
                container.innerHTML = `<div style="color: #ef4444; padding: 20px;">❌ Network error: ${e.message}</div>`;
            }
        }
    },

    async fetchDrivesSidebar() {
        try {
            const res = await fetch("/api/system/browse");
            const data = await res.json();
            if (data.success && data.items) {
                this.drives = data.items;
                this.renderSidebar();
            }
        } catch (e) {
            console.warn("Could not fetch drives for sidebar:", e);
        }
    },

    navigateUp() {
        if (this.parentPath !== null) {
            this.navigate(this.parentPath);
        } else {
            this.navigate("");
        }
    },

    renderBreadcrumbs(bcs) {
        const container = document.getElementById("browser-breadcrumbs");
        if (!container) return;

        let html = `
            <button type="button" class="browser-breadcrumb-chip" onclick="FolderBrowser.navigate('')" title="All Connected Storage Drives">
                💽 Drives
            </button>
        `;

        bcs.forEach((bc, idx) => {
            const isLast = idx === bcs.length - 1;
            const style = isLast ? "font-weight: 700; color: var(--amber-primary);" : "";
            const escaped = bc.path.replace(/\\/g, '\\\\');
            html += `
                <span style="color: var(--text-dim); font-size: 0.76rem;">›</span>
                <button type="button" class="browser-breadcrumb-chip" style="${style}" onclick="FolderBrowser.navigate('${escaped}')">
                    ${bc.name}
                </button>
            `;
        });

        container.innerHTML = html;
    },

    renderSidebar() {
        // 1. Quick locations
        const quickContainer = document.getElementById("browser-quick-locations");
        if (quickContainer) {
            const currentModels = document.getElementById("cfg-models-dir")?.value?.trim();
            const currentComfy = document.getElementById("cfg-comfyui-root")?.value?.trim();

            const locs = [
                { name: "💽 All Drives", path: "" }
            ];
            if (currentModels) locs.push({ name: "🎯 Current Models", path: currentModels });
            if (currentComfy) locs.push({ name: "⚡ ComfyUI Root", path: currentComfy });

            quickContainer.innerHTML = locs.map(l => {
                const escaped = l.path.replace(/\\/g, '\\\\');
                return `
                    <div class="browser-sidebar-item" onclick="FolderBrowser.navigate('${escaped}')">
                        <span>${l.name}</span>
                    </div>
                `;
            }).join("");
        }

        // 2. Drives list
        const drivesContainer = document.getElementById("browser-drives-list");
        if (drivesContainer && this.drives) {
            drivesContainer.innerHTML = this.drives.map(d => {
                const isSelected = this.currentPath.toUpperCase().startsWith(d.path.toUpperCase().slice(0, 2));
                const activeStyle = isSelected ? "border-color: var(--amber-primary); background: rgba(245,158,11,0.15);" : "";
                const freeStr = d.free_gb ? `${d.free_gb} GB` : '';
                const escaped = d.path.replace(/\\/g, '\\\\');
                return `
                    <div class="browser-sidebar-item" style="${activeStyle}" onclick="FolderBrowser.navigate('${escaped}')" title="${d.name}: ${d.free_gb || 0} GB free of ${d.total_gb || 0} GB">
                        <span style="font-weight: 600;">💽 ${d.path}</span>
                        <span style="font-size: 0.7rem; color: var(--text-dim);">${freeStr}</span>
                    </div>
                `;
            }).join("");
        }
    },

    renderItems(filterText = "") {
        const container = document.getElementById("browser-items-container");
        if (!container) return;

        const term = filterText.toLowerCase();
        const filtered = this.items.filter(it => it.name.toLowerCase().includes(term));

        if (filtered.length === 0) {
            container.innerHTML = `
                <div style="color: var(--text-dim); padding: 30px; text-align: center; font-size: 0.85rem;">
                    ${term ? `No subdirectories matching "${filterText}".` : "No subdirectories found in this folder."}
                </div>
            `;
            return;
        }

        container.innerHTML = filtered.map(it => {
            const isSelected = this.selectedPath.toLowerCase() === it.path.toLowerCase();
            const icon = it.is_drive ? "💽" : "📁";
            const modelBadge = it.is_model_dir 
                ? `<span style="background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">✨ Models</span>` 
                : "";

            const freeBadge = it.is_drive && it.free_gb 
                ? `<span style="color: var(--text-dim); font-size: 0.74rem;">${it.free_gb} GB free / ${it.total_gb} GB</span>` 
                : "";

            const escapedPath = it.path.replace(/\\/g, '\\\\');

            return `
                <div class="browser-folder-item ${isSelected ? 'selected' : ''}" 
                     onclick="FolderBrowser.selectItem('${escapedPath}')"
                     ondblclick="FolderBrowser.navigate('${escapedPath}')">
                    <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; flex: 1;">
                        <span style="font-size: 1.15rem;">${icon}</span>
                        <div style="overflow: hidden; flex: 1;">
                            <div style="font-weight: 500; font-size: 0.84rem; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                ${it.name}
                            </div>
                            <div class="mono" style="font-size: 0.7rem; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                ${it.path}
                            </div>
                        </div>
                        ${modelBadge}
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        ${freeBadge}
                        <button type="button" class="btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" onclick="event.stopPropagation(); FolderBrowser.navigate('${escapedPath}')">
                            Open ➔
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    },

    selectItem(path) {
        this.selectedPath = path;
        this.updateSelectedPreview();

        // Highlight selected row in DOM
        document.querySelectorAll(".browser-folder-item").forEach(el => {
            const pathText = el.querySelector(".mono")?.textContent?.trim();
            el.classList.toggle("selected", pathText?.toLowerCase() === path.toLowerCase());
        });
    },

    updateSelectedPreview() {
        const preview = document.getElementById("browser-selected-preview");
        if (preview) {
            preview.textContent = this.selectedPath || this.currentPath || "(No folder selected)";
        }
    },

    confirmSelection() {
        const chosen = this.selectedPath || this.currentPath;
        if (!chosen) {
            showToast("Please click on a folder or drive to select it.", "info");
            return;
        }

        if (this.onSelectCallback) {
            this.onSelectCallback(chosen);
        } else if (this.targetInputId) {
            const input = document.getElementById(this.targetInputId);
            if (input) {
                input.value = chosen;
                // Dispatch input and change events so listeners react
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        showToast(`Selected: ${chosen}`, "success");
        this.close();

        // If selecting models dir, automatically trigger scan
        if (this.targetInputId === "cfg-models-dir") {
            SettingsManager.scanModels();
        }
    }
};

window.SettingsManager = SettingsManager;
window.FolderBrowser = FolderBrowser;
