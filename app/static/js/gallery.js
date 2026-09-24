// Keyframe Gallery, Caption Editor, and Dataset Exporter
const Gallery = {
    gridEl: null,
    totalFramesEl: null,
    captionedFramesEl: null,
    uncaptionedFramesEl: null,
    completionPctEl: null,

    init() {
        this.gridEl = document.getElementById("keyframes-grid");
        this.totalFramesEl = document.getElementById("stat-total-frames");
        this.captionedFramesEl = document.getElementById("stat-captioned-frames");
        this.uncaptionedFramesEl = document.getElementById("stat-uncaptioned-frames");
        this.completionPctEl = document.getElementById("stat-completion-pct");

        document.getElementById("btn-refresh-gallery")?.addEventListener("click", () => this.refresh());
        document.getElementById("btn-batch-caption")?.addEventListener("click", () => this.batchCaption());
        document.getElementById("btn-export-kohya")?.addEventListener("click", () => this.exportKohya());
        document.getElementById("btn-export-comfyui")?.addEventListener("click", () => this.exportComfyUI());
    },

    async refresh() {
        if (!AppState.projectDir) return;

        try {
            const res = await fetch(`/api/datasets/frames?project_dir=${encodeURIComponent(AppState.projectDir)}`);
            const data = await res.json();

            // Update stats
            if (this.totalFramesEl) this.totalFramesEl.textContent = data.total_count;
            if (this.captionedFramesEl) this.captionedFramesEl.textContent = data.captioned_count;
            if (this.uncaptionedFramesEl) this.uncaptionedFramesEl.textContent = data.uncaptioned_count;
            if (this.completionPctEl) this.completionPctEl.textContent = `${data.completion_percentage}%`;

            this.renderFrames(data.frames);
        } catch (e) {
            console.warn("Could not refresh keyframes gallery:", e);
        }
    },

    renderFrames(frames) {
        if (!this.gridEl) return;

        if (!frames || frames.length === 0) {
            this.gridEl.innerHTML = `
                <div style="grid-column: 1 / -1; padding: 60px; text-align: center; color: var(--text-dim);" class="glass-panel">
                    <p style="font-size: 2.2rem; margin-bottom: 12px;">🎞️</p>
                    <h3>No Cel Frames In Vault Yet</h3>
                    <p style="margin-top: 6px; font-size: 0.9rem;">Switch to the <strong>Timeline Scrubber</strong> tab, find great character poses, and click <strong>Cut Cel Frame</strong>!</p>
                </div>
            `;
            return;
        }

        this.gridEl.innerHTML = "";
        frames.forEach(f => {
            const card = document.createElement("div");
            card.className = "keyframe-card glass-panel";
            card.id = `card-${f.file_name.replace(/[^a-zA-Z0-9]/g, '_')}`;

            const imgSrc = `/api/datasets/image?image_path=${encodeURIComponent(f.file_path)}`;

            card.innerHTML = `
                <div class="keyframe-thumb">
                    <img src="${imgSrc}" alt="${f.file_name}" loading="lazy" />
                </div>
                <div class="keyframe-meta">
                    <span class="mono" style="color: #38bdf8; font-weight: 500;">${f.file_name}</span>
                    <span>${f.file_size_kb} KB</span>
                </div>
                <div class="form-group">
                    <label class="form-label" style="display: flex; justify-content: space-between;">
                        <span>LoRA Training Caption (.txt)</span>
                        <span style="color: ${f.has_caption ? 'var(--emerald-accent)' : 'var(--amber-primary)'}; font-size: 0.75rem;">
                            ${f.has_caption ? '● Captioned' : '○ Needs Caption'}
                        </span>
                    </label>
                    <textarea class="form-textarea caption-editor-area" id="caption-txt-${f.file_name}">${f.caption || ""}</textarea>
                </div>
                <div class="card-actions">
                    <div style="display: flex; gap: 8px;">
                        <button class="btn-primary" style="padding: 6px 12px; font-size: 0.8rem;" onclick="Gallery.saveCaption('${encodeURIComponent(f.file_path)}', '${f.file_name}')">
                            💾 Save
                        </button>
                        <button class="btn-secondary" style="padding: 6px 12px; font-size: 0.8rem;" onclick="Gallery.recaption('${encodeURIComponent(f.file_path)}')">
                            ✨ Ollama
                        </button>
                    </div>
                    <button class="btn-secondary" style="padding: 6px 10px; font-size: 0.8rem; color: var(--rose-accent);" onclick="Gallery.deleteFrame('${encodeURIComponent(f.file_path)}')">
                        🗑️
                    </button>
                </div>
            `;
            this.gridEl.appendChild(card);
        });
    },

    async saveCaption(encodedPath, fileName) {
        const path = decodeURIComponent(encodedPath);
        const textarea = document.getElementById(`caption-txt-${fileName}`);
        if (!textarea) return;
        const newCaption = textarea.value.trim();

        try {
            const res = await fetch("/api/datasets/caption", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_path: path,
                    caption: newCaption
                })
            });
            const data = await res.json();
            if (data.success) {
                showToast(`Caption saved for ${fileName}!`, "success");
                this.refresh();
            }
        } catch (e) {
            showToast("Failed to save caption: " + e.message, "error");
        }
    },

    async recaption(encodedPath) {
        const path = decodeURIComponent(encodedPath);
        const charToken = document.getElementById("char-token-input")?.value.trim() || "BendyBot";
        const styleToken = document.getElementById("style-token-input")?.value.trim() || "vintage 1930s rubber hose cel animation";
        const visionModel = document.getElementById("vision-model-select")?.value || AppState.bestVisionModel;

        showToast("Generating Ollama caption...", "info");
        try {
            const res = await fetch("/api/captions/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_path: path,
                    trigger_token: charToken,
                    style_token: styleToken,
                    model_name: visionModel
                })
            });
            const data = await res.json();
            if (data.caption) {
                showToast("Ollama caption updated!", "success");
                this.refresh();
            }
        } catch (e) {
            showToast("Ollama captioning failed: " + e.message, "error");
        }
    },

    async deleteFrame(encodedPath) {
        const path = decodeURIComponent(encodedPath);
        if (!confirm("Are you sure you want to delete this frame and its caption?")) return;

        try {
            const res = await fetch("/api/datasets/frame", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_path: path })
            });
            const data = await res.json();
            if (data.success) {
                showToast("Frame removed from vault", "info");
                this.refresh();
            }
        } catch (e) {
            showToast("Failed to delete frame: " + e.message, "error");
        }
    },

    async batchCaption() {
        if (!AppState.projectDir) return;
        const visionModel = document.getElementById("vision-model-select")?.value || AppState.bestVisionModel;

        showToast(`Running batch Ollama captioning with ${visionModel}...`, "info");
        try {
            const res = await fetch("/api/captions/batch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    model_name: visionModel,
                    force_overwrite: false
                })
            });
            const data = await res.json();
            showToast(`Batch completed: captioned ${data.processed_count} frames!`, "success");
            this.refresh();
        } catch (e) {
            showToast("Batch captioning failed: " + e.message, "error");
        }
    },

    async exportKohya() {
        if (!AppState.projectDir) return;
        try {
            showToast("Exporting Kohya_ss folder hierarchy...", "info");
            const res = await fetch("/api/datasets/export/kohya", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir,
                    repeats: 10,
                    class_token: "character"
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast(`Exported ${data.exported_frames} frames to ${data.concept_folder}!`, "success");
            } else {
                showToast("Export error: " + data.detail, "error");
            }
        } catch (e) {
            showToast("Kohya export failed: " + e.message, "error");
        }
    },

    async exportComfyUI() {
        if (!AppState.projectDir) return;
        try {
            showToast("Exporting ComfyUI dataset with metadata.jsonl...", "info");
            const res = await fetch("/api/datasets/export/comfyui", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: AppState.projectDir
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast(`Exported ${data.exported_frames} frames for ComfyUI!`, "success");
            } else {
                showToast("ComfyUI export error: " + data.detail, "error");
            }
        } catch (e) {
            showToast("ComfyUI export failed: " + e.message, "error");
        }
    }
};

window.Gallery = Gallery;
document.addEventListener("DOMContentLoaded", () => Gallery.init());
