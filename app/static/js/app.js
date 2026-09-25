// Global Application State & Controller
const AppState = {
    currentProject: null,
    projectDir: null,
    videoPath: null,
    videoInfo: null,
    currentFrameIndex: 0,
    totalFrames: 0,
    fps: 24.0,
    isPlaying: false,
    bestVisionModel: "qwen2.5vl:7b"
};
window.AppState = AppState;

// Global helper: Show toast message
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = "toast";
    const icon = type === "success" ? "✅" : (type === "error" ? "❌" : "✨");
    const consoleBtn = type === "error"
        ? `<button type="button" onclick="if(window.DiagnosticsConsole) DiagnosticsConsole.openModal();" style="background: rgba(239, 68, 68, 0.25); border: 1px solid rgba(239, 68, 68, 0.5); color: #fca5a5; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; cursor: pointer; margin-left: 8px; font-weight: 600; white-space: nowrap;">🔍 View Console</button>`
        : '';
    toast.innerHTML = `<span style="font-size: 1rem;">${icon}</span> <span style="flex: 1; word-break: break-word;">${message}</span>${consoleBtn}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100%)";
        setTimeout(() => toast.remove(), 300);
    }, type === "error" ? 6500 : 3500);
}

// Navigation Tab Switcher
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".workspace-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });

    if (tabId === "gallery") {
        if (window.Gallery) window.Gallery.refresh();
    } else if (tabId === "training") {
        if (window.Training) window.Training.audit();
    } else if (tabId === "evaluation") {
        if (window.Evaluation) window.Evaluation.load();
    }
}

// Global initialization
document.addEventListener("DOMContentLoaded", async () => {
    // Tab event listeners
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
    });

    // Check Ollama and available models
    try {
        const res = await fetch("/api/captions/models");
        const data = await res.json();
        if (data.best_vision_model) {
            AppState.bestVisionModel = data.best_vision_model;
            const modelBadge = document.getElementById("model-indicator-text");
            if (modelBadge) modelBadge.textContent = `Ollama: ${data.best_vision_model}`;

            // Populate vision model select
            const modelSelect = document.getElementById("vision-model-select");
            if (modelSelect && data.vision_models) {
                modelSelect.innerHTML = "";
                data.vision_models.forEach(m => {
                    const opt = document.createElement("option");
                    opt.value = m.name;
                    opt.textContent = `${m.name} (${m.size_gb} GB)`;
                    if (m.name === data.best_vision_model) opt.selected = true;
                    modelSelect.appendChild(opt);
                });
            }
        }
    } catch (e) {
        console.warn("Could not query Ollama models:", e);
    }

    // Load projects list
    await loadProjectsList();

    // Setup global keyboard shortcuts
    window.addEventListener("keydown", (e) => {
        // Ignore if user is typing in input or textarea
        if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
            return;
        }

        if (e.key === "ArrowLeft") {
            e.preventDefault();
            if (e.shiftKey) {
                if (window.Scrubber) window.Scrubber.stepFrames(-10);
            } else {
                if (window.Scrubber) window.Scrubber.stepFrames(-1);
            }
        } else if (e.key === "ArrowRight") {
            e.preventDefault();
            if (e.shiftKey) {
                if (window.Scrubber) window.Scrubber.stepFrames(10);
            } else {
                if (window.Scrubber) window.Scrubber.stepFrames(1);
            }
        } else if (e.code === "Space") {
            e.preventDefault();
            if (window.Scrubber) window.Scrubber.togglePlay();
        } else if (e.key.toLowerCase() === "c") {
            e.preventDefault();
            if (window.Scrubber) window.Scrubber.cutCurrentFrame();
        }
    });

    // Setup New Project Modal
    const newProjBtn = document.getElementById("btn-new-project");
    const modal = document.getElementById("new-project-modal");
    const cancelModal = document.getElementById("btn-cancel-modal");
    const formModal = document.getElementById("form-new-project");

    if (newProjBtn && modal) {
        newProjBtn.addEventListener("click", () => modal.classList.add("active"));
    }
    if (cancelModal && modal) {
        cancelModal.addEventListener("click", () => modal.classList.remove("active"));
    }

    // Creative Presets Handling
    const presetChips = document.querySelectorAll(".preset-chip");
    const presetsMap = {
        "photoreal_portrait": {
            proj: "Portrait_Elena",
            char: "Elena",
            trigger: "elena_portrait",
            desc: "photorealistic human portrait, natural studio lighting, 85mm lens, sharp focus, hyperrealistic"
        },
        "anime_manga": {
            proj: "Anime_Hero_Reel",
            char: "Ren",
            trigger: "ren_anime",
            desc: "vibrant anime illustration, clean cel shading, expressive linework, makoto shinkai aesthetic"
        },
        "stylized_3d": {
            proj: "Stylized3D_Milo",
            char: "Milo",
            trigger: "milo_3d",
            desc: "stylized 3D character render, subsurface scattering, pixar feature aesthetic, soft three-point studio lighting"
        },
        "vintage_cartoon": {
            proj: "Bendy_Vintage_Reel",
            char: "BendyBot",
            trigger: "BendyBot",
            desc: "vintage 1930s rubber hose cel animation, monochrome ink lines, fleischer studio style"
        },
        "environment_concept": {
            proj: "Concept_Citadel",
            char: "Aethelgard",
            trigger: "aethelgard_citadel",
            desc: "monumental alien citadel, sweeping misty canyons, sci-fi matte painting, cinematic lighting, 8k uhd"
        },
        "product_design": {
            proj: "Product_ChronosWatch",
            char: "ChronosWatch",
            trigger: "chronos_watch",
            desc: "luxury product commercial photography, obsidian pedestal, softbox rim lighting, clean industrial design"
        }
    };

    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            presetChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            const key = chip.getAttribute("data-preset");
            const p = presetsMap[key];
            if (p) {
                const elProj = document.getElementById("modal-proj-name");
                const elChar = document.getElementById("modal-char-name");
                const elTrig = document.getElementById("modal-trigger-token");
                const elStyle = document.getElementById("modal-style-desc");
                if (elProj) elProj.value = p.proj;
                if (elChar) elChar.value = p.char;
                if (elTrig) elTrig.value = p.trigger;
                if (elStyle) elStyle.value = p.desc;
            }
        });
    });

    if (formModal) {
        formModal.addEventListener("submit", async (e) => {
            e.preventDefault();
            const projName = document.getElementById("modal-proj-name").value.trim();
            const charName = document.getElementById("modal-char-name").value.trim();
            const triggerToken = (document.getElementById("modal-trigger-token")?.value || charName).trim();
            const styleDesc = document.getElementById("modal-style-desc").value.trim();

            try {
                const res = await fetch("/api/projects/init", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        project_name: projName,
                        character_name: charName,
                        trigger_token: triggerToken,
                        style_description: styleDesc
                    })
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`Project '${projName}' initialized successfully!`, "success");
                    modal.classList.remove("active");
                    await loadProjectsList(data.project.folder_structure.Input_Video.replace(/[\\/]Input_Video$/, ""));
                }
            } catch (err) {
                showToast("Failed to initialize project: " + err, "error");
            }
        });
    }

    // Hardware Auto-Detection & Modal
    initHardwareProfile();

    // About Geekatplay Studio Modal
    initAboutModal();

    // System Storage & Paths Settings Modal
    if (window.SettingsManager) {
        window.SettingsManager.init();
    }
});

// Initialize Hardware Auto-Detection
async function initHardwareProfile() {
    const badgeText = document.getElementById("hw-badge-text");
    const badge = document.getElementById("hw-status-badge");
    const hwModal = document.getElementById("hardware-modal");
    const closeHwBtn = document.getElementById("btn-close-hw-modal");

    try {
        const res = await fetch("/api/system/hardware");
        const hw = await res.json();
        AppState.hardware = hw;

        if (badgeText) {
            badgeText.textContent = `${hw.device_name.replace("NVIDIA GeForce ", "")} (${hw.total_vram_gb}GB) • ${hw.tier_name.split(" ")[0]} Tier`;
        }

        // Fill hardware modal fields
        const elGpu = document.getElementById("hw-modal-gpu-name");
        const elTier = document.getElementById("hw-modal-tier-badge");
        const elVram = document.getElementById("hw-modal-vram");
        const elRam = document.getElementById("hw-modal-ram");
        const elRecBatch = document.getElementById("hw-rec-batch");
        const elRecPrec = document.getElementById("hw-rec-prec");
        const elRecModels = document.getElementById("hw-rec-models");

        if (elGpu) elGpu.textContent = hw.device_name;
        if (elTier) {
            elTier.textContent = hw.tier_name;
            elTier.style.color = hw.badge_color;
            elTier.style.borderColor = hw.badge_color;
        }
        if (elVram) elVram.textContent = `${hw.total_vram_gb} GB (${hw.free_vram_gb} GB Free)`;
        if (elRam) elRam.textContent = `${hw.system_ram_gb} GB (${hw.available_ram_gb} GB Available)`;
        if (elRecBatch) elRecBatch.textContent = `${hw.recommendations.batch_size} (Grad Accum: ${hw.recommendations.gradient_accumulation_steps})`;
        if (elRecPrec) elRecPrec.textContent = `${hw.recommendations.mixed_precision.toUpperCase()} (Optimizer: ${hw.recommendations.optimizer})`;
        if (elRecModels) elRecModels.textContent = hw.recommendations.recommended_models.join(", ");

        if (badge && hwModal) {
            badge.addEventListener("click", () => hwModal.classList.add("active"));
        }
        if (closeHwBtn && hwModal) {
            closeHwBtn.addEventListener("click", () => hwModal.classList.remove("active"));
        }
    } catch (e) {
        console.warn("Could not query hardware profile:", e);
        if (badgeText) badgeText.textContent = "Hardware: Ready";
    }
}

// Initialize About Geekatplay Studio Modal
function initAboutModal() {
    const aboutBtn = document.getElementById("btn-about");
    const aboutModal = document.getElementById("about-modal");
    const closeAboutBtn = document.getElementById("btn-close-about-modal");

    if (aboutBtn && aboutModal) {
        aboutBtn.addEventListener("click", () => aboutModal.classList.add("active"));
    }
    if (closeAboutBtn && aboutModal) {
        closeAboutBtn.addEventListener("click", () => aboutModal.classList.remove("active"));
    }
}

// Load projects list
async function loadProjectsList(selectPath = null) {
    try {
        const res = await fetch("/api/projects/list");
        const data = await res.json();
        const select = document.getElementById("project-selector");
        if (!select) return;

        select.innerHTML = '<option value="">-- Select Active Studio Project --</option>';
        data.projects.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.project_dir;
            opt.textContent = `${p.project_name} (${p.character_name}) [${p.total_frames} frames]`;
            if (selectPath && p.project_dir === selectPath) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });

        // Auto select first if available and none selected
        if (!selectPath && data.projects.length > 0) {
            select.value = data.projects[0].project_dir;
            setActiveProject(data.projects[0].project_dir);
        } else if (selectPath) {
            setActiveProject(selectPath);
        }

        select.addEventListener("change", (e) => {
            if (e.target.value) {
                setActiveProject(e.target.value);
            }
        });
    } catch (e) {
        console.warn("Could not load projects list:", e);
    }
}

// Set active project
async function setActiveProject(projectDir) {
    AppState.projectDir = projectDir;
    try {
        const res = await fetch(`/api/projects/info?project_dir=${encodeURIComponent(projectDir)}`);
        const data = await res.json();
        AppState.currentProject = data.metadata;

        const charBadge = document.getElementById("char-indicator-text");
        if (charBadge && data.metadata) {
            charBadge.textContent = `Character: [${data.metadata.trigger_token || data.metadata.character_name}]`;
        }

        // Fill default token inputs in scrubber
        const tokenInput = document.getElementById("char-token-input");
        if (tokenInput && data.metadata) {
            tokenInput.value = data.metadata.trigger_token || "BendyBot";
        }
        const styleInput = document.getElementById("style-token-input");
        if (styleInput && data.metadata) {
            styleInput.value = data.metadata.style_token || "vintage 1930s rubber hose cel animation";
        }

        showToast(`Loaded project: ${data.metadata ? data.metadata.project_name : 'Project'}`, "success");

        if (window.Gallery) window.Gallery.refresh();
        if (window.Training) window.Training.audit();
        if (window.Evaluation) window.Evaluation.load();
    } catch (e) {
        console.warn("Could not set active project:", e);
    }
}
