// Video Timeline Scrubber & Cel Frame Cutter Controller
const Scrubber = {
    videoEl: null,
    sliderEl: null,
    timecodeEl: null,
    frameCounterEl: null,
    playBtnEl: null,
    isSeeking: false,

    init() {
        this.videoEl = document.getElementById("main-video");
        this.sliderEl = document.getElementById("timeline-slider");
        this.timecodeEl = document.getElementById("timecode-readout");
        this.frameCounterEl = document.getElementById("frame-counter-readout");
        this.playBtnEl = document.getElementById("btn-play-pause");

        if (!this.videoEl || !this.sliderEl) return;

        // Slider scrubbing
        this.sliderEl.addEventListener("input", (e) => {
            if (!AppState.videoInfo) return;
            const targetFrame = parseInt(e.target.value);
            this.seekToFrame(targetFrame);
        });

        // Video timeupdate sync
        this.videoEl.addEventListener("timeupdate", () => {
            if (this.isSeeking || !AppState.videoInfo) return;
            const currentTime = this.videoEl.currentTime;
            const frame = Math.round(currentTime * AppState.fps);
            AppState.currentFrameIndex = frame;
            this.sliderEl.value = frame;
            this.updateReadouts(frame, currentTime);
        });

        // Video ended
        this.videoEl.addEventListener("ended", () => {
            AppState.isPlaying = false;
            this.updatePlayButton();
        });

        // Stepping buttons
        document.getElementById("btn-step-back-10")?.addEventListener("click", () => this.stepFrames(-10));
        document.getElementById("btn-step-back-1")?.addEventListener("click", () => this.stepFrames(-1));
        document.getElementById("btn-step-fwd-1")?.addEventListener("click", () => this.stepFrames(1));
        document.getElementById("btn-step-fwd-10")?.addEventListener("click", () => this.stepFrames(10));
        this.playBtnEl?.addEventListener("click", () => this.togglePlay());

        // Cut Frame button
        document.getElementById("btn-cut-frame")?.addEventListener("click", () => this.cutCurrentFrame());

        // Video File Upload / Path input
        const uploadInput = document.getElementById("video-file-input");
        if (uploadInput) {
            uploadInput.addEventListener("change", (e) => this.handleVideoUpload(e.target.files[0]));
        }

        const loadPathBtn = document.getElementById("btn-load-video-path");
        if (loadPathBtn) {
            loadPathBtn.addEventListener("click", () => {
                const pathInput = document.getElementById("video-path-input");
                if (pathInput && pathInput.value.trim()) {
                    this.loadVideoFromPath(pathInput.value.trim());
                }
            });
        }
    },

    async loadVideoFromPath(path) {
        try {
            showToast("Probing video reel...", "info");
            const res = await fetch(`/api/video/info?video_path=${encodeURIComponent(path)}`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to load video");
            }
            const info = await res.json();
            AppState.videoPath = path;
            AppState.videoInfo = info;
            AppState.fps = info.fps || 24.0;
            AppState.totalFrames = info.total_frames || 1;
            AppState.currentFrameIndex = 0;

            // Set video source
            this.videoEl.src = `/api/video/stream?video_path=${encodeURIComponent(path)}`;
            this.sliderEl.max = Math.max(0, AppState.totalFrames - 1);
            this.sliderEl.value = 0;

            // Update UI elements
            document.getElementById("video-placeholder")?.style.setProperty("display", "none");
            this.videoEl.style.display = "block";
            this.updateReadouts(0, 0);

            showToast(`Loaded reel: ${info.file_name} (${info.width}x${info.height} @ ${info.fps}fps)`, "success");
        } catch (e) {
            showToast("Video load failed: " + e.message, "error");
        }
    },

    async handleVideoUpload(file) {
        if (!file) return;
        if (!AppState.projectDir) {
            showToast("Please select or initialize a project first!", "error");
            return;
        }

        const formData = new FormData();
        formData.append("project_dir", AppState.projectDir);
        formData.append("file", file);

        try {
            showToast("Uploading video scan into Input_Video...", "info");
            const res = await fetch("/api/video/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                showToast("Video uploaded successfully!", "success");
                await this.loadVideoFromPath(data.saved_path);
            }
        } catch (e) {
            showToast("Upload failed: " + e.message, "error");
        }
    },

    seekToFrame(frameIndex) {
        if (!AppState.videoInfo) return;
        this.isSeeking = true;
        const clamped = Math.max(0, Math.min(frameIndex, AppState.totalFrames - 1));
        AppState.currentFrameIndex = clamped;
        const targetTime = clamped / AppState.fps;
        this.videoEl.currentTime = targetTime;
        this.sliderEl.value = clamped;
        this.updateReadouts(clamped, targetTime);
        setTimeout(() => { this.isSeeking = false; }, 50);
    },

    stepFrames(delta) {
        if (!AppState.videoInfo) return;
        if (AppState.isPlaying) {
            this.videoEl.pause();
            AppState.isPlaying = false;
            this.updatePlayButton();
        }
        this.seekToFrame(AppState.currentFrameIndex + delta);
    },

    togglePlay() {
        if (!AppState.videoInfo) return;
        if (AppState.isPlaying) {
            this.videoEl.pause();
            AppState.isPlaying = false;
        } else {
            this.videoEl.play();
            AppState.isPlaying = true;
        }
        this.updatePlayButton();
    },

    updatePlayButton() {
        if (!this.playBtnEl) return;
        this.playBtnEl.innerHTML = AppState.isPlaying ? "⏸" : "▶";
    },

    updateReadouts(frame, seconds) {
        if (this.frameCounterEl) {
            const pad = (n, len = 4) => String(n).padStart(len, "0");
            this.frameCounterEl.textContent = `Frame ${pad(frame)} / ${pad(AppState.totalFrames)}`;
        }
        if (this.timecodeEl) {
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            const ms = Math.floor((seconds - Math.floor(seconds)) * 1000);
            const pad = (n, len = 2) => String(n).padStart(len, "0");
            this.timecodeEl.textContent = `${pad(m)}:${pad(s)}.${pad(ms, 3)}`;
        }
    },

    async cutCurrentFrame() {
        if (!AppState.videoPath) {
            showToast("Please load a video reel first!", "error");
            return;
        }
        if (!AppState.projectDir) {
            showToast("Please select or initialize an active project!", "error");
            return;
        }

        const charToken = document.getElementById("char-token-input")?.value.trim() || "BendyBot";
        const styleToken = document.getElementById("style-token-input")?.value.trim() || "vintage 1930s rubber hose cel animation";
        const autoCaption = document.getElementById("auto-caption-check")?.checked ?? true;
        const visionModel = document.getElementById("vision-model-select")?.value || AppState.bestVisionModel;

        const btn = document.getElementById("btn-cut-frame");
        const origText = btn.innerHTML;
        btn.innerHTML = "<span>⏳ Cutting & Scanning...</span>";
        btn.disabled = true;

        try {
            showToast(`Extracting Cel Frame #${AppState.currentFrameIndex}...`, "info");
            const res = await fetch("/api/video/cut_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_path: AppState.videoPath,
                    frame_index: AppState.currentFrameIndex,
                    project_dir: AppState.projectDir,
                    character_token: charToken,
                    style_token: styleToken,
                    auto_caption: autoCaption,
                    vision_model: visionModel
                })
            });

            const data = await res.json();
            if (data.success) {
                const fname = data.frame.file_name;
                const captionSnip = data.caption && data.caption.caption ? data.caption.caption.slice(0, 60) + "..." : "Saved without caption";
                showToast(`Cut ${fname}! Caption: "${captionSnip}"`, "success");

                // Trigger gallery update
                if (window.Gallery) window.Gallery.refresh();
            } else {
                throw new Error(data.detail || "Cut failed");
            }
        } catch (e) {
            showToast("Failed to cut frame: " + e.message, "error");
        } finally {
            btn.innerHTML = origText;
            btn.disabled = false;
        }
    }
};

window.Scrubber = Scrubber;
document.addEventListener("DOMContentLoaded", () => Scrubber.init());
