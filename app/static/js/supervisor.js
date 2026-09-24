// Supervisor Agent Interactive Studio Console
const Supervisor = {
    chatHistoryEl: null,
    chatInputEl: null,
    scriptPreviewEl: null,
    chatHistory: [],

    init() {
        this.chatHistoryEl = document.getElementById("chat-history");
        this.chatInputEl = document.getElementById("chat-input");
        this.scriptPreviewEl = document.getElementById("script-json-preview");

        document.getElementById("btn-send-chat")?.addEventListener("click", () => this.sendMessage());
        this.chatInputEl?.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Quick action prompt chips
        document.querySelectorAll(".prompt-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const text = chip.getAttribute("data-prompt") || chip.textContent.trim();
                this.sendMessage(text);
            });
        });

        document.getElementById("btn-copy-script")?.addEventListener("click", () => {
            if (this.scriptPreviewEl) {
                navigator.clipboard.writeText(this.scriptPreviewEl.textContent);
                showToast("Script JSON copied to clipboard!", "success");
            }
        });
    },

    async sendMessage(customText = null) {
        const text = customText || this.chatInputEl.value.trim();
        if (!text) return;

        if (!customText) this.chatInputEl.value = "";

        // Append user message
        this.appendMessage("user", text);
        this.chatHistory.push({ role: "user", content: text });

        // Show thinking indicator
        const typingId = "typing-" + Date.now();
        this.appendTypingIndicator(typingId);

        try {
            const res = await fetch("/api/supervisor/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    project_dir: AppState.projectDir,
                    history: this.chatHistory
                })
            });
            const data = await res.json();
            document.getElementById(typingId)?.remove();

            this.appendMessage("supervisor", data.reply);
            this.chatHistory.push({ role: "assistant", content: data.reply });

            // If script data returned, update script sidepanel
            if (data.data && data.tool_called === "generate_animation_script") {
                this.updateScriptSidepanel(data.data);
            }
        } catch (e) {
            document.getElementById(typingId)?.remove();
            this.appendMessage("supervisor", "Apologies, Director! A momentary static in the transmission: " + e.message);
        }
    },

    appendMessage(sender, text) {
        if (!this.chatHistoryEl) return;
        const msg = document.createElement("div");
        msg.className = `chat-message ${sender}`;

        const avatar = sender === "supervisor" ? "🎬" : "👤";
        msg.innerHTML = `
            <div class="msg-avatar">${avatar}</div>
            <div class="msg-bubble">${this.formatMarkdown(text)}</div>
        `;

        this.chatHistoryEl.appendChild(msg);
        this.chatHistoryEl.scrollTop = this.chatHistoryEl.scrollHeight;
    },

    appendTypingIndicator(id) {
        if (!this.chatHistoryEl) return;
        const msg = document.createElement("div");
        msg.className = "chat-message supervisor";
        msg.id = id;
        msg.innerHTML = `
            <div class="msg-avatar">🎬</div>
            <div class="msg-bubble" style="color: var(--amber-primary);">
                <em>Consulting specialized studio agents...</em>
            </div>
        `;
        this.chatHistoryEl.appendChild(msg);
        this.chatHistoryEl.scrollTop = this.chatHistoryEl.scrollHeight;
    },

    formatMarkdown(text) {
        // Simple safe markdown formatting (bold, backticks, linebreaks)
        let formatted = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/`([^`]+)`/g, "<code>$1</code>");
        return formatted;
    },

    updateScriptSidepanel(scriptData) {
        if (!this.scriptPreviewEl) return;
        this.scriptPreviewEl.textContent = JSON.stringify(scriptData, null, 2);
    }
};

window.Supervisor = Supervisor;
document.addEventListener("DOMContentLoaded", () => Supervisor.init());
