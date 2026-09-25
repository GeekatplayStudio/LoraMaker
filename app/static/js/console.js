/**
 * Geekatplay Studio - Vladimir Chopine
 * Repository: https://github.com/GeekatplayStudio/LoraMaker.git
 * Diagnostics, Logging & Error Telemetry Console
 */

const DiagnosticsConsole = {
    modal: null,
    activeTab: 'errors',
    errors: [],
    logs: [],
    diagnosticsData: null,
    pollTimer: null,

    init() {
        this.modal = document.getElementById('diagnostics-console-modal');
        this.setupGlobalErrorCatchers();
        this.fetchDiagnostics(false);

        // Fetch logs periodically if modal is open
        setInterval(() => {
            if (this.modal && this.modal.classList.contains('active') && this.activeTab === 'logs') {
                this.fetchLogs();
            }
        }, 2500);
    },

    setupGlobalErrorCatchers() {
        // Intercept uncaught window errors
        window.addEventListener('error', (event) => {
            this.recordClientError('Client Exception', event.message, event.error?.stack || `${event.filename}:${event.lineno}:${event.colno}`);
        });

        // Intercept unhandled promise rejections (e.g. Failed to fetch)
        window.addEventListener('unhandledrejection', (event) => {
            const reason = event.reason;
            const message = reason?.message || String(reason);
            const stack = reason?.stack || 'No client stack trace';
            this.recordClientError('Unhandled Promise Rejection', message, stack);
        });
    },

    recordClientError(type, message, stack) {
        const errorItem = {
            id: 'client_' + Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            endpoint: window.location.pathname,
            method: 'CLIENT',
            status_code: 0,
            error_type: type,
            message: message,
            traceback: stack,
            suggestion: message.includes('Failed to fetch') 
                ? 'Network connection error. Ensure the Python backend is running on http://127.0.0.1:7860.'
                : 'Review client JavaScript error above.'
        };
        this.errors.unshift(errorItem);
        this.updateBadge();
        if (this.modal && this.modal.classList.contains('active') && this.activeTab === 'errors') {
            this.renderErrors();
        }
    },

    async fetchDiagnostics(showToastNotice = false) {
        try {
            const res = await fetch('/api/system/diagnostics');
            if (!res.ok) return;
            const data = await res.json();
            this.diagnosticsData = data;
            
            // Merge backend errors
            if (data.recent_errors && Array.isArray(data.recent_errors)) {
                this.errors = data.recent_errors;
            }
            if (data.recent_logs && Array.isArray(data.recent_logs)) {
                this.logs = data.recent_logs;
            }
            this.updateBadge();
            if (this.modal && this.modal.classList.contains('active')) {
                this.renderActiveTab();
            }
            if (showToastNotice && typeof showToast === 'function') {
                showToast('Diagnostics refreshed successfully.', 'success');
            }
        } catch (e) {
            console.warn('Diagnostics fetch failed:', e);
        }
    },

    async fetchLogs() {
        try {
            const res = await fetch('/api/system/logs?limit=100');
            if (!res.ok) return;
            const data = await res.json();
            if (data.logs) {
                this.logs = data.logs;
                this.renderLogs();
            }
        } catch (e) {
            console.warn('Failed to fetch logs:', e);
        }
    },

    updateBadge() {
        const badge = document.getElementById('console-error-badge');
        if (!badge) return;
        const count = this.errors.length;
        if (count > 0) {
            badge.textContent = count > 99 ? '99+' : count;
            badge.style.display = 'inline-block';
        } else {
            badge.style.display = 'none';
        }
    },

    openModal(tab = 'errors') {
        if (!this.modal) this.modal = document.getElementById('diagnostics-console-modal');
        if (!this.modal) return;
        this.modal.classList.add('active');
        this.switchTab(tab);
        this.fetchDiagnostics();
    },

    closeModal() {
        if (this.modal) {
            this.modal.classList.remove('active');
        }
    },

    switchTab(tab) {
        this.activeTab = tab;
        document.querySelectorAll('.diag-tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });
        document.querySelectorAll('.diag-tab-content').forEach(content => {
            content.style.display = content.id === `diag-panel-${tab}` ? 'block' : 'none';
        });
        this.renderActiveTab();
    },

    renderActiveTab() {
        if (this.activeTab === 'errors') this.renderErrors();
        else if (this.activeTab === 'logs') this.renderLogs();
        else if (this.activeTab === 'system') this.renderSystem();
    },

    renderErrors() {
        const container = document.getElementById('diag-errors-list');
        if (!container) return;

        if (!this.errors || this.errors.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-dim);">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">✨</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #34d399;">No Errors Recorded</div>
                    <div style="font-size: 0.8rem; margin-top: 4px;">System operations, endpoints, and training preflights are running cleanly.</div>
                </div>
            `;
            return;
        }

        container.innerHTML = this.errors.map((err, idx) => {
            const statusClass = (err.status_code >= 500 || err.status_code === 0) ? 'status-500' : 'status-400';
            const statusText = err.status_code === 0 ? 'NETWORK / CLIENT' : `HTTP ${err.status_code}`;
            const method = err.method || 'POST';
            const endpoint = err.endpoint || 'Unknown endpoint';
            const errorType = err.error_type || 'Error';
            const message = err.message || 'Unknown error occurred';
            const suggestion = err.suggestion || '';
            const traceback = err.traceback || '';
            const payloadStr = err.request_payload ? JSON.stringify(err.request_payload, null, 2) : '';

            return `
                <div class="diag-error-card" id="err-card-${idx}">
                    <div class="diag-error-header">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <span class="diag-status-pill ${statusClass}">${statusText}</span>
                            <span class="mono" style="font-weight: 600; color: #fde68a;">${method} ${endpoint}</span>
                            <span class="mono" style="font-size: 0.75rem; color: var(--text-dim);">${err.timestamp || ''}</span>
                        </div>
                        <div style="display: flex; gap: 6px;">
                            <button class="btn-ghost" style="padding: 3px 8px; font-size: 0.75rem;" onclick="DiagnosticsConsole.copyErrorItem(${idx})" title="Copy this specific error to clipboard">
                                📋 Copy
                            </button>
                        </div>
                    </div>
                    
                    <div class="diag-error-title">
                        <span style="color: #ef4444; font-weight: 700;">${errorType}:</span> ${this.escapeHtml(message)}
                    </div>

                    ${suggestion ? `
                        <div class="diag-error-suggestion">
                            💡 <strong>Actionable Suggestion:</strong> ${this.escapeHtml(suggestion)}
                        </div>
                    ` : ''}

                    <div style="margin-top: 10px; display: flex; gap: 10px;">
                        ${traceback ? `
                            <details style="flex: 1;">
                                <summary style="font-size: 0.78rem; cursor: pointer; color: var(--cyan-accent); font-weight: 600;">
                                    📜 View Python Stack Trace / Traceback
                                </summary>
                                <pre class="diag-traceback mono">${this.escapeHtml(traceback)}</pre>
                            </details>
                        ` : ''}

                        ${payloadStr ? `
                            <details style="flex: 1;">
                                <summary style="font-size: 0.78rem; cursor: pointer; color: #a78bfa; font-weight: 600;">
                                    📦 View Request Payload
                                </summary>
                                <pre class="diag-traceback mono">${this.escapeHtml(payloadStr)}</pre>
                            </details>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
    },

    renderLogs() {
        const viewer = document.getElementById('diag-logs-viewer');
        if (!viewer) return;

        if (!this.logs || this.logs.length === 0) {
            viewer.innerHTML = '<div style="color: var(--text-dim); padding: 10px;">No server logs recorded yet.</div>';
            return;
        }

        viewer.innerHTML = this.logs.map(log => {
            const level = (log.level || 'INFO').toUpperCase();
            let color = '#94a3b8';
            if (level === 'ERROR' || level === 'CRITICAL') color = '#ef4444';
            else if (level === 'WARNING') color = '#fbbf24';
            else if (level === 'INFO') color = '#38bdf8';

            return `
                <div style="line-height: 1.4; padding: 2px 0;">
                    <span style="color: var(--text-dim);">${log.timestamp || ''}</span>
                    <span style="color: ${color}; font-weight: 600; margin: 0 4px;">[${level}]</span>
                    <span style="color: #cbd5e1;">${this.escapeHtml(log.message || '')}</span>
                </div>
            `;
        }).join('');
        viewer.scrollTop = viewer.scrollHeight;
    },

    renderSystem() {
        const container = document.getElementById('diag-system-container');
        if (!container || !this.diagnosticsData) return;

        const d = this.diagnosticsData;
        const hw = d.hardware || {};
        const env = d.environment || {};
        const st = d.storage || {};

        container.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">
                <!-- Card 1: Hardware & Compute -->
                <div class="glass-panel" style="padding: 16px;">
                    <h3 style="font-size: 0.9rem; color: #fde68a; margin-bottom: 8px;">🎮 Compute & GPU Hardware</h3>
                    <div style="font-size: 0.8rem; line-height: 1.6;">
                        <div><strong>Device:</strong> <span class="mono">${hw.device_name || 'CPU Only'}</span></div>
                        <div><strong>CUDA Available:</strong> <span style="color: ${hw.cuda_available ? '#34d399' : '#ef4444'}; font-weight: 600;">${hw.cuda_available ? 'Yes (Active)' : 'No'}</span></div>
                        <div><strong>VRAM:</strong> <span class="mono">${hw.total_vram_gb || 0} GB (${hw.tier_name || 'Standard'})</span></div>
                    </div>
                </div>

                <!-- Card 2: Runtime Environment -->
                <div class="glass-panel" style="padding: 16px;">
                    <h3 style="font-size: 0.9rem; color: var(--cyan-accent); margin-bottom: 8px;">🐍 Python & AI Packages</h3>
                    <div style="font-size: 0.8rem; line-height: 1.6;">
                        <div><strong>Python:</strong> <span class="mono">${env.python || 'Unknown'} (${env.os || ''})</span></div>
                        <div><strong>PyTorch:</strong> <span class="mono">${env.torch || 'Not installed'}</span></div>
                        <div><strong>Diffusers:</strong> <span class="mono">${env.diffusers || 'Not installed'}</span></div>
                        <div><strong>Transformers:</strong> <span class="mono">${env.transformers || 'Not installed'}</span></div>
                        <div><strong>Safetensors:</strong> <span class="mono">${env.safetensors || 'Not installed'}</span></div>
                    </div>
                </div>

                <!-- Card 3: Storage & Model Folders -->
                <div class="glass-panel" style="padding: 16px; grid-column: 1 / -1;">
                    <h3 style="font-size: 0.9rem; color: #a78bfa; margin-bottom: 8px;">💾 Storage Locations & Model Roots</h3>
                    <div style="font-size: 0.8rem; line-height: 1.6;">
                        <div><strong>Primary Models Directory:</strong> <code class="mono" style="color: #fde68a;">${st.models_dir || 'Not set'}</code></div>
                        <div><strong>Safe Download Directory:</strong> <code class="mono" style="color: #34d399;">${st.download_dir || 'Not set'}</code></div>
                        <div style="margin-top: 6px;"><strong>Extra Model Search Folders (${(st.extra_model_paths || []).length}):</strong></div>
                        <ul style="margin: 4px 0 0 16px; padding: 0;">
                            ${(st.extra_model_paths || []).map(p => `<li><code class="mono">${p}</code></li>`).join('') || '<li style="color: var(--text-dim);">No extra paths configured</li>'}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    },

    async clearErrors() {
        try {
            await fetch('/api/system/diagnostics/clear', { method: 'POST' });
            this.errors = [];
            this.updateBadge();
            this.renderErrors();
            if (typeof showToast === 'function') showToast('Error history cleared.', 'info');
        } catch (e) {
            console.warn('Failed to clear errors:', e);
        }
    },

    async copyDiagnosticReport() {
        await this.fetchDiagnostics(false);
        const d = this.diagnosticsData || {};
        const hw = d.hardware || {};
        const env = d.environment || {};
        const st = d.storage || {};
        const activeProj = window.AppState?.projectDir || 'None selected';
        const timestamp = new Date().toISOString();

        let report = `### 🛠️ Geekatplay LoRA Maker - System Diagnostic Report\n\n`;
        report += `- **Report Timestamp:** ${timestamp}\n`;
        report += `- **Studio:** ${d.studio || 'Geekatplay Studio - Vladimir Chopine'}\n`;
        report += `- **App Version:** ${d.version || '1.0.0'}\n`;
        report += `- **Active Project:** \`${activeProj}\`\n\n`;

        report += `#### 🎮 Hardware & Compute Specs\n`;
        report += `- **GPU Device:** ${hw.device_name || 'CPU'}\n`;
        report += `- **CUDA Available:** ${hw.cuda_available ? 'Yes' : 'No'}\n`;
        report += `- **VRAM Tier:** ${hw.total_vram_gb || 0} GB (${hw.tier_name || 'Standard'})\n`;
        report += `- **Python Version:** ${env.python || 'Unknown'} on ${env.platform || 'Windows'}\n`;
        report += `- **PyTorch:** ${env.torch || 'N/A'} (CUDA: ${env.torch_cuda ? 'Yes' : 'No'})\n`;
        report += `- **Diffusers:** ${env.diffusers || 'N/A'}\n`;
        report += `- **Transformers:** ${env.transformers || 'N/A'}\n`;
        report += `- **Safetensors:** ${env.safetensors || 'N/A'}\n\n`;

        report += `#### 💾 Storage & Configured Model Directories\n`;
        report += `- **Primary Models Dir:** \`${st.models_dir || 'N/A'}\`\n`;
        report += `- **Safe Download Dir:** \`${st.download_dir || 'N/A'}\`\n`;
        if (st.extra_model_paths && st.extra_model_paths.length > 0) {
            report += `- **Extra Model Paths:**\n` + st.extra_model_paths.map(p => `  - \`${p}\``).join('\n') + `\n`;
        }

        report += `\n#### ❌ Recent Errors & Exceptions (${this.errors.length})\n`;
        if (this.errors.length === 0) {
            report += `_No errors recorded. All preflight checks and pipelines completed cleanly._\n`;
        } else {
            this.errors.slice(0, 5).forEach((err, idx) => {
                report += `\n**[Error #${idx + 1}] ${err.method || 'POST'} ${err.endpoint || ''} (${err.status_code === 0 ? 'Network Error' : 'HTTP ' + err.status_code})**\n`;
                report += `- **Timestamp:** ${err.timestamp}\n`;
                report += `- **Error Type:** \`${err.error_type}\`\n`;
                report += `- **Message:** ${err.message}\n`;
                if (err.suggestion) report += `- **Suggestion:** ${err.suggestion}\n`;
                if (err.request_payload) {
                    report += `\n**Request Payload:**\n\`\`\`json\n${JSON.stringify(err.request_payload, null, 2)}\n\`\`\`\n`;
                }
                if (err.traceback && err.traceback.length > 5) {
                    report += `\n**Python Stack Trace:**\n\`\`\`\n${err.traceback}\n\`\`\`\n`;
                }
            });
        }

        if (this.logs && this.logs.length > 0) {
            report += `\n#### 📜 Recent Server Engine Logs (Last ${Math.min(25, this.logs.length)} lines)\n\`\`\`\n`;
            report += this.logs.slice(-25).map(l => `[${l.timestamp}] [${l.level}] ${l.message}`).join('\n');
            report += `\n\`\`\`\n`;
        }

        this.copyToClipboard(report, 'Full diagnostic report copied to clipboard! You can now paste this directly into chat.');
    },

    copyErrorItem(idx) {
        const err = this.errors[idx];
        if (!err) return;
        let text = `### ❌ Error in ${err.method || 'POST'} ${err.endpoint || ''}\n`;
        text += `- **Status:** ${err.status_code === 0 ? 'Network/Client Error' : 'HTTP ' + err.status_code}\n`;
        text += `- **Type:** \`${err.error_type}\`\n`;
        text += `- **Message:** ${err.message}\n`;
        if (err.suggestion) text += `- **Suggestion:** ${err.suggestion}\n`;
        if (err.request_payload) {
            text += `\n**Payload:**\n\`\`\`json\n${JSON.stringify(err.request_payload, null, 2)}\n\`\`\`\n`;
        }
        if (err.traceback) {
            text += `\n**Traceback:**\n\`\`\`\n${err.traceback}\n\`\`\`\n`;
        }
        this.copyToClipboard(text, 'Error snippet copied to clipboard!');
    },

    copyToClipboard(text, successMsg) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => {
                if (typeof showToast === 'function') showToast(`📋 ${successMsg}`, 'success');
            }).catch(() => this.fallbackCopy(text, successMsg));
        } else {
            this.fallbackCopy(text, successMsg);
        }
    },

    fallbackCopy(text, successMsg) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            if (typeof showToast === 'function') showToast(`📋 ${successMsg}`, 'success');
        } catch (e) {
            alert('Failed to copy to clipboard. Please copy manually from the console.');
        }
        document.body.removeChild(ta);
    },

    /**
     * Render an inline error card directly in a page panel (e.g. Training Tab)
     */
    renderInlineError(containerId, errorObj) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const errType = errorObj.error_type || 'Error';
        const msg = errorObj.message || errorObj.detail || 'An unexpected error occurred';
        const suggestion = errorObj.suggestion || 'Click below to view full details or copy the report.';
        const status = errorObj.status_code ? `HTTP ${errorObj.status_code}` : 'Network / Connection Error';

        container.style.display = 'block';
        container.innerHTML = `
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: var(--radius-md); padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                            <span style="font-size: 1.1rem;">⚠️</span>
                            <span style="font-weight: 700; color: #fca5a5; font-size: 0.95rem;">Training Pipeline Encountered an Error (${status})</span>
                        </div>
                        <div style="color: #fff; font-size: 0.88rem; margin-bottom: 6px;">
                            <strong style="color: #ef4444;">${this.escapeHtml(errType)}:</strong> ${this.escapeHtml(msg)}
                        </div>
                        <div style="color: #fde68a; font-size: 0.8rem; line-height: 1.4;">
                            💡 <strong>Recommendation:</strong> ${this.escapeHtml(suggestion)}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <button class="btn-primary" onclick="DiagnosticsConsole.copyDiagnosticReport()" style="padding: 6px 14px; font-size: 0.8rem; background: #ef4444; border-color: #dc2626;" title="Copy complete error report with traceback to clipboard">
                            📋 Copy Full Diagnostic Report
                        </button>
                        <button class="btn-secondary" onclick="DiagnosticsConsole.openModal('errors')" style="padding: 6px 12px; font-size: 0.8rem;" title="Open console to inspect Python traceback">
                            🖥️ Open Console
                        </button>
                    </div>
                </div>
            </div>
        `;
    },

    clearInlineError(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.style.display = 'none';
            container.innerHTML = '';
        }
    },

    escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
};

window.DiagnosticsConsole = DiagnosticsConsole;
document.addEventListener('DOMContentLoaded', () => DiagnosticsConsole.init());
