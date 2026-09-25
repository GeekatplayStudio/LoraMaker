"""Geekatplay LoRA Maker Application Launcher
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git
"""

import sys
import time
import argparse
import webbrowser
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure Windows terminal doesn't crash on emoji characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.core.config import settings
from app.services.hardware_service import HardwareService

def main():
    parser = argparse.ArgumentParser(description=f"Launch {settings.APP_NAME}")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    hw = HardwareService.get_hardware_profile()

    print("=" * 72)
    print(f"  ⚡ {settings.APP_NAME} v{settings.VERSION}")
    print(f"  🎨 {settings.APP_TAGLINE}")
    print(f"  👤 Author: {settings.AUTHOR} ({settings.STUDIO})")
    print(f"  ⭐ Repository: {settings.REPOSITORY}")
    print("-" * 72)
    print(f"  💻 Hardware: {hw['device_name']} ({hw['total_vram_gb']}GB VRAM) [{hw['tier_name']}]")
    print(f"  🧠 System RAM: {hw['system_ram_gb']}GB | CPU Cores: {hw['cpu_cores_logical']}")
    print(f"  ⚙️  Ollama Host: {settings.OLLAMA_HOST}")
    print(f"  🌐 Studio UI: {url}")
    print("=" * 72)

    if not args.no_browser:
        import threading
        def open_browser():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
