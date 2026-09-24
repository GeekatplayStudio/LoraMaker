import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

class AppSettings(BaseModel):
    APP_NAME: str = "Geekatplay LoRA Maker"
    APP_TAGLINE: str = "Universal LoRA Training, Asset Extraction & Creative Studio"
    VERSION: str = "2.0.0"
    AUTHOR: str = "Vladimir Chopine"
    STUDIO: str = "Geekatplay Studio"
    REPOSITORY: str = "https://github.com/GeekatplayStudio/LoraMaker.git"
    HOST: str = "127.0.0.1"
    PORT: int = 7860
    
    # Base workspace directory (dynamically resolved, no personal paths)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DEFAULT_PROJECTS_DIR: Path = BASE_DIR / "projects"
    
    # Required Standard Folder Structure (as specified in rec.txt)
    STANDARD_FOLDERS: List[str] = [
        "Input_Video",
        "Keyframes_Out",
        "Prompts",
        "Assets",
        "Training",
        "Output_Video"
    ]
    
    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Preferred Vision Models (in rank priority)
    PREFERRED_VISION_MODELS: List[str] = [
        "qwen2.5vl:7b",
        "qwen3-vl:4b",
        "llama3.2-vision:latest",
        "llava:latest",
        "moondream:latest",
        "minicpm-v:latest"
    ]
    
    # Preferred LLM models for Supervisor and Story Agent
    PREFERRED_LLM_MODELS: List[str] = [
        "qwen2.5-coder:14b",
        "qwen3:8b",
        "gemma3:latest",
        "gemma3:4b",
        "llama3.1:8b",
        "llama3.2:3b"
    ]
    
    # Supported LoRA base architectures (Image, Video, and Turbo/Distilled variants)
    SUPPORTED_LORA_ARCHITECTURES: List[dict] = [
        # Image Models
        {
            "id": "flux-1-dev",
            "name": "FLUX.1 [dev]",
            "category": "image",
            "is_turbo": False,
            "description": "State-of-the-art 12B rectified flow transformer for ultra-detailed imagery and typography.",
            "default_dim": 16,
            "default_alpha": 16,
            "resolution": 1024,
            "trainer_support": ["kohya_ss", "ai-toolkit", "diffusers"]
        },
        {
            "id": "flux-1-schnell",
            "name": "FLUX.1 [schnell] ⚡ Turbo",
            "category": "image",
            "is_turbo": True,
            "description": "4-step distilled rectified flow model for ultra-fast character generation.",
            "default_dim": 16,
            "default_alpha": 16,
            "resolution": 1024,
            "trainer_support": ["kohya_ss", "ai-toolkit", "diffusers"]
        },
        {
            "id": "qwen-image",
            "name": "Qwen Image (DiT / Qwen2.5-VL)",
            "category": "image",
            "is_turbo": False,
            "description": "Alibaba Qwen multimodal vision transformer with unmatched prompt compliance and character consistency.",
            "default_dim": 16,
            "default_alpha": 16,
            "resolution": 1024,
            "trainer_support": ["diffusers", "kohya_ss", "comfyui"]
        },
        {
            "id": "z-image",
            "name": "Z-Image (Zhipu / Zero-terminal DiT)",
            "category": "image",
            "is_turbo": False,
            "description": "High-fidelity stylized cel & rubber-hose diffusion transformer architecture.",
            "default_dim": 32,
            "default_alpha": 16,
            "resolution": 1024,
            "trainer_support": ["kohya_ss", "comfyui", "diffusers"]
        },
        {
            "id": "sdxl-1.0",
            "name": "Stable Diffusion XL 1.0",
            "category": "image",
            "is_turbo": False,
            "description": "High-compatibility base model ideal for vintage rubber-hose and 1930s cel art.",
            "default_dim": 32,
            "default_alpha": 16,
            "resolution": 1024,
            "trainer_support": ["kohya_ss", "comfyui", "diffusers"]
        },
        # Video Models (including Turbo/Distilled)
        {
            "id": "minimax-video",
            "name": "MiniMax / HunyuanVideo (Hailuo)",
            "category": "video",
            "is_turbo": False,
            "description": "High-fidelity cinematic animation motion diffusion with dynamic cartoon physics and squash/stretch.",
            "default_dim": 32,
            "default_alpha": 32,
            "resolution": 720,
            "trainer_support": ["comfyui", "diffusers"]
        },
        {
            "id": "wan-2.1-t2v",
            "name": "Wan2.1 (Video Diffusion 14B)",
            "category": "video",
            "is_turbo": False,
            "description": "Cutting-edge open video model with motion brushes and camera trajectory controls.",
            "default_dim": 32,
            "default_alpha": 32,
            "resolution": 720,
            "trainer_support": ["kohya_ss", "comfyui", "diffusers"]
        },
        {
            "id": "wan-2.1-turbo",
            "name": "Wan2.1 ⚡ Turbo (4-8 Step)",
            "category": "video",
            "is_turbo": True,
            "description": "Few-step distilled Wan2.1 video diffusion for rapid motion iteration and animation tests.",
            "default_dim": 32,
            "default_alpha": 32,
            "resolution": 720,
            "trainer_support": ["kohya_ss", "comfyui"]
        },
        {
            "id": "ltx-video",
            "name": "LTX-Video (Lightricks)",
            "category": "video",
            "is_turbo": False,
            "description": "High-speed video generation model for 2-4 second motion clips.",
            "default_dim": 16,
            "default_alpha": 16,
            "resolution": 768,
            "trainer_support": ["diffusers", "comfyui"]
        },
        {
            "id": "ltx-video-turbo",
            "name": "LTX-Video ⚡ Turbo (Real-Time)",
            "category": "video",
            "is_turbo": True,
            "description": "Real-time 8-step motion diffusion, ultra-low VRAM footprint (<12GB).",
            "default_dim": 16,
            "default_alpha": 16,
            "resolution": 768,
            "trainer_support": ["diffusers", "comfyui"]
        },
        {
            "id": "cogvideox-5b",
            "name": "CogVideoX-5B",
            "category": "video",
            "is_turbo": False,
            "description": "THUDM temporal diffusion architecture with high frame coherence for animated character cycles.",
            "default_dim": 32,
            "default_alpha": 32,
            "resolution": 720,
            "trainer_support": ["diffusers", "comfyui"]
        }
    ]

settings = AppSettings()
