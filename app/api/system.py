"""System, Hardware & Studio Presets API Router
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git
"""

from fastapi import APIRouter
from app.services.hardware_service import HardwareService
from app.core.config import settings

router = APIRouter(prefix="/api/system", tags=["System & Hardware"])

GENERAL_PRESETS = [
    {
        "id": "photoreal_portrait",
        "name": "Photorealistic Character & Portrait",
        "category": "Photorealism",
        "icon": "👤",
        "description": "Crisp human likeness, skin texture, studio lighting, and photorealistic focal depth.",
        "recommended_base": "flux-1-dev",
        "suggested_trigger": "ohwx_person",
        "default_rank": 32,
        "test_prompts": [
            "{trigger}, close up portrait, golden hour lighting, 85mm lens, sharp focus, hyperrealistic",
            "{trigger}, walking down a rainy city street at night, neon reflections, cinematic film still",
            "{trigger}, sitting in a minimalist modern cafe, natural window light, candid photography",
            "{trigger}, high fashion editorial photoshoot, studio backdrop, dramatic chiaroscuro lighting"
        ]
    },
    {
        "id": "anime_manga",
        "name": "Anime & Manga Style",
        "category": "Illustration",
        "icon": "✨",
        "description": "Vibrant Japanese anime aesthetics, clean cel shading, expressive linework, and keyframe colors.",
        "recommended_base": "sdxl-1.0",
        "suggested_trigger": "anime_style",
        "default_rank": 16,
        "test_prompts": [
            "{trigger}, heroic battle stance on a floating island, glowing magical aura, dynamic anime keyframe",
            "{trigger}, relaxed school rooftop scene, sakura petals floating in wind, makoto shinkai aesthetic",
            "{trigger}, cyberpunk pilot in high-tech cockpit, holographic HUD displays, retro-futuristic anime",
            "{trigger}, emotional close up with tearful smile, sunset backlight, detailed anime illustration"
        ]
    },
    {
        "id": "vintage_cartoon",
        "name": "Vintage 1930s & Classic Animation",
        "category": "Retro",
        "icon": "🎞️",
        "description": "1930s rubber-hose, classic theatrical shorts, aged film grain, and pie-eye aesthetics.",
        "recommended_base": "sdxl-1.0",
        "suggested_trigger": "rubberhose_style",
        "default_rank": 16,
        "test_prompts": [
            "{trigger}, whistling cheerfully at the ship helm, black and white 1930s cartoon, film scratches",
            "{trigger}, playing the trombone with rubbery bending limbs, vintage theatrical short, vignette",
            "{trigger}, panic reaction with eyes popping out, classic slapstick animation, ink and paint cel",
            "{trigger}, riding a vintage steam locomotive through rolling hills, sepia toned 1928 animation"
        ]
    },
    {
        "id": "stylized_3d",
        "name": "Stylized 3D & Animation Feature",
        "category": "3D Render",
        "icon": "🧸",
        "description": "Rich 3D character design, subsurface scattering, Pixar/DreamWorks feature aesthetic, and clay render.",
        "recommended_base": "flux-1-dev",
        "suggested_trigger": "stylized3d_char",
        "default_rank": 32,
        "test_prompts": [
            "{trigger}, holding a glowing crystal orb with curious expression, subsurface scattering, 3d render",
            "{trigger}, running through a magical enchanted forest, volumetric godrays, rendered in redshift",
            "{trigger}, character turnaround showcase, soft three-point studio lighting, octane render",
            "{trigger}, cozy winter outfit with wool texture, smiling warmly in falling snow, 3d animation still"
        ]
    },
    {
        "id": "environment_concept",
        "name": "Sci-Fi & Fantasy Concept Art",
        "category": "Environment",
        "icon": "🪐",
        "description": "Sweeping cinematic vistas, architectural designs, matte painting, and atmospheric worldbuilding.",
        "recommended_base": "flux-1-dev",
        "suggested_trigger": "concept_world",
        "default_rank": 32,
        "test_prompts": [
            "{trigger}, monumental alien citadel overlooking vast misty canyons, epic scale, concept art",
            "{trigger}, subterranean cyberpunk market with dense neon cables, volumetric smoke, cinematic matte",
            "{trigger}, ancient overgrown cathedral reclaimed by giant bioluminescent flora, twilight lighting",
            "{trigger}, orbital space station habitat rotating above an ocean planet, hard sci-fi realism"
        ]
    },
    {
        "id": "product_design",
        "name": "Industrial Product & Tech Design",
        "category": "Commercial",
        "icon": "⌚",
        "description": "Clean commercial product photography, sleek industrial materials, and luxury presentation.",
        "recommended_base": "sdxl-1.0",
        "suggested_trigger": "product_tech",
        "default_rank": 16,
        "test_prompts": [
            "{trigger}, hero shot on matte obsidian pedestal, diffuse rim lighting, luxury commercial photo",
            "{trigger}, exploded isometric view showcasing internal engineering components, clean studio render",
            "{trigger}, lifestyle presentation resting on minimalist wooden desk, warm ambient lighting",
            "{trigger}, outdoor waterproof splash action shot with water droplets suspended in mid-air"
        ]
    }
]

@router.get("/hardware")
async def get_hardware():
    """Get dynamic hardware profile, GPU VRAM tier, and adaptive scaling recommendations."""
    return HardwareService.get_hardware_profile()

@router.get("/presets")
async def get_presets():
    """Get generalized creative presets spanning photorealism, anime, 3D, retro, and concept art."""
    return {"presets": GENERAL_PRESETS}

@router.get("/info")
async def get_studio_info():
    """Get Geekatplay Studio branding, metadata, and repository links."""
    return {
        "app_name": settings.APP_NAME,
        "tagline": settings.APP_TAGLINE,
        "version": settings.VERSION,
        "author": settings.AUTHOR,
        "studio": settings.STUDIO,
        "repository": settings.REPOSITORY,
        "hardware_tier": HardwareService.get_hardware_profile()["tier_name"],
        "gpu_name": HardwareService.get_hardware_profile()["device_name"]
    }

from pydantic import BaseModel
from typing import Optional, List
from app.services.settings_service import SettingsService

class UpdateSettingsRequest(BaseModel):
    COMFYUI_ROOT: Optional[str] = None
    MODELS_DIR: Optional[str] = None
    EXTRA_MODEL_PATHS: Optional[List[str]] = None
    DOWNLOAD_DIR: Optional[str] = None
    KOHYA_ROOT: Optional[str] = None
    KOHYA_FALLBACK_ROOT: Optional[str] = None
    OLLAMA_HOST: Optional[str] = None

class ModelPathRequest(BaseModel):
    path: str

class ScanModelsRequest(BaseModel):
    models_dir: Optional[str] = None
    models_path: Optional[str] = None

@router.get("/settings")
def get_user_settings():
    """Retrieve user configured model folders, download paths, and drive storage stats."""
    return SettingsService.get_user_settings()

@router.post("/settings")
def save_user_settings(req: UpdateSettingsRequest):
    """Save user configured paths and redirect downloads/caches to protect primary drive."""
    return SettingsService.save_user_settings(req.model_dump(exclude_none=True))

@router.post("/settings/add_model_path")
def add_model_path(req: ModelPathRequest):
    """Add a new model directory across any connected drive to search paths."""
    return SettingsService.add_model_path(req.path)

@router.post("/settings/remove_model_path")
def remove_model_path(req: ModelPathRequest):
    """Remove a model directory from search paths."""
    return SettingsService.remove_model_path(req.path)

@router.post("/settings/scan")
def scan_models_dir(req: ScanModelsRequest):
    """Scan specified models directory or all configured folders and return detailed model inventory."""
    target_dir = req.models_dir or req.models_path
    return SettingsService.scan_models_directory(target_dir)

class BrowseDirectoryRequest(BaseModel):
    path: Optional[str] = None

@router.get("/browse")
def browse_fs_get(path: Optional[str] = None):
    """Interactive server-side directory navigator: lists drives or subfolders with breadcrumbs."""
    return SettingsService.browse_filesystem(path)

@router.post("/browse")
def browse_fs_post(req: BrowseDirectoryRequest):
    """Interactive server-side directory navigator (POST variant)."""
    return SettingsService.browse_filesystem(req.path)
