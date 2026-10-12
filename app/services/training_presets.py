"""Verified training presets.

Every preset is a *step-budgeted* recipe: the number of optimizer steps is fixed
up-front so run time is predictable no matter how many keyframes a project has.
Presets are resolved against the detected hardware tier and the dataset size,
which yields the exact values the trainer receives (and that are written into the
run manifest for auditing).

Design rules that keep runs fast and reliable:
- Latents and text-encoder outputs are cached once, so the VAE / CLIP never run
  per step.
- Only the UNet / transformer is trained; text encoders are frozen.
- A hard ``max_train_steps`` cap replaces epoch guessing.
- Resolution never exceeds what the hardware tier can hold.
- 8-bit AdamW is only requested when ``bitsandbytes`` is actually importable by
  the trainer interpreter; otherwise plain AdamW is used so the run cannot fail
  on a missing optional dependency.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Hard preset table.  ``steps`` is the optimizer-step budget.  ``epochs`` is the
# number of passes used to derive per-image repeats (and the number of save
# points the trainer will produce when it supports epoch saves).
PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke_test": {
        "name": "Smoke test (verify trainer, ~2 min)",
        "description": (
            "Runs 20 real optimizer steps at 512px with a rank-4 adapter and writes a "
            "loadable LoRA. Use it first to prove the pipeline trains on this machine."
        ),
        "steps": 20,
        "epochs": 1,
        "resolution": 512,
        "rank": 4,
        "alpha": 4,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "estimated_minutes": {"ultra": 1, "high": 2, "mid": 3, "entry": 4},
    },
    "fast": {
        "name": "Fast character (768px, ~400 steps)",
        "description": (
            "Rank 8 at 768px with cached latents. Good likeness for characters with "
            "15-40 keyframes in roughly 10-15 minutes on a 12 GB card."
        ),
        "steps": 400,
        "epochs": 4,
        "resolution": 768,
        "rank": 8,
        "alpha": 8,
        "learning_rate": 1.5e-4,
        "batch_size": 1,
        "estimated_minutes": {"ultra": 5, "high": 8, "mid": 14, "entry": 25},
    },
    "balanced": {
        "name": "Balanced (1024px, ~900 steps)",
        "description": "Rank 16 at 1024px. The default for most character and style LoRAs.",
        "steps": 900,
        "epochs": 6,
        "resolution": 1024,
        "rank": 16,
        "alpha": 8,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "estimated_minutes": {"ultra": 12, "high": 20, "mid": 35, "entry": 60},
    },
    "quality": {
        "name": "Quality (1024px, ~1600 steps)",
        "description": "Rank 32 at 1024px with a lower learning rate for detailed styles and large datasets.",
        "steps": 1600,
        "epochs": 8,
        "resolution": 1024,
        "rank": 32,
        "alpha": 16,
        "learning_rate": 8e-5,
        "batch_size": 1,
        "estimated_minutes": {"ultra": 22, "high": 38, "mid": 65, "entry": 110},
    },
}

DEFAULT_PRESET = "fast"

# Per-tier resolution ceilings.  Training above these is the most common cause of
# CUDA out-of-memory failures.
_TIER_MAX_RESOLUTION = {
    "ultra": 1024,
    "high": 1024,
    "mid": 1024,
    "entry": 768,
    "cpu_fallback": 512,
}


def list_presets() -> List[Dict[str, Any]]:
    return [
        {"id": key, **{k: v for k, v in spec.items()}}
        for key, spec in PRESETS.items()
    ]


def resolve_preset(
    preset_id: str,
    hardware: Dict[str, Any],
    dataset_frame_count: int,
    base_model: str = "sdxl-1.0",
    bitsandbytes_available: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn a preset id into concrete trainer parameters.

    ``overrides`` may contain any of rank, alpha, learning_rate, resolution,
    max_train_steps, batch_size.  Only keys with a non-None value are applied.
    """
    if preset_id not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_id}'. Choose one of: {', '.join(PRESETS)}")
    spec = PRESETS[preset_id]
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    tier = hardware.get("tier_id", "mid")
    vram = float(hardware.get("total_vram_gb") or 0.0)
    supports_bf16 = bool(hardware.get("supports_bf16"))

    resolution = int(overrides.get("resolution", spec["resolution"]))
    max_res = _TIER_MAX_RESOLUTION.get(tier, 1024)
    resolution = max(64, min(resolution, max_res))
    resolution -= resolution % 64

    max_steps = int(overrides.get("max_train_steps", spec["steps"]))
    max_steps = max(1, max_steps)
    batch_size = max(1, int(overrides.get("batch_size", spec["batch_size"])))

    frames = max(1, int(dataset_frame_count))
    epochs = max(1, int(spec["epochs"]))
    # Choose repeats so that frames * repeats * epochs covers the step budget.
    repeats = max(1, math.ceil(max_steps * batch_size / (frames * epochs)))
    repeats = min(repeats, 50)

    rank = max(1, int(overrides.get("rank", spec["rank"])))
    alpha = max(1, int(overrides.get("alpha", spec["alpha"])))
    learning_rate = float(overrides.get("learning_rate", spec["learning_rate"]))

    is_flux = "flux" in base_model
    if is_flux:
        # Flux is heavier: rank stays, resolution is capped at 768 under 24 GB.
        if vram < 23.0:
            resolution = min(resolution, 768)
        learning_rate = min(learning_rate, 1e-4)

    mixed_precision = "bf16" if supports_bf16 else "fp16"
    if tier == "cpu_fallback":
        mixed_precision = "no"

    # Plain AdamW is faster and simpler when memory allows; 8-bit only when the
    # dependency is really there.
    if vram >= 23.0 or tier == "cpu_fallback":
        optimizer = "AdamW"
    else:
        optimizer = "AdamW8bit" if bitsandbytes_available else "AdamW"

    gradient_checkpointing = tier not in {"ultra"} or is_flux

    est = spec["estimated_minutes"].get(tier)
    return {
        "preset": preset_id,
        "preset_name": spec["name"],
        "resolution": resolution,
        "max_train_steps": max_steps,
        "epochs": epochs,
        "repeats": repeats,
        "batch_size": batch_size,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "learning_rate": learning_rate,
        "mixed_precision": mixed_precision,
        "optimizer_type": optimizer,
        "gradient_checkpointing": gradient_checkpointing,
        "cache_latents": True,
        "train_unet_only": True,
        "lr_scheduler": "constant",
        "estimated_minutes": est,
        "tier_id": tier,
        "dataset_frame_count": frames,
        "planned_samples": frames * repeats * epochs,
    }
