"""Hardware Auto-Detection & Adaptive Scaling Service
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git

Dynamically detects user GPU hardware, computes VRAM constraints,
and recommends optimal training & inference configurations across hardware tiers:
- Ultra Tier (24GB+): FLUX.1 Dev, SDXL Full, Hunyuan/Wan Video, rank 32-64, batch size 2-4
- High Tier (16GB - 23GB): FLUX.1 Schnell, SDXL, Wan Video, rank 16-32, batch size 1-2
- Mid Tier (10GB - 15GB): SDXL Optimized, rank 16, batch size 1, fp16 + gradient checkpointing
- Entry Tier (6GB - 9GB): SDXL Low-VRAM / SD1.5, rank 8, 8-bit AdamW, CPU offloading
- CPU / MPS Fallback: Minimal memory profile with advisory telemetry
"""

import os
import psutil
import torch
from typing import Dict, Any, Optional

class HardwareService:
    @staticmethod
    def get_hardware_profile() -> Dict[str, Any]:
        """Detect and return complete system hardware profile with intelligent tuning recommendations."""
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        # System RAM
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        available_ram_gb = round(vm.available / (1024 ** 3), 2)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False) or cpu_count

        if cuda_available:
            dev_idx = 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_vram_bytes = props.total_memory
            total_vram_gb = round(total_vram_bytes / (1024 ** 3), 2)
            
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
                free_vram_gb = round(free_bytes / (1024 ** 3), 2)
            except Exception:
                free_vram_gb = total_vram_gb

            compute_capability = f"{props.major}.{props.minor}"
            supports_bf16 = props.major >= 8  # Ampere (30xx), Ada (40xx), Hopper (H100)
            gpu_count = torch.cuda.device_count()
            backend = "cuda"
        elif mps_available:
            gpu_name = "Apple Silicon (MPS GPU)"
            total_vram_gb = round(total_ram_gb * 0.75, 2)  # Unified memory estimation
            free_vram_gb = round(available_ram_gb * 0.75, 2)
            compute_capability = "Apple Metal"
            supports_bf16 = True
            gpu_count = 1
            backend = "mps"
        else:
            gpu_name = "CPU Only (No discrete GPU detected)"
            total_vram_gb = 0.0
            free_vram_gb = 0.0
            compute_capability = "N/A"
            supports_bf16 = False
            gpu_count = 0
            backend = "cpu"

        # Determine Tier
        if total_vram_gb >= 23.0:
            tier_id = "ultra"
            tier_name = "Ultra Studio Tier (24GB+)"
            badge_color = "#10b981"  # Emerald
            recommended_models = ["flux-1-dev", "flux-1-schnell", "sdxl-1.0", "wan-2.1-t2v", "minimax-video", "qwen-image"]
            rec_batch_size = 2 if total_vram_gb >= 24.0 else 1
            rec_grad_accum = 1
            rec_rank = 32
            rec_optimizer = "AdamW"
            rec_mixed_precision = "bf16" if supports_bf16 else "fp16"
            rec_gradient_checkpointing = False
            rec_resolution = 1024
            can_train_flux = True
            can_train_video = True
            can_train_sdxl = True
        elif total_vram_gb >= 15.0:
            tier_id = "high"
            tier_name = "High Performance Tier (16GB - 23GB)"
            badge_color = "#3b82f6"  # Blue
            recommended_models = ["flux-1-schnell", "sdxl-1.0", "wan-2.1-turbo", "ltx-video", "qwen-image"]
            rec_batch_size = 1
            rec_grad_accum = 2
            rec_rank = 16
            rec_optimizer = "AdamW8bit" if not supports_bf16 else "AdamW"
            rec_mixed_precision = "bf16" if supports_bf16 else "fp16"
            rec_gradient_checkpointing = True
            rec_resolution = 1024
            can_train_flux = True
            can_train_video = True
            can_train_sdxl = True
        elif total_vram_gb >= 9.5:
            tier_id = "mid"
            tier_name = "Mid-Range Tier (10GB - 15GB)"
            badge_color = "#f59e0b"  # Amber
            recommended_models = ["sdxl-1.0", "ltx-video-turbo", "flux-1-schnell"]
            rec_batch_size = 1
            rec_grad_accum = 4
            rec_rank = 16
            rec_optimizer = "AdamW8bit"
            rec_mixed_precision = "fp16"
            rec_gradient_checkpointing = True
            rec_resolution = 1024
            can_train_flux = False  # requires extreme quantization
            can_train_video = False
            can_train_sdxl = True
        elif total_vram_gb >= 5.5:
            tier_id = "entry"
            tier_name = "Entry / Low-VRAM Tier (6GB - 9GB)"
            badge_color = "#ec4899"  # Pink
            recommended_models = ["sdxl-1.0"]
            rec_batch_size = 1
            rec_grad_accum = 4
            rec_rank = 8
            rec_optimizer = "AdamW8bit"
            rec_mixed_precision = "fp16"
            rec_gradient_checkpointing = True
            rec_resolution = 768
            can_train_flux = False
            can_train_video = False
            can_train_sdxl = True
        else:
            tier_id = "cpu_fallback"
            tier_name = "CPU / Minimal Memory Fallback (<6GB)"
            badge_color = "#ef4444"  # Red
            recommended_models = ["sdxl-1.0"]
            rec_batch_size = 1
            rec_grad_accum = 8
            rec_rank = 4
            rec_optimizer = "AdamW"
            rec_mixed_precision = "no"
            rec_gradient_checkpointing = True
            rec_resolution = 512
            can_train_flux = False
            can_train_video = False
            can_train_sdxl = False

        return {
            "backend": backend,
            "device_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
            "compute_capability": compute_capability,
            "supports_bf16": supports_bf16,
            "gpu_count": gpu_count,
            "system_ram_gb": total_ram_gb,
            "available_ram_gb": available_ram_gb,
            "cpu_cores_logical": cpu_count,
            "cpu_cores_physical": cpu_physical,
            "tier_id": tier_id,
            "tier_name": tier_name,
            "badge_color": badge_color,
            "recommendations": {
                "batch_size": rec_batch_size,
                "gradient_accumulation_steps": rec_grad_accum,
                "lora_rank": rec_rank,
                "optimizer": rec_optimizer,
                "mixed_precision": rec_mixed_precision,
                "gradient_checkpointing": rec_gradient_checkpointing,
                "resolution": rec_resolution,
                "recommended_models": recommended_models,
                "can_train_flux": can_train_flux,
                "can_train_video": can_train_video,
                "can_train_sdxl": can_train_sdxl
            }
        }
