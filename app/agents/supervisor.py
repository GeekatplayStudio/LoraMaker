import json
import logging
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.vision_service import VisionService
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.agents.visual_agent import VisualAgent
from app.agents.video_agent import VideoAgent
from app.agents.story_agent import StoryAgent
from app.agents.trainer_agent import TrainerAgent
import ollama

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are the Orchestrator for "Geekatplay LoRA Maker" (created by Vladimir Chopine / Geekatplay Studio: https://github.com/GeekatplayStudio/LoraMaker.git).
Geekatplay LoRA Maker is a universal desktop AI studio designed to assist creators in training high-performance LoRAs across any visual subject or aesthetic—including:
- Photorealistic human likeness, portraits, and cinema stills
- Stylized 3D character design and subsurface scattering
- Anime, manga, and modern illustrative keyframes
- Vintage animation (1930s rubber-hose, classic theatrical cel)
- Sci-Fi, fantasy, and architectural concept environments
- Commercial product photography and industrial design

Your primary role is to coordinate the specialized agents (Video Agent, Visual Agent, Story Agent, Trainer Agent) and guide the creator efficiently based on their specific hardware capabilities.

Core Responsibilities:
1. **Project & Subject Initialization**:
    - Guide users in selecting a project folder, character or concept name, and unique trigger token.
    - Identify the creative goal: Is it a person, character, art style, product, or cinematic world?
    - Help users choose the best base model (FLUX.1, SDXL, Wan2.1, Qwen) according to their detected GPU VRAM tier.

2. **Dataset & Caption Engineering**:
    - Coordinate keyframe extraction from video footage or import of image folders.
    - Use Ollama multimodal vision models (e.g. Qwen2.5-VL, LLaVA) to generate rich, descriptive, trigger-prefixed captions.
    - Validate aspect ratios to ensure images are bucketed or fitted with zero distortion, stretching, or squeezing.

3. **Hardware-Adaptive Training Guidance**:
    - Monitor GPU VRAM and recommend optimal batch sizes, gradient accumulation, and optimizer type (AdamW8bit vs AdamW).
    - Guide users to train locally via real PyTorch CUDA or export clean configurations to Kohya_ss and ComfyUI.

Operational Guidelines:
- Be encouraging, knowledgeable, and creative. Assume the user has an artistic vision but may need technical guidance.
- Adapt tone to the user's genre (photorealism, animation, concept art, etc.).
- Always emphasize clean trigger words and non-distorted aspect ratios."""

class SupervisorAgent:
    """
    Supervisor Agent / Orchestrator for Geekatplay LoRA Maker.
    Directs specialized agents, manages studio project workflows, and interacts with user.
    """

    @classmethod
    def get_available_models(cls) -> Dict[str, Any]:
        """
        Tool: Checks which AI models are currently supported, installed locally, and recommends
        the best options based on auto-detected hardware VRAM.
        """
        local_ollama = VisionService.list_installed_models()
        best_vision = VisionService.get_best_vision_model()
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        capabilities = TrainingService.get_training_capabilities()
        available_ids = {item["id"] for item in capabilities["architectures"] if item["available"]}
        available_models = [item for item in settings.SUPPORTED_LORA_ARCHITECTURES if item["id"] in available_ids]
        
        return {
            "studio": "Geekatplay Studio",
            "author": "Vladimir Chopine",
            "repository": "https://github.com/GeekatplayStudio/LoraMaker.git",
            "hardware": hw,
            "best_vision_model": best_vision,
            "installed_ollama_models": local_ollama,
            "supported_lora_architectures": available_models,
            "training_capabilities": capabilities,
            "recommended_image_model": "flux-1-dev" if "flux-1-dev" in available_ids else ("sdxl-1.0" if "sdxl-1.0" in available_ids else None),
            "recommended_video_model": None
        }

    @classmethod
    def start_new_project(
        cls,
        project_dir: str,
        project_name: str,
        character_name: str,
        style_description: str = "vintage 1930s rubber hose cel animation"
    ) -> Dict[str, Any]:
        """
        Tool: Initializes project folder structure and metadata.
        """
        meta = DatasetService.initialize_project(
            project_dir=project_dir,
            project_name=project_name,
            character_name=character_name,
            style_description=style_description
        )

        explanation = (
            f"Salutations, Animator! Project '{project_name}' has been established for our star '{character_name}'.\n"
            f"Here is your retro animation studio folder layout:\n"
            f"📁 Input_Video: Place raw cartoon reels or motion reference scans here.\n"
            f"📁 Keyframes_Out: High-resolution cel frame scans will be cut and collected here.\n"
            f"📁 Prompts: Storyboards and animation scripts.\n"
            f"📁 Assets: Model checkpoints, triggers, and turnaround reference sheets.\n"
            f"📁 Training: LoRA configuration and fine-tuned weights (.safetensors).\n"
            f"📁 Output_Video: Rendered vintage motion diffusion reels."
        )

        return {
            "meta": meta,
            "explanation": explanation
        }

    @classmethod
    def analyze_input_video(cls, video_path: str) -> Dict[str, Any]:
        """
        Tool: Delegates video analysis to Video and Visual Agents.
        """
        video_analysis = VideoAgent.analyze_source_video(video_path)
        pose_candidates = VisualAgent.analyze_video_poses(video_path)

        return {
            "video_specs": video_analysis,
            "suggested_keyframes": pose_candidates,
            "recommendation": "Use the Timeline Scrubber to navigate to these keyframe moments, inspect the cel scans, and click 'Cut Cel Frame' to build your character LoRA dataset."
        }

    @classmethod
    def process_chat_message(
        cls,
        user_message: str,
        project_dir: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Processes a user message to the Supervisor. Evaluates tools and replies with retro studio flair.
        """
        msg_lower = user_message.lower()

        # Intent: Model discovery
        if "model" in msg_lower or "support" in msg_lower or "wan" in msg_lower or "flux" in msg_lower or "ollama" in msg_lower:
            models_info = cls.get_available_models()
            best_v = models_info["best_vision_model"]
            return {
                "reply": (
                    f"Welcome to the Model Vault, Director! 🎬\n\n"
                    f"✨ **Active Vision Scanner (Ollama)**: `{best_v}` is primed and ready to caption your cel frame scans with rich LoRA triggers.\n\n"
                    f"🎨 **Visual Image LoRAs Supported**:\n"
                    f"- **FLUX.1 [dev]**: Highest fidelity for intricate ink linework & expressive character designs.\n"
                    f"- **Qwen Image (DiT / Qwen2.5-VL)**: Alibaba multimodal vision transformer with unmatched prompt compliance & character identity.\n"
                    f"- **Z-Image DiT**: High-resolution stylized cel linework and rubber-hose character adapter.\n"
                    f"- **SDXL 1.0**: Perfect vintage 1930s rubber-hose cel aesthetics and high flexibility.\n\n"
                    f"🎞️ **Video Motion LoRAs Supported**:\n"
                    f"- **Wan2.1 (Video Diffusion)**: Outstanding 24fps motion brushes & dynamic camera panning.\n"
                    f"- **LTX-Video**: Lightning-fast 2-4 second motion clip generation.\n"
                    f"- **MiniMax / HunyuanVideo**: High cinematic cartoon physics and motion diffusion."
                ),
                "tool_called": "get_available_models",
                "data": models_info
            }

        # Intent: Character Setup Status
        if "status" in msg_lower or "ready" in msg_lower or "dataset" in msg_lower:
            if project_dir:
                audit = TrainerAgent.audit_dataset_readiness(project_dir)
                status_str = "READY FOR LORA TRAINING! 🚀" if audit["is_ready_for_training"] else "WORK IN PROGRESS ⏳"
                return {
                    "reply": (
                        f"**Studio Character Setup Audit**: {status_str}\n\n"
                        f"- Total Cel Scans Cut: **{audit['total_frames']}**\n"
                        f"- Captioned Cel Scans: **{audit['captioned_frames']}**\n"
                        f"- Pending Captions: **{audit['uncaptioned_frames']}**\n\n"
                        + ("\n".join(f"⚠️ {i}" for i in audit["issues"]) if audit["issues"] else "✅ All keyframe scans are captioned with triggers!")
                    ),
                    "tool_called": "audit_dataset_readiness",
                    "data": audit
                }
            else:
                return {
                    "reply": "Kindly select or create a project first so I can inspect the cel frame scans in your `Keyframes_Out` vault!"
                }

        # Intent: LoRA Testing, Stats, Inspection, Metrics, and Suggested Prompts
        if any(w in msg_lower for w in ["test", "stat", "metric", "inspect", "eval", "prompt", "validate", "diagnos"]):
            if project_dir:
                from app.services.evaluation_service import EvaluationService
                inspect_res = EvaluationService.inspect_lora_model(project_dir)
                analytics_res = EvaluationService.get_training_analytics(project_dir)
                prompts_res = EvaluationService.generate_suggested_prompts(project_dir)

                if inspect_res.get("success"):
                    sample_p = prompts_res["prompts"][0]["positive"] if prompts_res.get("prompts") else ""
                    return {
                        "reply": (
                            f"🔬 **LoRA Inspection & Diagnostics Report**\n\n"
                            f"- **Model**: `{inspect_res['model_name']}` ({inspect_res['file_size_kb']} KB)\n"
                            f"- **Architecture**: LoRA Rank={inspect_res['rank']}, Alpha={inspect_res['alpha']}, Scale={inspect_res['scale']}\n"
                            f"- **Weight Parameters**: {inspect_res['total_parameters']:,} params across {inspect_res['tensor_count']} tensors\n"
                            f"- **Weight Delta Magnitude**: `{inspect_res['delta_weight_magnitude']}` (Active feature adaptation)\n"
                            f"- **Health Status**: {inspect_res['health_status']}\n"
                            f"- **Loss Reduction**: {analytics_res['initial_loss']} ➔ {analytics_res['final_loss']} (**-{analytics_res['loss_reduction_percent']}% convergence**)\n"
                            f"- **ComfyUI Ready**: {'✅ Deployed to models/loras' if inspect_res['comfyui_deployed'] else 'Ready to deploy with 1-click'}\n\n"
                            f"🎯 **Suggested Verification Prompt**:\n"
                            f"> `{sample_p}`\n\n"
                            f"Head over to the **Model Diagnostics & Testing Lab** tab to run interactive test renders!"
                        ),
                        "tool_called": "inspect_lora_model",
                        "data": {
                            "inspection": inspect_res,
                            "analytics": analytics_res,
                            "suggested_prompts": prompts_res.get("prompts", [])
                        }
                    }
                else:
                    return {
                        "reply": f"⚠️ Inspection notice: {inspect_res.get('error', 'No trained model found')}. Once you launch a training run, I will provide full parameter telemetry and validation tests!"
                    }
            else:
                return {
                    "reply": "Please select your studio project first so I can inspect its trained LoRA weights!"
                }


        # Intent: Story / Animation script generation
        if "script" in msg_lower or "story" in msg_lower or "scene" in msg_lower:
            char = "BendyBot"
            if project_dir:
                meta = DatasetService.load_project_meta(project_dir)
                if meta:
                    char = meta.get("character_name", char)
            
            script = StoryAgent.generate_animation_script(
                character_name=char,
                premise=user_message,
                num_scenes=3
            )
            if project_dir:
                StoryAgent.save_script_to_project(project_dir, script)

            return {
                "reply": (
                    f"🎬 The Story Agent has inked a brand new vintage animation script for **{char}**!\n"
                    f"Title: *{script.get('title', 'Adventures')}*\n"
                    f"Scenes created: {len(script.get('scenes', []))} panels with motion diffusion prompts and camera trajectories.\n"
                    f"Saved to `Prompts/animation_script.json`."
                ),
                "tool_called": "generate_animation_script",
                "data": script
            }

        # Fallback interactive LLM chat via Ollama if available
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST, timeout=12)
            messages = [
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            res = client.chat(model="qwen2.5-coder:14b", messages=messages)
            content = res.get("message", {}).get("content", "")
            if content:
                return {"reply": content, "tool_called": "llm_chat"}
        except Exception:
            pass

        # Standard Supervisor Greeting / Assistance
        return {
            "reply": (
                f"Welcome to **Geekatplay LoRA Maker** by Vladimir Chopine (Geekatplay Studio)!\n\n"
                f"I am standing by to coordinate your specialized creative agents:\n"
                f"- **Video Agent**: Feed me video footage to scrub keyframes with aspect-ratio bucketing.\n"
                f"- **Visual Agent**: Inking character triggers, style tokens, and auto-captioning scans via local vision models.\n"
                f"- **Story Agent**: Formulating scene-by-scene prompts, shot lists, and motion camera cues.\n"
                f"- **Trainer Agent**: Adaptive training with hardware auto-detection for FLUX.1, SDXL, and Video diffusion.\n\n"
                f"Select a preset (Photorealism, Anime, Stylized 3D, Vintage Cartoon, Concept Art) or import your media to begin!"
            )
        }
