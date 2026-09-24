import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import ollama
from app.core.config import settings

logger = logging.getLogger(__name__)

class StoryAgent:
    """
    Story Specialist Agent:
    Constructs detailed, standardized vintage cartoon screenplay scripts formatted as clean JSON
    with scene action descriptions, timing cues, and camera trajectory prompts for video diffusion models.
    """

    @classmethod
    def generate_animation_script(
        cls,
        character_name: str = "BendyBot",
        premise: str = "A mischievous robot in a 1930s vintage factory escapes a conveyor belt",
        num_scenes: int = 3,
        style_preset: str = "vintage 1930s rubber hose cel animation",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates structured storyboard animation script formatted for Wan2.1 / LTX-Video engines.
        """
        # Attempt LLM generation if available
        llm_model = model_name or cls._get_best_llm_model()
        script = None

        if llm_model:
            try:
                script = cls._call_llm_for_script(
                    model=llm_model,
                    character=character_name,
                    premise=premise,
                    num_scenes=num_scenes,
                    style=style_preset
                )
            except Exception as e:
                logger.warning(f"Ollama story generation failed ({e}), using curated screenplay template.")

        if not script:
            script = cls._build_curated_script(character_name, premise, num_scenes, style_preset)

        return script

    @classmethod
    def _get_best_llm_model(cls) -> Optional[str]:
        """Finds fastest/best available text LLM in Ollama."""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST, timeout=5)
            installed = [m.model for m in client.list().models]
            for pref in settings.PREFERRED_LLM_MODELS:
                for inst in installed:
                    if inst == pref or inst.startswith(pref.split(":")[0]):
                        return inst
            if installed:
                return installed[0]
        except Exception:
            pass
        return None

    @classmethod
    def _call_llm_for_script(
        cls,
        model: str,
        character: str,
        premise: str,
        num_scenes: int,
        style: str
    ) -> Optional[Dict[str, Any]]:
        client = ollama.Client(host=settings.OLLAMA_HOST, timeout=30)
        prompt = f"""You are the Story & Script Agent for Geekatplay LoRA Maker (Geekatplay Studio). Create a structured animation or video diffusion script for the specified style ({style}).
Output ONLY valid JSON matching this exact structure:
{{
  "title": "Short title",
  "character": "{character}",
  "style": "{style}",
  "total_scenes": {num_scenes},
  "scenes": [
    {{
      "scene_number": 1,
      "duration_sec": 3.0,
      "timing_cues": "00:00 - 00:03",
      "camera_motion": "slow cinematic dolly in, shallow depth of field",
      "action_description": "Detailed action description of {character}",
      "diffusion_prompt": "High-fidelity diffusion prompt including {character} and {style} visual descriptors",
      "negative_prompt": "blurry, distorted anatomy, jitter, low quality"
    }}
  ]
}}
Premise: {premise}
"""
        response = client.generate(model=model, prompt=prompt, stream=False)
        raw = response.get("response", "").strip()
        # Find JSON block
        if "{" in raw and "}" in raw:
            json_str = raw[raw.find("{"):raw.rfind("}")+1]
            return json.loads(json_str)
        return None

    @classmethod
    def _build_curated_script(
        cls,
        character: str,
        premise: str,
        num_scenes: int,
        style: str
    ) -> Dict[str, Any]:
        """Curated fallback ensuring robust zero-latency storyboard generation across genres."""
        is_retro = any(w in style.lower() for w in ["retro", "vintage", "rubberhose", "cartoon", "1930"])
        is_anime = any(w in style.lower() for w in ["anime", "manga", "cel"])
        is_photoreal = any(w in style.lower() for w in ["photo", "real", "portrait", "cinema"])

        if is_anime:
            cam1, act1 = "wide establishing shot, cherry blossoms drifting", f"{character} stands poised on high ground, looking toward the horizon with determined eyes."
            prm1 = f"{character}, {style}, heroic pose against sweeping sky, vibrant color palette, high detail anime keyframe"
            cam2, act2 = "rapid camera zoom with speed lines", f"{character} unleashes signature technique with focused expression."
            prm2 = f"{character}, {style}, dynamic battle motion, glowing particle effects, intense anime action sequence"
            cam3, act3 = "gentle pan down as wind settles", f"{character} catches breath with a confident, subtle smile."
            prm3 = f"{character}, {style}, emotional close up, sunset rim lighting, masterwork anime illustration"
        elif is_photoreal:
            cam1, act1 = "slow tracking shot, 50mm anamorphic lens", f"{character} enters the frame thoughtfully, adjusting clothing under cinematic lighting."
            prm1 = f"{character}, {style}, natural candid motion, shallow depth of field, 8k uhd photorealistic film still"
            cam2, act2 = "medium close-up, dramatic rim light", f"{character} turns toward the camera, speaking with expressive nuance."
            prm2 = f"{character}, {style}, detailed skin texture, catchlights in eyes, atmospheric haze, cinema lighting"
            cam3, act3 = "slow dolly back into ambient shadows", f"{character} pauses and looks into the distance as background elements move naturally."
            prm3 = f"{character}, {style}, moody golden hour glow, photographic composition, hyperrealistic award-winning cinematography"
        elif is_retro:
            cam1, act1 = "wide establishing shot, slow pan with vintage vignetting", f"{character} taps foot to upbeat tempo, blinking oversized pie-eyes."
            prm1 = f"{character}, {style}, standing in front of vintage backdrop, rubber hose limb sway, 24fps cel animation, high contrast monochrome ink lines"
            cam2, act2 = "medium close-up, dramatic squash and stretch camera tilt", f"{character} notices a giant mechanical lever and scurries toward it."
            prm2 = f"{character}, {style}, dynamic exaggerated running pose, dust clouds billowing, bouncy animation timing, vintage theatrical cel scan"
            cam3, act3 = "dolly back as steam bursts, iris-out ending transition", f"{character} pulls the lever, causing a comical celebration before tipping their hat."
            prm3 = f"{character}, {style}, tipping cartoon hat, celebratory victory pose, vintage confetti and musical notes, film grain texture"
        else:
            cam1, act1 = "cinematic crane down, soft ambient lighting", f"{character} is introduced showcasing distinctive styling and posture."
            prm1 = f"{character}, {style}, establishing scene composition, clean focal depth, high aesthetic coherence"
            cam2, act2 = "dynamic orbital push in", f"{character} engages in characteristic motion showcasing unique materials and personality."
            prm2 = f"{character}, {style}, dynamic lighting highlights, rich textural details, masterpiece rendering"
            cam3, act3 = "slow pull back revealing full environment", f"{character} strikes final signature pose as environmental elements react."
            prm3 = f"{character}, {style}, iconic hero shot, balanced color harmony, pristine visual fidelity"

        scene_templates = [
            {
                "scene_number": 1,
                "duration_sec": 3.0,
                "timing_cues": "00:00 - 00:03",
                "camera_motion": cam1,
                "action_description": act1,
                "diffusion_prompt": prm1,
                "negative_prompt": "blurry, low quality, distorted anatomy, jitter"
            },
            {
                "scene_number": 2,
                "duration_sec": 3.5,
                "timing_cues": "00:03 - 00:06.5",
                "camera_motion": cam2,
                "action_description": act2,
                "diffusion_prompt": prm2,
                "negative_prompt": "static pose, stiff motion, low contrast, artifacts"
            },
            {
                "scene_number": 3,
                "duration_sec": 4.0,
                "timing_cues": "00:06.5 - 00:10.5",
                "camera_motion": cam3,
                "action_description": act3,
                "diffusion_prompt": prm3,
                "negative_prompt": "jitter, distorted features, color noise, flickering"
            }
        ]

        return {
            "title": f"The Adventures of {character}",
            "character": character,
            "style": style,
            "premise": premise,
            "total_scenes": min(num_scenes, len(scene_templates)),
            "scenes": scene_templates[:num_scenes]
        }

    @staticmethod
    def save_script_to_project(project_dir: str, script_data: Dict[str, Any]) -> str:
        """Saves generated storyboard script to Prompts/animation_script.json."""
        p_dir = Path(project_dir)
        prompts_dir = p_dir / "Prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        target_file = prompts_dir / "animation_script.json"
        target_file.write_text(json.dumps(script_data, indent=2), encoding="utf-8")
        return str(target_file.resolve())
