import json
import shutil
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class DatasetService:
    """
    Manages project lifecycle, folder creation, keyframe/caption tracking,
    and dataset formatting for Kohya_ss and ComfyUI LoRA training.
    """

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def fit_image_aspect_ratio(
        image: Any,
        target_width: int = 512,
        target_height: int = 512,
        mode: str = "pad",
        bg_color: tuple = (128, 128, 128)
    ) -> Any:
        """
        Guarantees that an image of ANY aspect ratio (16:9, 9:16, 2:3, 3:2, 4:3, 1:1)
        is NEVER squeezed or stretched:
        - 'pad': Preserves exact ratio without cropping by adding neutral padding (letterbox/pillarbox).
        - 'crop': Scales uniformly to fill canvas and center-crops excess without distortion.
        - 'bucket': Snaps to the closest multiple-of-64 aspect-ratio bucket preserving native shape.
        """
        from PIL import Image
        import math

        if not isinstance(image, Image.Image):
            return image

        w, h = image.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (target_width, target_height), bg_color)

        if mode == "crop":
            scale = max(target_width / w, target_height / h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - target_width) // 2
            top = (nh - target_height) // 2
            return resized.crop((left, top, left + target_width, top + target_height))
        elif mode == "bucket":
            aspect = w / h
            total_pixels = target_width * target_height
            raw_bw = math.sqrt(total_pixels * aspect)
            raw_bh = math.sqrt(total_pixels / aspect)
            bw = max(256, round(raw_bw / 64) * 64)
            bh = max(256, round(raw_bh / 64) * 64)
            scale = min(bw / w, bh / h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (bw, bh), bg_color)
            canvas.paste(resized, ((bw - nw) // 2, (bh - nh) // 2))
            return canvas
        else: # 'pad'
            scale = min(target_width / w, target_height / h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_width, target_height), bg_color)
            canvas.paste(resized, ((target_width - nw) // 2, (target_height - nh) // 2))
            return canvas

    @classmethod
    def get_aspect_ratio_resolution(cls, aspect_ratio: str = "1:1", base_resolution: int = 512) -> tuple:
        """
        Returns multiple-of-64 (width, height) optimal for training or generation
        without any stretching or distortion.
        """
        table_512 = {
            "1:1": (512, 512),
            "16:9": (640, 384),
            "9:16": (384, 640),
            "2:3": (448, 672),
            "3:2": (672, 448),
            "4:3": (576, 448),
            "3:4": (448, 576),
        }
        table_1024 = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "2:3": (832, 1216),
            "3:2": (1216, 832),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
        }
        if base_resolution >= 1024:
            return table_1024.get(aspect_ratio, (1024, 1024))
        return table_512.get(aspect_ratio, (512, 512))

    @classmethod
    def detect_dataset_aspect_ratio(cls, project_dir: str) -> Dict[str, Any]:
        """
        Scans Keyframes_Out in the project to detect the dominant native aspect ratio
        of extracted video frames (e.g. 16:9, 9:16, 2:3, 3:2, 4:3, 1:1) and recommends training buckets.
        """
        from PIL import Image
        p_dir = Path(project_dir)
        keyframes_dir = p_dir / "Keyframes_Out"
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        
        widths = []
        heights = []
        if keyframes_dir.exists():
            for p in list(keyframes_dir.iterdir())[:24]:
                if p.is_file() and p.suffix.lower() in valid_exts:
                    try:
                        with Image.open(p) as img:
                            w, h = img.size
                            if w > 0 and h > 0:
                                widths.append(w)
                                heights.append(h)
                    except Exception:
                        pass
        
        if not widths:
            return {
                "detected_ratio": "1:1",
                "aspect_float": 1.0,
                "native_resolution": (512, 512),
                "recommended_512": (512, 512),
                "recommended_1024": (1024, 1024),
                "is_vertical": False,
                "is_landscape": False
            }
        
        median_w = sorted(widths)[len(widths) // 2]
        median_h = sorted(heights)[len(heights) // 2]
        aspect = median_w / median_h

        # Known aspect ratios map
        ratios = [
            ("16:9", 16 / 9, (640, 384), (1344, 768)),
            ("9:16", 9 / 16, (384, 640), (768, 1344)),
            ("2:3", 2 / 3, (448, 672), (832, 1216)),
            ("3:2", 3 / 2, (672, 448), (1216, 832)),
            ("4:3", 4 / 3, (576, 448), (1152, 896)),
            ("3:4", 3 / 4, (448, 576), (896, 1152)),
            ("1:1", 1.0, (512, 512), (1024, 1024))
        ]
        
        closest = min(ratios, key=lambda r: abs(r[1] - aspect))
        return {
            "detected_ratio": closest[0],
            "aspect_float": round(aspect, 3),
            "native_resolution": (median_w, median_h),
            "recommended_512": closest[2],
            "recommended_1024": closest[3],
            "is_vertical": aspect < 0.9,
            "is_landscape": aspect > 1.1
        }

    @staticmethod
    def initialize_project(
        project_dir: str,
        project_name: str,
        character_name: str,
        style_description: str = "photorealistic portrait, natural lighting",
        trigger_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initializes the standard project folder structure specified in rec.txt:
        - Input_Video
        - Keyframes_Out
        - Prompts
        - Assets
        - Training
        - Output_Video
        """
        p_dir = Path(project_dir)
        p_dir.mkdir(parents=True, exist_ok=True)

        created_folders = {}
        for folder_name in settings.STANDARD_FOLDERS:
            folder_path = p_dir / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            created_folders[folder_name] = str(folder_path.resolve())

        # Clean trigger token for LoRA training
        raw_token = trigger_token or character_name
        clean_token = "".join(c for c in raw_token if c.isalnum() or c in ("_", "-")).strip() or "char_token"

        meta = {
            "project_name": project_name,
            "character_name": character_name,
            "trigger_token": clean_token,
            "style_token": style_description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "folder_structure": created_folders,
            "status": "initialized"
        }

        meta_file = p_dir / "project.json"
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return meta

    @staticmethod
    def load_project_meta(project_dir: str) -> Optional[Dict[str, Any]]:
        """Loads project.json if exists."""
        meta_file = Path(project_dir) / "project.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Error loading project metadata: {e}")
        return None

    @staticmethod
    def get_project_dataset(project_dir: str) -> Dict[str, Any]:
        """
        Scans Keyframes_Out and aggregates all frame pairs (image + txt caption).
        """
        p_dir = Path(project_dir)
        keyframes_dir = p_dir / "Keyframes_Out"
        if not keyframes_dir.exists():
            return {"frames": [], "total_count": 0, "captioned_count": 0, "uncaptioned_count": 0}

        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = sorted([f for f in keyframes_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

        frames = []
        captioned_count = 0

        for img in image_files:
            txt_file = img.with_suffix(".txt")
            has_caption = txt_file.exists()
            caption_text = txt_file.read_text(encoding="utf-8").strip() if has_caption else ""
            if has_caption and caption_text:
                captioned_count += 1

            frames.append({
                "file_name": img.name,
                "file_path": str(img.resolve()),
                "relative_path": f"Keyframes_Out/{img.name}",
                "file_size_kb": round(img.stat().st_size / 1024, 1),
                "modified_at": datetime.fromtimestamp(img.stat().st_mtime).isoformat(),
                "has_caption": has_caption and bool(caption_text),
                "caption": caption_text,
                "caption_file": str(txt_file.resolve()) if has_caption else None
            })

        total = len(frames)
        return {
            "project_dir": str(p_dir.resolve()),
            "total_count": total,
            "captioned_count": captioned_count,
            "uncaptioned_count": total - captioned_count,
            "completion_percentage": round((captioned_count / total * 100), 1) if total > 0 else 0,
            "frames": frames
        }

    @staticmethod
    def update_frame_caption(image_path: str, new_caption: str) -> Dict[str, Any]:
        """Updates or creates the .txt caption file for an image."""
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Frame image not found: {image_path}")

        txt_p = img_p.with_suffix(".txt")
        txt_p.write_text(new_caption.strip(), encoding="utf-8")

        return {
            "success": True,
            "image_path": str(img_p),
            "caption_path": str(txt_p),
            "caption": new_caption.strip()
        }

    @staticmethod
    def delete_frame_pair(image_path: str) -> Dict[str, Any]:
        """Removes the frame image and its matching .txt caption file."""
        img_p = Path(image_path)
        txt_p = img_p.with_suffix(".txt")

        deleted = []
        if img_p.exists():
            img_p.unlink()
            deleted.append(str(img_p))
        if txt_p.exists():
            txt_p.unlink()
            deleted.append(str(txt_p))

        return {
            "success": True,
            "deleted_files": deleted
        }

    @staticmethod
    def export_for_kohya(
        project_dir: str,
        repeats: int = 10,
        class_token: str = "character",
        target_resolution: int = 1024,
        framing_mode: str = "pad"
    ) -> Dict[str, Any]:
        """
        Formats dataset for Kohya_ss sd-scripts:
        Structure:
          Training/
            img/
              {repeats}_{trigger_token} {class_token}/
                image01.png
                image01.txt
        Pre-fits all images to target_resolution using aspect-ratio preservation (pad or crop)
        to prevent Kohya 'image size is small' errors.
        Generates and saves Training/dataset_manifest.json with full transparency metrics.
        """
        from PIL import Image
        p_dir = Path(project_dir)
        meta = DatasetService.load_project_meta(project_dir) or {}
        trigger = meta.get("trigger_token", "BendyBot")

        dataset_info = DatasetService.get_project_dataset(project_dir)
        frames = [f for f in dataset_info["frames"] if f["has_caption"]]

        if not frames:
            raise ValueError("No captioned keyframes found in Keyframes_Out to export.")

        training_dir = p_dir / "Training"
        img_root = training_dir / "img"
        concept_folder_name = f"{repeats}_{trigger} {class_token}".strip()
        concept_dir = img_root / concept_folder_name

        # Clean and recreate concept dir
        if concept_dir.exists():
            shutil.rmtree(concept_dir)
        concept_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        total_upscaled = 0
        total_padded = 0
        total_cropped = 0
        manifest_items = []

        for f in frames:
            src_img = Path(f["file_path"])
            src_txt = Path(f["caption_file"])

            dst_img = concept_dir / src_img.name
            dst_txt = concept_dir / src_txt.name

            orig_w, orig_h = 0, 0
            color_mode = "unknown"
            is_upscaled = False
            aspect_str = "1:1"

            # Attempt image loading and pre-fitting
            try:
                with Image.open(src_img) as raw_im:
                    orig_w, orig_h = raw_im.size
                    color_mode = raw_im.mode
                    if orig_w > 0 and orig_h > 0:
                        ratio = orig_w / orig_h
                        if ratio > 1.6:
                            aspect_str = "16:9"
                        elif ratio > 1.25:
                            aspect_str = "4:3"
                        elif ratio < 0.65:
                            aspect_str = "9:16"
                        elif ratio < 0.8:
                            aspect_str = "3:4"
                        else:
                            aspect_str = "1:1"

                    if orig_w < target_resolution or orig_h < target_resolution:
                        is_upscaled = True
                        total_upscaled += 1

                    if framing_mode == "crop":
                        total_cropped += 1
                    else:
                        total_padded += 1

                    fitted = DatasetService.fit_image_aspect_ratio(
                        raw_im.convert("RGB"),
                        target_width=target_resolution,
                        target_height=target_resolution,
                        mode=framing_mode,
                        bg_color=(255, 255, 255)
                    )
                    fitted.save(dst_img, "PNG")
            except Exception as exc:
                # A copied corrupt image can look like a successful dataset
                # export and waste a full training run.  Export is an integrity
                # boundary, so fail before it can reach a trainer.
                raise ValueError(f"Unreadable reference image '{src_img.name}': {exc}") from exc

            # Copy or write companion caption file
            shutil.copy2(src_txt, dst_txt)
            exported_count += 1

            caption_content = ""
            try:
                caption_content = src_txt.read_text(encoding="utf-8").strip()
            except Exception:
                pass

            manifest_items.append({
                "filename": src_img.name,
                "source_sha256": DatasetService._sha256(src_img),
                "export_sha256": DatasetService._sha256(dst_img),
                "caption_sha256": DatasetService._sha256(dst_txt),
                "original_width": orig_w,
                "original_height": orig_h,
                "original_mode": color_mode,
                "original_aspect": aspect_str,
                "fitted_width": target_resolution,
                "fitted_height": target_resolution,
                "framing_mode": framing_mode,
                "upscaled": is_upscaled,
                "caption": caption_content
            })

        # Save transparent dataset manifest
        manifest_data = {
            "project_dir": str(p_dir.resolve()),
            "exported_at": datetime.now().isoformat(),
            "target_resolution": target_resolution,
            "framing_mode": framing_mode,
            "total_images": exported_count,
            "total_upscaled": total_upscaled,
            "total_padded": total_padded,
            "total_cropped": total_cropped,
            "repeats_per_image": repeats,
            "total_steps_estimate": exported_count * repeats,
            "images": manifest_items
        }

        manifest_file = training_dir / "dataset_manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return {
            "format": "kohya_ss",
            "destination_dir": str(concept_dir.resolve()),
            "concept_folder": concept_folder_name,
            "exported_frames": exported_count,
            "total_upscaled": total_upscaled,
            "total_padded": total_padded,
            "total_cropped": total_cropped,
            "repeats": repeats,
            "target_resolution": target_resolution,
            "manifest_file": str(manifest_file.resolve()),
            "total_steps_estimate": exported_count * repeats
        }

    @staticmethod
    def get_dataset_manifest(project_dir: str) -> Dict[str, Any]:
        """
        Retrieves the dataset manifest detailing how images were fitted, upscaled,
        and captioned for complete transparency.
        """
        p_dir = Path(project_dir)
        manifest_file = p_dir / "Training" / "dataset_manifest.json"
        if manifest_file.exists():
            try:
                return json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read dataset manifest: {e}")

        # If not yet exported, build live summary from Keyframes_Out
        from PIL import Image
        keyframes_dir = p_dir / "Keyframes_Out"
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        items = []
        upscaled_cnt = 0
        if keyframes_dir.exists():
            for p in sorted(keyframes_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    txt = p.with_suffix(".txt")
                    cap = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
                    orig_w, orig_h = 0, 0
                    try:
                        with Image.open(p) as im:
                            orig_w, orig_h = im.size
                    except Exception:
                        pass
                    is_up = orig_w < 1024 or orig_h < 1024
                    if is_up:
                        upscaled_cnt += 1
                    items.append({
                        "filename": p.name,
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "original_aspect": f"{orig_w}:{orig_h}" if orig_h else "unknown",
                        "fitted_width": 1024,
                        "fitted_height": 1024,
                        "framing_mode": "pad",
                        "upscaled": is_up,
                        "caption": cap
                    })

        return {
            "project_dir": str(p_dir.resolve()),
            "exported_at": None,
            "target_resolution": 1024,
            "framing_mode": "pad",
            "total_images": len(items),
            "total_upscaled": upscaled_cnt,
            "total_padded": len(items),
            "total_cropped": 0,
            "repeats_per_image": 10,
            "total_steps_estimate": len(items) * 10,
            "images": items
        }

    @staticmethod
    def export_for_comfyui(project_dir: str) -> Dict[str, Any]:
        """
        Formats dataset for ComfyUI LoRA training nodes:
        Creates JSONL metadata and standardized asset folder in Assets/ComfyUI_Dataset.
        """
        p_dir = Path(project_dir)
        dataset_info = DatasetService.get_project_dataset(project_dir)
        frames = [f for f in dataset_info["frames"] if f["has_caption"]]

        if not frames:
            raise ValueError("No captioned keyframes found to export for ComfyUI.")

        export_dir = p_dir / "Assets" / "ComfyUI_Dataset"
        export_dir.mkdir(parents=True, exist_ok=True)

        metadata_lines = []
        for f in frames:
            src_img = Path(f["file_path"])
            dst_img = export_dir / src_img.name
            try:
                from PIL import Image
                with Image.open(src_img) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable reference image '{src_img.name}': {exc}") from exc
            if not str(f.get("caption", "")).strip():
                raise ValueError(f"Reference image '{src_img.name}' has an empty caption.")
            shutil.copy2(src_img, dst_img)

            # ComfyUI JSONL format
            metadata_lines.append(json.dumps({
                "file_name": src_img.name,
                "text": f["caption"]
            }))

        jsonl_file = export_dir / "metadata.jsonl"
        jsonl_file.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

        return {
            "format": "comfyui",
            "destination_dir": str(export_dir.resolve()),
            "metadata_file": str(jsonl_file.resolve()),
            "exported_frames": len(frames)
        }
