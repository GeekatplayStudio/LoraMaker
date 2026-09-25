import os
from pathlib import Path
import pytest
from app.services.dataset_service import DatasetService
from app.core.config import settings

def test_initialize_project_structure(temp_project_dir):
    meta = DatasetService.initialize_project(
        project_dir=temp_project_dir,
        project_name="TestVintageStudio",
        character_name="BendyBot",
        style_description="vintage 1930s rubber hose"
    )

    assert meta["project_name"] == "TestVintageStudio"
    assert meta["trigger_token"] == "BendyBot"

    # Verify all 6 standard folders exist as required by rec.txt
    for folder_name in settings.STANDARD_FOLDERS:
        f_path = Path(temp_project_dir) / folder_name
        assert f_path.exists() and f_path.is_dir(), f"Folder {folder_name} missing"

    # Verify project.json
    meta_loaded = DatasetService.load_project_meta(temp_project_dir)
    assert meta_loaded is not None
    assert meta_loaded["character_name"] == "BendyBot"

def test_dataset_tracking_and_exports(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "ExportTest", "OldManPete")

    keyframes_dir = Path(temp_project_dir) / "Keyframes_Out"

    # Create 3 synthetic frames
    from PIL import Image
    for i in range(1, 4):
        img_file = keyframes_dir / f"OldManPete_{i:04d}.png"
        Image.new("RGB", (64, 64), (i * 20, 30, 40)).save(img_file)
        txt_file = keyframes_dir / f"OldManPete_{i:04d}.txt"
        txt_file.write_text(f"OldManPete, cel scan pose #{i}", encoding="utf-8")

    # Get dataset summary
    summary = DatasetService.get_project_dataset(temp_project_dir)
    assert summary["total_count"] == 3
    assert summary["captioned_count"] == 3
    assert summary["uncaptioned_count"] == 0
    assert summary["completion_percentage"] == 100.0

    # Test Kohya export
    kohya_res = DatasetService.export_for_kohya(temp_project_dir, repeats=10, class_token="character")
    assert kohya_res["exported_frames"] == 3
    assert "10_OldManPete character" in kohya_res["concept_folder"]
    dest_path = Path(kohya_res["destination_dir"])
    assert dest_path.exists()
    assert (dest_path / "OldManPete_0001.png").exists()
    assert (dest_path / "OldManPete_0001.txt").exists()

    # Test ComfyUI export
    comfy_res = DatasetService.export_for_comfyui(temp_project_dir)
    assert comfy_res["exported_frames"] == 3
    assert Path(comfy_res["metadata_file"]).exists()

def test_caption_update_and_delete(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "EditTest", "Bendy")
    keyframes_dir = Path(temp_project_dir) / "Keyframes_Out"
    img = keyframes_dir / "Bendy_0001.png"
    img.write_bytes(b"image")

    # Update caption
    res = DatasetService.update_frame_caption(str(img), "Bendy, updated caption")
    assert res["success"] is True
    assert (keyframes_dir / "Bendy_0001.txt").read_text(encoding="utf-8") == "Bendy, updated caption"

    # Delete pair
    del_res = DatasetService.delete_frame_pair(str(img))
    assert del_res["success"] is True
    assert not img.exists()
    assert not (keyframes_dir / "Bendy_0001.txt").exists()

def test_export_rejects_corrupt_reference_image(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "Integrity", "Bendy")
    keyframes = Path(temp_project_dir) / "Keyframes_Out"
    (keyframes / "Bendy_0001.png").write_bytes(b"not a png")
    (keyframes / "Bendy_0001.txt").write_text("Bendy, cel scan", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreadable reference image"):
        DatasetService.export_for_kohya(temp_project_dir)
