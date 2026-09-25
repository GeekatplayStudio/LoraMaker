from pathlib import Path

from app.core.config import AppSettings
from app.services.runtime_service import RuntimeService


def test_discover_uses_configured_roots_and_finds_scripts(tmp_path):
    comfy_root = tmp_path / "comfy"
    (comfy_root / "models" / "checkpoints").mkdir(parents=True)
    (comfy_root / "models" / "loras").mkdir()

    kohya_checkout = tmp_path / "kohya"
    scripts = kohya_checkout / "sd-scripts"
    scripts.mkdir(parents=True)
    (scripts / "sdxl_train_network.py").write_text("# test", encoding="utf-8")
    (scripts / "flux_train_network.py").write_text("# test", encoding="utf-8")

    config = AppSettings(
        COMFYUI_ROOT=comfy_root,
        KOHYA_ROOT=kohya_checkout,
        KOHYA_FALLBACK_ROOT=tmp_path / "missing-fallback",
    )
    discovered = RuntimeService.discover(config)

    assert discovered["comfyui"]["exists"] is True
    assert discovered["comfyui"]["checkpoints_exists"] is True
    assert discovered["comfyui"]["loras_exists"] is True
    assert discovered["kohya"][0]["sd_scripts_root"] == str(scripts.resolve())
    assert discovered["kohya"][0]["sdxl_train_network.py_exists"] is True
    assert discovered["kohya"][0]["flux_train_network.py_exists"] is True


def test_discover_accepts_direct_sd_scripts_root_and_missing_paths(tmp_path):
    scripts = tmp_path / "sd-scripts"
    scripts.mkdir()
    config = AppSettings(
        COMFYUI_ROOT=tmp_path / "does-not-exist",
        KOHYA_ROOT=scripts,
        KOHYA_FALLBACK_ROOT=scripts,
    )

    discovered = RuntimeService.discover(config)

    assert discovered["comfyui"]["exists"] is False
    assert len(discovered["kohya"]) == 1
    assert discovered["kohya"][0]["sd_scripts_root"] == str(scripts.resolve())
    assert discovered["kohya"][0]["sdxl_train_network.py_exists"] is False


def test_runtime_profile_is_serializable_and_reports_safe_accelerator_data(tmp_path):
    profile = RuntimeService.runtime_profile(
        AppSettings(COMFYUI_ROOT=tmp_path / "comfy", KOHYA_ROOT=tmp_path / "kohya")
    )

    assert profile["accelerator"]["backend"] in {"cpu", "cuda", "mps"}
    assert isinstance(profile["accelerator"]["torch_available"], bool)
    assert "installations" in profile
