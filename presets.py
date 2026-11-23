"""
Model preset registry (ported from Video-Caption-Suite backend/model_presets.py).

Each preset describes which HuggingFace model to load and which loader
strategy to use:

    image_text_to_text -> Qwen-VL family and similar (one image block per frame)
    gemma4             -> Google Gemma 4 (single video block carrying the frames)
"""

from typing import Any, Dict, Optional

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3-vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "label": "Qwen3-VL 8B (default)",
        "loader": "image_text_to_text",
        "supports_torch_compile": True,
        "supports_multi_gpu_shard": False,
        "quantization": None,
        "default_max_frames": 16,
        "default_frame_size": 336,
        "approx_vram_gb": 16,
        "frame_size_divisor": None,
    },
    "gemma-4-26b-a4b": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B-A4B (bf16, ~52GB, shards across GPUs)",
        "loader": "gemma4",
        "supports_torch_compile": False,
        "supports_multi_gpu_shard": True,
        "quantization": None,
        "default_max_frames": 32,
        "default_frame_size": 336,
        "approx_vram_gb": 52,
        "frame_size_divisor": 48,
        "gen_defaults": {"top_p": 0.95, "top_k": 64},
    },
    "gemma-4-26b-a4b-int4": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B-A4B (int4 via TorchAo, ~15GB)",
        "loader": "gemma4",
        "supports_torch_compile": False,
        "supports_multi_gpu_shard": False,
        "quantization": "int4_torchao",
        "default_max_frames": 32,
        "default_frame_size": 336,
        "approx_vram_gb": 15,
        "frame_size_divisor": 48,
        "gen_defaults": {"top_p": 0.95, "top_k": 64},
    },
}

DEFAULT_PRESET = "qwen3-vl-8b"


def resolve_preset(preset_id: Optional[str], custom_model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the preset dict for `preset_id`.

    If `custom_model_id` is given, the chosen preset's loader strategy is kept
    but the HuggingFace repo id is replaced with the custom one.
    """
    preset = MODEL_PRESETS.get(preset_id or DEFAULT_PRESET, MODEL_PRESETS[DEFAULT_PRESET])
    custom_model_id = (custom_model_id or "").strip()
    if custom_model_id:
        preset = dict(preset)
        preset["model_id"] = custom_model_id
        preset["label"] = f"Custom: {custom_model_id}"
    return preset


def frame_size_for_preset(preset: Dict[str, Any], requested: int) -> int:
    """Round a requested frame size up to the preset's divisor (if any)."""
    divisor = preset.get("frame_size_divisor")
    if not divisor:
        return requested
    return ((requested + divisor - 1) // divisor) * divisor
