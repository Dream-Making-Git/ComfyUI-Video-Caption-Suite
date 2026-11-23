"""
Model loading and caption generation.

Ported from Video-Caption-Suite backend/model_loader.py with the FastAPI /
config coupling removed. Differences from the original:
  * no SageAttention monkey-patch (it patched torch globally, which would affect
    every other node in ComfyUI, and was disabled for Qwen3-VL anyway)
  * trust_remote_code is opt-in instead of always on
  * optional seed for reproducible sampling
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .presets import resolve_preset

_MODEL_CACHE: Dict[str, Any] = {}

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def get_dtype(name: str) -> torch.dtype:
    return _DTYPES.get(name.lower(), torch.bfloat16)


def cache_key(preset: Dict[str, Any], device: str, dtype: str) -> str:
    return f"{preset['model_id']}_{device}_{dtype}_{preset.get('quantization') or 'none'}"


def is_cached(handle: Dict[str, Any]) -> bool:
    preset = resolve_preset(handle["preset_id"], handle.get("custom_model_id"))
    return cache_key(preset, handle["device"], handle["dtype"]) in _MODEL_CACHE


def download_model(model_id: str, local_dir: Path) -> Path:
    """Download weights into `local_dir/<repo-name>` if not already there."""
    from huggingface_hub import snapshot_download

    model_path = Path(local_dir) / model_id.split("/")[-1]
    if (model_path / "config.json").exists():
        print(f"[VideoCaptionSuite] Model already downloaded: {model_path}")
        return model_path

    print(f"[VideoCaptionSuite] Downloading {model_id} to {model_path} (this can take a while)")
    start = time.time()
    model_path.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(repo_id=model_id, local_dir=str(model_path))
    print(f"[VideoCaptionSuite] Download complete in {time.time() - start:.1f}s")
    return Path(downloaded)


def _from_pretrained(cls, path: str, dtype: torch.dtype, **kwargs):
    """`dtype=` on new transformers, `torch_dtype=` on older ones."""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


# ---------------------------------------------------------------------------
# Loader strategies
# ---------------------------------------------------------------------------

def _load_image_text_to_text(preset, model_path, torch_dtype, torch_device, use_torch_compile, trust_remote_code):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = _from_pretrained(
        AutoModelForImageTextToText, str(model_path), torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
    ).to(torch_device)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)

    compiled = False
    if use_torch_compile and preset["supports_torch_compile"]:
        try:
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            compiled = True
        except Exception as e:  # noqa: BLE001
            print(f"[VideoCaptionSuite] torch.compile failed, continuing without it: {e}")

    return {"model": model, "processor": processor, "torch_compiled": compiled, "device_map": None}


def _load_gemma4(preset, model_path, torch_dtype, torch_device):
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    quantization_config = None
    if preset.get("quantization") == "int4_torchao":
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        from transformers import TorchAoConfig

        quantization_config = TorchAoConfig(
            Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
        )

    shard = (
        preset.get("supports_multi_gpu_shard", False)
        and torch.cuda.device_count() > 1
        and quantization_config is None
    )

    kwargs: Dict[str, Any] = {"attn_implementation": "sdpa"}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    kwargs["device_map"] = "auto" if shard else {"": torch_device}

    model = _from_pretrained(Gemma4ForConditionalGeneration, str(model_path), torch_dtype, **kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path))

    return {
        "model": model, "processor": processor, "torch_compiled": False,
        "device_map": "auto" if shard else "single",
    }


def load_model(handle: Dict[str, Any], models_dir: Path) -> Dict[str, Any]:
    """Load (or fetch from cache) the model described by a VCS_MODEL handle."""
    preset = resolve_preset(handle["preset_id"], handle.get("custom_model_id"))
    device, dtype = handle["device"], handle["dtype"]

    key = cache_key(preset, device, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    print(f"[VideoCaptionSuite] Loading {preset['model_id']} ({preset['loader']}, {dtype}, {device})")
    start = time.time()

    model_path = download_model(preset["model_id"], models_dir)
    torch_dtype = get_dtype(dtype)
    torch_device = torch.device(device)

    if preset["loader"] == "image_text_to_text":
        info = _load_image_text_to_text(
            preset, model_path, torch_dtype, torch_device,
            handle.get("use_torch_compile", False), handle.get("trust_remote_code", False),
        )
    elif preset["loader"] == "gemma4":
        info = _load_gemma4(preset, model_path, torch_dtype, torch_device)
    else:
        raise ValueError(f"Unknown loader strategy: {preset['loader']}")

    info.update({"model_id": preset["model_id"], "device": torch_device, "preset": preset})
    _MODEL_CACHE[key] = info
    print(f"[VideoCaptionSuite] Model ready in {time.time() - start:.1f}s")
    return info


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_image_text_to_text(info, images, prompt, max_tokens, temperature):
    model, processor, device = info["model"], info["processor"], info["device"]

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = {**inputs, "max_new_tokens": max_tokens, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3}
    if temperature > 0.1:
        gen_kwargs.update({"temperature": temperature, "do_sample": True, "top_p": 0.95, "top_k": 40})
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _generate_gemma4(info, images, prompt, max_tokens, temperature, video_fps):
    model, processor, device, preset = info["model"], info["processor"], info["device"], info["preset"]

    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": images}, {"type": "text", "text": prompt}],
    }]

    fps = float(video_fps) if (video_fps and video_fps > 0) else 24.0
    videos_kwargs = {
        "do_sample_frames": False,
        "video_metadata": [{
            "fps": fps,
            "duration": len(images) / fps,
            "total_num_frames": len(images),
            "frames_indices": list(range(len(images))),
        }],
    }

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        processor_kwargs={"videos_kwargs": videos_kwargs},
    )
    inputs.pop("token_type_ids", None)
    if info.get("device_map") != "auto":
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    defaults = preset.get("gen_defaults") or {}
    gen_kwargs = {**inputs, "max_new_tokens": max_tokens}
    if temperature > 0.1:
        gen_kwargs.update({
            "temperature": temperature, "do_sample": True,
            "top_p": defaults.get("top_p", 0.95), "top_k": defaults.get("top_k", 64),
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)

    text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    try:  # strips thinking/channel tokens when the processor supports it
        parsed = processor.parse_response(text)
        if isinstance(parsed, str) and parsed:
            text = parsed
        elif isinstance(parsed, dict) and parsed.get("response"):
            text = parsed["response"]
    except Exception:  # noqa: BLE001
        pass
    return text.strip()


def generate_caption(
    info: Dict[str, Any],
    images: List,
    prompt: str,
    max_tokens: int,
    temperature: float,
    video_fps: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    loader = info["preset"]["loader"]
    if loader == "image_text_to_text":
        return _generate_image_text_to_text(info, images, prompt, max_tokens, temperature).strip()
    if loader == "gemma4":
        return _generate_gemma4(info, images, prompt, max_tokens, temperature, video_fps)
    raise ValueError(f"Unknown loader strategy: {loader}")


def clear_cache() -> None:
    """Drop all cached models and free GPU memory."""
    for info in list(_MODEL_CACHE.values()):
        info.pop("model", None)
        info.pop("processor", None)
    _MODEL_CACHE.clear()

    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    gc.collect()
    print("[VideoCaptionSuite] Model unloaded and GPU memory freed")
