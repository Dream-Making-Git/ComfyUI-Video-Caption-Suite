"""ComfyUI node definitions for Video Caption Suite."""

import os
from pathlib import Path

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar

from . import captioner
from .media import (
    batch_to_pil_frames,
    effective_fps,
    find_media,
    load_media_file,
)
from .presets import DEFAULT_PRESET, MODEL_PRESETS, frame_size_for_preset

CATEGORY = "Video Caption Suite"
MODELS_DIR = Path(folder_paths.models_dir) / "LLM"

DEFAULT_PROMPT = """Describe this video in detail. Include:
- The main subject and their actions
- The setting and environment
- Any notable objects or elements
- The overall mood or atmosphere
- Any text visible in the video"""

MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return str(mm.get_torch_device())
    return choice


def _generation_inputs():
    """Inputs shared by the caption and folder nodes."""
    return {
        "prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True,
                              "tooltip": "Instruction sent to the vision-language model."}),
        "max_frames": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1,
                               "tooltip": "Frames sampled uniformly from the clip. More = better "
                                          "understanding, slower, more VRAM. Gemma 4 handles ~32 well."}),
        "frame_size": ("INT", {"default": 336, "min": 112, "max": 1344, "step": 8,
                               "tooltip": "Longest side (px) frames are resized to. "
                                          "Gemma 4 rounds this up to a multiple of 48."}),
        "max_tokens": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16,
                               "tooltip": "Maximum caption length in tokens."}),
        "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                                  "tooltip": "0.1 or lower = deterministic. Higher = more varied."}),
        "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,
                         "tooltip": "Only matters when temperature is above 0.1."}),
        "keep_model_loaded": ("BOOLEAN", {"default": True,
                                          "tooltip": "Turn off to unload the captioner and free VRAM "
                                                     "after this node finishes."}),
    }


def _caption_frames(info, frames, prompt, max_tokens, temperature, seed, fps):
    return captioner.generate_caption(
        info, frames, prompt,
        max_tokens=max_tokens, temperature=temperature, video_fps=fps, seed=seed,
    )


def _finish(keep_model_loaded: bool):
    if not keep_model_loaded:
        captioner.clear_cache()
        mm.soft_empty_cache()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class VCSLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(MODEL_PRESETS.keys()), {
                    "default": DEFAULT_PRESET,
                    "tooltip": "qwen3-vl-8b ~16GB VRAM | gemma-4-26b-a4b-int4 ~15GB | "
                               "gemma-4-26b-a4b ~52GB (shards across GPUs). Weights download "
                               "to models/LLM on first use.",
                }),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "custom_model_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional HuggingFace repo id (e.g. Qwen/Qwen3-VL-4B-Instruct). "
                               "Uses the selected preset's loading strategy.",
                }),
                "offload_comfy_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload other ComfyUI models from VRAM before loading the captioner.",
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Qwen only. Faster after a slow first run.",
                }),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only enable for custom model ids that require it. "
                               "Runs code from the model repository.",
                }),
            },
        }

    RETURN_TYPES = ("VCS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, preset, dtype, device, custom_model_id="", offload_comfy_models=True,
             use_torch_compile=False, trust_remote_code=False):
        # The handle is only a description; heavy objects stay in captioner's cache so
        # keep_model_loaded=False can really free memory and later runs reload cleanly.
        handle = {
            "preset_id": preset,
            "custom_model_id": custom_model_id.strip(),
            "dtype": dtype,
            "device": _resolve_device(device),
            "use_torch_compile": use_torch_compile,
            "trust_remote_code": trust_remote_code,
        }
        if not captioner.is_cached(handle):
            if offload_comfy_models:
                mm.unload_all_models()
                mm.soft_empty_cache()
            captioner.load_model(handle, MODELS_DIR)
        return (handle,)


# ---------------------------------------------------------------------------
# Caption an IMAGE batch / VIDEO
# ---------------------------------------------------------------------------

class VCSCaptionVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("VCS_MODEL",), **_generation_inputs()},
            "optional": {
                "images": ("IMAGE", {"tooltip": "Video frames (or a single image) as an IMAGE batch, "
                                                "e.g. from Load Video (VHS) or any image node."}),
                "video": ("VIDEO", {"tooltip": "ComfyUI VIDEO input (e.g. Load Video). "
                                               "Takes priority over 'images' if both are connected."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                                  "tooltip": "Frame rate of 'images'. Ignored when 'video' is connected."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = CATEGORY

    def caption(self, model, prompt, max_frames, frame_size, max_tokens, temperature, seed,
                keep_model_loaded, images=None, video=None, fps=24.0):
        if video is not None:
            try:
                components = video.get_components()
                images, fps = components.images, float(components.frame_rate)
            except AttributeError as e:
                raise ValueError("Unsupported VIDEO input: could not read frames from it.") from e
        if images is None:
            raise ValueError("Connect either 'images' or 'video' to Caption Video.")

        info = captioner.load_model(model, MODELS_DIR)
        size = frame_size_for_preset(info["preset"], frame_size)

        frames, total = batch_to_pil_frames(images, max_frames, size)
        sample_fps = effective_fps(fps, len(frames), total)

        mm.throw_exception_if_processing_interrupted()
        text = _caption_frames(info, frames, prompt, max_tokens, temperature, seed, sample_fps)
        _finish(keep_model_loaded)
        return (text,)


# ---------------------------------------------------------------------------
# Batch caption a folder and write .txt files (the original app's workflow)
# ---------------------------------------------------------------------------

class VCSCaptionFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VCS_MODEL",),
                "directory": ("STRING", {"default": "", "tooltip": "Folder containing videos/images to caption."}),
                "include_videos": ("BOOLEAN", {"default": True}),
                "include_images": ("BOOLEAN", {"default": False}),
                "traverse_subfolders": ("BOOLEAN", {"default": False}),
                "skip_existing": ("BOOLEAN", {"default": True,
                                              "tooltip": "Skip files that already have a caption file."}),
                "output_extension": ("STRING", {"default": ".txt"}),
                **_generation_inputs(),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # folder contents change outside ComfyUI, always re-run

    def run(self, model, directory, include_videos, include_images, traverse_subfolders,
            skip_existing, output_extension, prompt, max_frames, frame_size, max_tokens,
            temperature, seed, keep_model_loaded):
        root = Path(os.path.expanduser(directory.strip().strip('"')))
        if not root.is_dir():
            raise ValueError(f"Not a folder: {root}")
        if not (include_videos or include_images):
            raise ValueError("Enable include_videos and/or include_images.")

        ext = output_extension.strip() or ".txt"
        if not ext.startswith("."):
            ext = "." + ext

        files = find_media(root, traverse_subfolders, include_videos, include_images)
        if not files:
            return self._result("No matching media files found.")

        info = captioner.load_model(model, MODELS_DIR)
        size = frame_size_for_preset(info["preset"], frame_size)

        done, skipped, failed = 0, 0, []
        pbar = ProgressBar(len(files))

        for path in files:
            mm.throw_exception_if_processing_interrupted()
            out_path = path.with_suffix(ext)

            if skip_existing and out_path.exists():
                skipped += 1
                pbar.update(1)
                continue

            try:
                frames, fps = load_media_file(path, max_frames, size)
                text = _caption_frames(info, frames, prompt, max_tokens, temperature, seed, fps)
                out_path.write_text(text, encoding="utf-8")
                done += 1
                print(f"[VideoCaptionSuite] {path.name} -> {out_path.name}")
            except mm.InterruptProcessingException:
                raise
            except Exception as e:  # noqa: BLE001
                failed.append(f"{path.name}: {e}")
                print(f"[VideoCaptionSuite] FAILED {path.name}: {e}")
            pbar.update(1)

        _finish(keep_model_loaded)

        summary = f"Captioned {done}, skipped {skipped}, failed {len(failed)} of {len(files)} files."
        if failed:
            summary += "\n" + "\n".join(failed[:20])
        return self._result(summary)

    @staticmethod
    def _result(summary):
        return {"ui": {"text": [summary]}, "result": (summary,)}


NODE_CLASS_MAPPINGS = {
    "VCSLoadModel": VCSLoadModel,
    "VCSCaptionVideo": VCSCaptionVideo,
    "VCSCaptionFolder": VCSCaptionFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VCSLoadModel": "Load Caption Model (VCS)",
    "VCSCaptionVideo": "Caption Video / Images (VCS)",
    "VCSCaptionFolder": "Caption Folder to .txt (VCS)",
}
