"""
Frame sampling / resizing / media discovery helpers.

Deliberately free of torch and ComfyUI imports so they can be tested alone.
cv2 is only imported when a video *file* has to be read (folder batch node).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".flv", ".wmv", ".m4v", ".mpeg", ".mpg",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def resize_image(image: Image.Image, max_size: int = 448, min_size: int = 224) -> Image.Image:
    """Resize preserving aspect ratio (same rules as the original suite)."""
    width, height = image.size
    max_dim = max(width, height)
    min_dim = min(width, height)

    if max_dim > max_size:
        scale = max_size / max_dim
    elif min_dim < min_size:
        scale = min_size / min_dim
    else:
        return image

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def sample_indices(total: int, max_frames: int) -> List[int]:
    """Uniformly sample up to `max_frames` indices from `total` frames."""
    if total <= max_frames:
        return list(range(total))
    return np.linspace(0, total - 1, max_frames, dtype=int).tolist()


def effective_fps(native_fps: Optional[float], sampled: int, total: int) -> Optional[float]:
    """
    fps to report for the *sampled* frames so per-frame timestamps still span
    the real clip duration (matters for Gemma 4, which prints timestamps).
    """
    if not native_fps or native_fps <= 0 or total <= 0:
        return native_fps
    return native_fps * sampled / total


def batch_to_pil_frames(images, max_frames: int, frame_size: int) -> Tuple[List[Image.Image], int]:
    """
    Convert a ComfyUI IMAGE batch ([N, H, W, C] float 0..1, torch tensor or
    array) into a list of resized PIL frames. Returns (frames, total_frames).
    """
    total = int(images.shape[0])
    if total == 0:
        raise ValueError("Received an empty image/video batch.")

    frames = []
    for i in sample_indices(total, max_frames):
        arr = images[i]
        if hasattr(arr, "detach"):  # torch tensor
            arr = arr.detach().cpu().float().numpy()
        arr = np.clip(np.asarray(arr), 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] >= 4:
            arr = arr[..., :3]
        frames.append(resize_image(Image.fromarray(arr), max_size=frame_size))
    return frames, total


def load_media_file(path: Path, max_frames: int, frame_size: int) -> Tuple[List[Image.Image], Optional[float]]:
    """
    Load an image or video file into PIL frames.
    Returns (frames, fps) where fps is None for still images.
    """
    ext = path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return [resize_image(img, max_size=frame_size)], None

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "Reading video files needs OpenCV. Install it with: pip install opencv-python-headless"
        ) from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            raise ValueError(f"Video has no frames: {path}")

        frames = []
        for idx in sample_indices(total, max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(resize_image(Image.fromarray(frame), max_size=frame_size))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Could not extract any frames from: {path}")

    return frames, effective_fps(native_fps, len(frames), total)


def find_media(
    directory: Path,
    traverse_subfolders: bool = False,
    include_videos: bool = True,
    include_images: bool = True,
) -> List[Path]:
    """Find video/image files in `directory`, sorted by name."""
    wanted = set()
    if include_videos:
        wanted |= VIDEO_EXTENSIONS
    if include_images:
        wanted |= IMAGE_EXTENSIONS

    found: List[Path] = []
    if traverse_subfolders:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if os.path.splitext(name)[1].lower() in wanted:
                    found.append(Path(root) / name)
    else:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in wanted:
                        found.append(Path(entry.path))
        except PermissionError:
            pass

    found.sort(key=lambda p: str(p).lower())
    return found
