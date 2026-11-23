# ComfyUI Video Caption Suite

ComfyUI custom nodes for captioning videos and images with a vision-language model
(Qwen3-VL-8B by default, Gemma 4 optional). Adapted from
[filliptm/Video-Caption-Suite](https://github.com/filliptm/Video-Caption-Suite), which
is a standalone FastAPI + Vue app. The frame sampling, model loading and generation
code is ported; the web server, UI and multi-GPU worker pool are not.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <your repo or copy this folder here> ComfyUI-Video-Caption-Suite

# use the same Python that runs ComfyUI
python -m pip install -r ComfyUI-Video-Caption-Suite/requirements.txt
```

Restart ComfyUI. Nodes appear under **Video Caption Suite**.

> `requirements.txt` asks for `transformers>=5.5.0` (needed for Gemma 4). If that
> conflicts with other custom nodes in your environment, you can try an older
> `transformers` with the Qwen preset only, but that combination is untested.

Model weights download automatically on first use into `ComfyUI/models/LLM/`.

## Nodes

**Load Caption Model (VCS)** - picks the model and loads it.
Output `model` goes into the caption nodes.

| Preset | VRAM |
|---|---|
| `qwen3-vl-8b` (default) | ~16 GB |
| `gemma-4-26b-a4b-int4` | ~15 GB |
| `gemma-4-26b-a4b` (bf16) | ~52 GB, shards across GPUs |

`custom_model_id` lets you point at another HuggingFace repo using the chosen preset's
loading strategy. `offload_comfy_models` unloads other ComfyUI models first so the
captioner has room.

**Caption Video / Images (VCS)** - returns a `STRING` caption.
Connect an `IMAGE` batch (video frames from Load Video (VHS), a single image, etc.)
or a native `VIDEO`. Wire the `caption` output into any text input, a Preview Any
node, or a save-text node.

**Caption Folder to .txt (VCS)** - the original app's batch workflow: point it at a
folder and it writes `<name>.txt` next to each video/image. Supports subfolders,
skip-existing, and cancellation from the ComfyUI queue.

Typical workflow:

```
Load Caption Model (VCS) ──► Caption Video / Images (VCS) ──► Preview Any
                                      ▲
                 Load Video ──────────┘  (images or video input)
```

## Differences from the original app

- No web UI, API server, analytics or resource monitor - ComfyUI is the interface.
- No parallel multi-GPU worker pool. Gemma 4 bf16 can still shard one model across
  GPUs; otherwise one model runs on one device.
- SageAttention support removed (it globally patched PyTorch attention, which would
  affect other nodes, and was disabled for Qwen3-VL anyway).
- `trust_remote_code` is off unless you enable it.
- Added a `seed` input and a `keep_model_loaded` toggle (turn off to free VRAM after
  captioning).
- When frames are subsampled, Gemma 4 receives an adjusted fps so its per-frame
  timestamps still span the real clip length.

## License

MIT, same as the original project.
