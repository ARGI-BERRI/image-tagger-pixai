# Image tagger

Simple tagging system for images using `pixai-labs/pixai-tagger-v1.0` model.

> [!NOTE]
> Only the pair of Windows and ROCm is supported.
> Inference will fail in other patterns like Linux and CUDA.

## Prerequisites

* Windows 11
* Radeon GPU (`gfx1201` series, e.g., RX 9070 XT)
* Radeon Driver that supports ROCm 10.1

## Setup and run inference

* `pixai_tagger/main.py` will use Sixel to preview images in your terminal

```bash
# Install dependencies
uv sync

# Load images in images/
uv run pixai_tagger/main.py

# Start an API server
uv run fastapi run ./pixai_tagger/server.py
```

## License

Apache-2.0
