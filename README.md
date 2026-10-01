# enhanced-video

A command-line tool that upscales and smooths video on **AMD GPUs**, with no
CUDA and no PyTorch.

The pipeline is: **super-resolution** with Real-ESRGAN, **frame interpolation**
with RIFE, then reassembly that **keeps the original audio**. Both models run
through their NCNN + Vulkan builds, so the same code works on AMD, Intel and
NVIDIA. The Python package is only an orchestrator around FFmpeg and those two
executables.

> **Status:** early development. See [roadmap.md](roadmap.md) for what is built
> and what is next.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe on `PATH` (on Windows, a "full" build that includes AMF)
- A Vulkan-capable GPU with up-to-date drivers (Adrenalin on Windows, a recent
  Mesa/RADV on Linux)

## Setup

1. Install FFmpeg and verify it: `ffmpeg -version`.
2. Download the NCNN releases for your OS and unpack them, with their model
   folders, into `bin/`:
   - [realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) → `bin/realesrgan/`
   - [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) → `bin/rife/`
3. Install the package:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```

4. Check that everything is wired up:

   ```bash
   python -m videoenhance.cli --doctor
   ```

`bin/` is gitignored; the binaries are never committed.

## Usage

```bash
# inspect a file without processing it
python -m videoenhance.cli input.mp4 --probe-only

# 2x upscale to 60 fps
python -m videoenhance.cli input.mp4 -o output.mp4 --scale 2 --fps 60
```

## Development

```bash
pytest -q
```

Project conventions and the full design live in [CLAUDE.md](CLAUDE.md).
