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

### Chunks, and why they are on by default

PNG frames are big: one 30-second chunk of 720p at 60 fps costs a few GB, and a
feature film processed in one pass would need hundreds. So anything longer than
30 seconds is processed **one chunk at a time** — cut, upscale, interpolate,
encode that chunk, delete its frames, move on — and the encoded parts are
joined at the end. Peak disk stays at about one chunk no matter how long the
video is.

```bash
# the default: 30-second chunks
python -m videoenhance.cli input.mp4 -o output.mp4 --scale 2 --fps 60

# bigger chunks: fewer boundaries, more disk
python -m videoenhance.cli input.mp4 -o output.mp4 --chunk 60

# the old single-pass behaviour, which needs room for every frame at once
python -m videoenhance.cli input.mp4 -o output.mp4 --chunk 0
```

`--dry-run` prints the estimate before anything runs, which is the quickest way
to see what a setting costs.

The audio is muxed once, at the end, from the original file — never per part,
which is what would let the sound drift away from the picture over a long video.

### Stopping and continuing later

An interrupted run keeps its work and writes down what it was doing, so you do
not have to remember the command:

```bash
# what did I leave unfinished?
python -m videoenhance.cli --jobs

JOB            INPUT     PROGRESS    STOPPED       SETTINGS
film-7f3a91c2  film.mkv  18/94 parts 3 hours ago   2x, 60fps, chunk 30s

# carry on, with the options it was started with
python -m videoenhance.cli --resume film-7f3a91c2     # or just --resume film
```

A part file only appears once it is complete, so a resume picks up at the first
missing one. Re-running the original command works too — `--resume` just saves
you from retyping it.

**Resuming with different settings is refused**, and this matters: every stage
decides whether it already ran by counting frames, and the frame count does not
change when `--scale` or `--model` does. A job started at 2x and resumed at 4x
would skip the upscale and hand you a 2x video while reporting 4x. So the
settings are recorded and checked:

```
error: this job was started with different options:
  --scale  2 -> 4
18 of 94 parts already encoded use the old ones.
Use --restart to throw that away and start over, or run with the original options.
```

`--restart` discards the saved work and begins again. Settings that do not
change the pixels — `--gpu`, `--tile`, `--keep-temp` — are free to differ.

## Development

```bash
pytest -q
```

Project conventions and the full design live in [CLAUDE.md](CLAUDE.md).
