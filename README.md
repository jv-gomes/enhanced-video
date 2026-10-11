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

### Reusing repeated frames

Real-ESRGAN is deterministic, so a frame whose pixels are identical to one it has
already upscaled does not need upscaling again. Each frame is hashed, the
distinct ones go to the GPU, and the repeats are hard-linked to the result. This
is on by default and lossless — `--no-dedup` turns it off.

Measured on a 90-second 640x360 clip upscaled 2x to 60 fps, with each drawing
held for three frames:

| | without | with |
|---|---|---|
| total run | 3 min 45 s | **2 min 32 s** |
| upscale work | 2700 frames | **900 frames** |
| `frames_up/` on disk | 36 MB | **12 MB** |

All 5398 output frames came out bit-identical. On a clip with no repeated frames
the cost is the hashing pass alone: 4 min 12 s became 4 min 15 s, about 1%.

**How much it saves depends on the source**, so the summary reports the real
figure per run rather than promising one:

```
upscale: 1000 frames, 2x with realesr-animevideov3; 334 of 1000 frames generated (67% saved)
```

A held drawing only counts as a repeat if its pixels come back *identical*, and
that survives a lossy encode for flat, uniform artwork but often not for line
art or fine detail — B-frame prediction and per-frame rate decisions change the
low bits. Sources where the repeats are certain to be there are lossless or
intermediate files, and VFR recordings, whose CFR conversion duplicates frames
losslessly. Run `--dry-run` on a short clip to see what your own files give.

Frames generated for one chunk are kept for the next, under a ceiling set by
`--cache-size` (2 GiB by default, `0` for no ceiling). The ceiling matters: with
no limit the cache would grow with the length of the video and undo the point of
processing in chunks.

#### Would reusing blocks of a frame save more?

Not built yet, but measurable. When only part of the picture moves — a mouth
over a static background — no frame repeats as a whole, yet most of it does.
Splitting frames into a grid and upscaling only the blocks not seen before
would catch that, at a price: each block has to be upscaled with a margin of
its neighbours, or the stitched frame shows seams, and those margins are paid
again on every block. Whether it wins depends on the footage, so measure first:

```bash
# FFmpeg only: GPU work for each grid, against today's whole-frame dedup
python scripts/measure_tile_reuse.py input.mp4 --grid 3 4 6 --duration 60

# also upscale one frame whole and as blocks, and compare the two
python scripts/measure_tile_reuse.py input.mp4 --grid 3 --seam-test
```

On the test fixture `tests/fixtures/held.mp4`, ten drawings held three frames
each at 320x240, it prints:

```
strategy                     GPU work  vs none  vs today
no dedup                       30.0 f     100%
whole frames (today)           10.0 f      33%         -
3x3 blocks, margin 16          10.3 f      34%       +3%
4x4 blocks, margin 16           7.2 f      24%      -28%
```

GPU work is upscaled area counted in frames. A grid can lose to whole frames,
as 3x3 does here: on a frame this small, a 16-pixel margin more than doubles
the area of a block.

The seam test reports PSNR against the whole-frame upscale for margins 0, 8, 16
and 32, and writes each stitched frame next to a brightened difference image in
`work/tile-test/`, where a seam shows as lines along the block edges.

### Limiting how hard the GPU works

```bash
python -m videoenhance.cli input.mp4 --gpu-load 50
```

`--gpu-load` takes 1-100 (default 100). The NCNN tools have no throttle of their
own, so the process is suspended for the rest of each half-second cycle: at 50
the card works half the time and idles the other half. Output is identical;
only the run takes longer, in return for a cooler and quieter card.

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
change the pixels — `--gpu`, `--tile`, `--gpu-load`, `--keep-temp` — are free to differ.

## Development

```bash
pytest -q
```

Project conventions and the full design live in [CLAUDE.md](CLAUDE.md).
