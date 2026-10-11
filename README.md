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
2. Download the two NCNN releases for your OS and unpack them into `bin/`, as
   described under [Installing the NCNN binaries](#installing-the-ncnn-binaries).
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

### Installing the NCNN binaries

Both programs run on AMD, Intel and NVIDIA through Vulkan, and each needs its
model files beside it. This is the layout the tool looks for:

```
bin/
├── realesrgan/
│   ├── realesrgan-ncnn-vulkan        (.exe on Windows)
│   └── models/                       realesr-animevideov3-x2.bin, ...
└── rife/
    ├── rife-ncnn-vulkan              (.exe on Windows)
    └── rife-v4.6/                    flownet.bin, flownet.param
```

Pick the zip for your OS:

| | Linux | Windows | macOS |
|---|---|---|---|
| **Real-ESRGAN** | `realesrgan-ncnn-vulkan-20220424-ubuntu.zip` | `…-windows.zip` | `…-macos.zip` |
| **RIFE** | `rife-ncnn-vulkan-20221029-ubuntu.zip` | `…-windows.zip` | `…-macos.zip` |
| from | [Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) | same | same |
| from | [rife-ncnn-vulkan 20221029](https://github.com/nihui/rife-ncnn-vulkan/releases/tag/20221029) | same | same |

(macOS builds exist, but this project is developed and tested on Linux with an
AMD card.)

On Linux, macOS and Git Bash, `scripts/fetch-bins.sh` does all of this: it picks the
zip for your OS, fetches both releases, puts each into its folder with the right layout
and keeps only `rife-v4.6`. `--dry-run` shows what it would download first, `--only
realesrgan` or `--only rife` fetches one, `--all-rife-models` keeps the rest, and
`--force` fetches again over an existing install. The manual steps follow.

Two things catch people out:

- **Download Real-ESRGAN from `xinntao/Real-ESRGAN`, not from
  `Real-ESRGAN-ncnn-vulkan`.** The releases of the latter hold the executable
  alone, with no `models/` folder, and the tool then reports "missing models".
- **The two zips are laid out differently.** The Real-ESRGAN one is flat: unpack
  it straight into `bin/realesrgan/`. The RIFE one has a wrapper folder,
  `rife-ncnn-vulkan-20221029-ubuntu/`, whose *contents* belong in `bin/rife/`.

On Linux, for example:

```bash
mkdir -p bin/realesrgan bin/rife
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d bin/realesrgan
unzip rife-ncnn-vulkan-20221029-ubuntu.zip -d /tmp/rife
mv /tmp/rife/rife-ncnn-vulkan-20221029-ubuntu/* bin/rife/
chmod +x bin/realesrgan/realesrgan-ncnn-vulkan bin/rife/rife-ncnn-vulkan
```

The RIFE zip is about 430 MB because it carries every model. Only `rife-v4.6`
is needed: it is the default, and the command line has no option to pick another
one yet, so the rest can be deleted. The Real-ESRGAN zip also has two demo
images and a video that can go.

Executables that live somewhere else are found through `REALESRGAN_BIN` and
`RIFE_BIN`, and a `realesrgan-ncnn-vulkan` or `rife-ncnn-vulkan` on `PATH` is
used when `bin/` has none; the models are looked up beside the binary either way.
When done, `python -m videoenhance.cli --doctor` checks the whole setup.

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

#### Encoding while the next chunk is generated

Encoding a part needs no GPU compute, so by default it runs beside the next
chunk: while FFmpeg encodes part N, the GPU is already upscaling and
interpolating chunk N+1, instead of idling through every encode. The parts are
still encoded one at a time and in order, so they all come out of the same
encoder and can be joined.

The cost is disk: two chunks' frames exist at once, one being encoded and one
being generated, and the estimate in the plan counts both. `--no-overlap` goes
back to one chunk at a time when space is tight.

How much it saves is however long the encodes take, since that is what it
hides. On the development machine, 10 seconds of 960x540 upscaled 2x to 60 fps
in 2-second chunks went from 89.2 s to 83.4 s with `libx265` and from 82.6 s to
81.0 s with `hevc_vaapi`: the GPU stages dominate there, and encoding the whole
1080p60 result with `libx265` takes about 6 seconds, most of which the overlap
hid. A slower encoder or a larger output gives it more to hide.

Ctrl-C still stops both at once: the encode in flight is killed rather than
retried with the next encoder, and the parts already finished are kept for
`--resume`.

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

#### Reusing parts of a frame: `--blocks`

When only part of the picture moves — a mouth over a static background — no frame
repeats as a whole, yet most of it does. `--blocks N` splits every frame that is
not a repeat into an N x N grid and upscales only the blocks it has not seen
before. It is **off by default**, because it can cost more than it saves.

The reason is the margin. Real-ESRGAN's output at a pixel depends on its
neighbours, so a block upscaled alone differs at its edges from the same pixels
inside the whole frame. Each block is therefore upscaled with a margin of its
neighbours (`--block-margin`, 16 px by default) and the margin is cut off
afterwards; and a block is reused only if its margin is identical too. That
margin is paid again on every block, including on a frame that changes
everywhere, which is how a grid ends up doing *more* GPU work than whole frames
on footage with nothing static in it.

**Measure first.** This reads your video with FFmpeg alone and needs no GPU:

```bash
python scripts/measure_tile_reuse.py input.mp4 --grid 3 4 6 --duration 60
```

```
strategy                     GPU work  vs none  vs today
no dedup                       30.0 f     100%
whole frames (today)           10.0 f      33%         -
3x3 blocks, margin 16          10.3 f      34%       +3%
4x4 blocks, margin 16           7.2 f      24%      -28%
```

GPU work is upscaled area counted in frames; "vs today" below zero means less
work than the whole-frame dedup you already have. (That is
`tests/fixtures/held.mp4`, ten drawings held three frames each at 160x120: on a
frame that small a 16 px margin more than doubles a block, and 3x3 loses.) To see
what a margin does to the picture, add `--seam-test`, which upscales one frame
whole and as blocks and reports the PSNR for margins 0, 8, 16 and 32, with a
brightened difference image in `work/tile-test/`. On the fixture it read 47.7 dB
with no margin, with a visible line along every block edge, and over 60 dB from 8.

Then turn it on:

```bash
python -m videoenhance.cli input.mp4 --blocks 3 --block-margin 16
```

Every frame that reaches the GPU goes through blocks, the unchanged ones
included, because a frame upscaled whole next to one stitched from blocks would
shift slightly and flicker. The output is therefore very slightly different from
`--blocks 1` — about 59 dB against it at margin 16 — and the setting is one that
`--resume` refuses to change. The summary reports the real figure per run:

```
upscale: 300 frames, 2x with realesr-animevideov3; 3x3 blocks: 392 of 2700 generated, 51.0 frames of work for 300 (83% saved)
```

A grid whose blocks would be under 32 px on a side is refused, and a margin under
8 px, or a grid that would cost more than 50% extra on a frame that changes
everywhere, gets a warning. `--no-dedup` turns blocks off with a note.

#### The config file

The grid and the margin can live in `videoenhance.toml` instead of the command
line. Copy `videoenhance.example.toml`, which says what each setting means:

```toml
[dedup]
blocks = 3     # per side; 1 = whole frames only (default)
margin = 16    # pixels of neighbours upscaled with each block
```

It is looked for in this order: `--config PATH`, `$VIDEOENHANCE_CONFIG`, then
`videoenhance.toml` in the project root; having none is fine. A typed flag wins
over the file, and on `--resume` the recorded job wins over it, so editing the file
never makes a half-finished job refuse to continue. An unknown key is an error
that names it, since a typo that did nothing would leave you waiting for a saving.
Reading TOML needs Python 3.11, or `pip install tomli` on 3.10.

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

## Troubleshooting

Run `python -m videoenhance.cli --doctor` first. It checks FFmpeg, the encoders,
Vulkan, both executables and their models, and says what to do about each failure.

### The GPU runs out of memory

**Symptom:** `the GPU ran out of memory at every tile size tried (0, 256, 128)`,
or `vkAllocateMemory failed` / `out of device memory` in the output of Real-ESRGAN.

**Fix:** the tool already retries at tile sizes 256 and then 128 by itself. If that
was not enough, start lower with `--tile 64`, ask for a smaller `--scale`, and close
whatever else is using the card (a browser with hardware acceleration will do it).
On 4K-or-larger frames RIFE's UHD mode is the first thing to give way, and it is
retried without it automatically. `--frame-format jpg` does not help here: it saves
disk, not video memory.

### It is using the wrong GPU

**Symptom:** the run is far slower than it should be, or `--doctor` lists more than
one device (`+1 more; select with --gpu`).

**Fix:** a machine with an integrated and a dedicated GPU numbers them, and the
default is device 0. Real-ESRGAN prints the devices it sees as it starts, in the form
`[0 AMD Radeon RX 6650 XT (RADV NAVI23)]`, and the number in the brackets is what
`--gpu` takes: run with `-v`, or call `realesrgan-ncnn-vulkan` on any image, to see it.
Then pass the dedicated card's number, for example `--gpu 1`.

### Vulkan sees no GPU

**Symptom:** `--doctor` reports `[FAIL] vulkan: no Vulkan device found`.

**Fix:** update the driver: Adrenalin on Windows, a recent Mesa with RADV on Linux
(`vulkan-radeon` or `mesa-vulkan-drivers`, depending on the distribution), then check
that `vulkaninfo --summary` names your card.

### The card gets hot or loud

**Fix:** `--gpu-load 50` makes the upscaler and interpolator run half the time and
rest the other half, so the card cools between bursts. The output is identical and the
run takes longer. See [Limiting how hard the GPU works](#limiting-how-hard-the-gpu-works).

### The warning says the frames will not fit on the disk

**Symptom:** `warning: the intermediate frames need about 40.2 GB but only 12.0 GB is free.`

**Fix:** frames are PNG and large. Chunked processing is on by default and keeps the cost
to one chunk, so first check that the estimate is for a chunk (`--chunk 0` turns that
off). Then, in order of effect: a smaller `--chunk 10`, `--frame-format jpg` (about a
fifth of the size, slightly lossy), a smaller `--scale`, or `--work-dir` on a bigger
disk. `--dry-run` prints the estimate without processing anything.

### The audio drifts out of sync

**Symptom:** the sound and the picture agree at the start and part ways later.

**What it usually is:** a variable-frame-rate source, which is what phones record. Extracting
frames and re-encoding them at a constant rate shifts the audio. The tool normalises such a
source to a constant rate first, and the plan it prints says `(VFR)` when it did; the audio
is also muxed once, at the end, so chunks cannot accumulate an error between them.

**If it drifts anyway:** run `--probe-only` and compare the frame rates and the duration
with what you expect, then try `--chunk 0` on a short clip to see whether chunking is
involved, and keep the original file: that is a bug worth reporting with both outputs.

### Ghost or smeared frames at a cut

**Symptom:** a frame between two shots blends them.

**Why:** RIFE invents the frames between two others, and at a hard cut there is nothing
to blend. Detecting cuts and holding the frame instead is on the
[roadmap](roadmap.md) but not built.

**Workaround:** `--fps` set to the source's own rate skips interpolation entirely, so only
the upscale runs. Or split the video at the cuts that matter and process the parts apart.

### Visible seams or flicker with `--blocks`

**Symptom:** faint lines along a grid, or parts of the picture that shift between frames.

**Fix:** keep `--block-margin` at 8 or more (the default is 16): with none, the seams
measured 47.7 dB against a whole-frame upscale and were visible. If it persists, `--blocks 1`
returns to whole frames. The grid is a pixel option, so a job started with one value
cannot be resumed with another.

### `hevc_vaapi` or another encoder fails

**Symptom:** `encoded ... with libx264` where you expected hardware, or an `EncodeError`.

**Fix:** the encoder chain falls back by itself, hardware first and CPU last, so the video
is still produced. `--doctor` and `--dry-run` print the chain. To pick one, `--encoder
libx265` (or any encoder `ffmpeg -encoders` lists). On Linux the hardware encoders need
`/dev/dri/renderD128`, which `--doctor` checks.

### `error: this job was started with different options`

**Why:** the frames already on disk were made with the old options, and a frame count does
not change when `--scale` or `--model` does, so continuing would silently mix the two.

**Fix:** run again with the options the message lists, or `--restart` to discard the saved
work. `--jobs` lists what is unfinished; `--resume <id>` continues one. See
[Stopping and continuing later](#stopping-and-continuing-later).

### `missing models` or `not found` from `--doctor`

**Fix:** see [Installing the NCNN binaries](#installing-the-ncnn-binaries). The usual cause
is a Real-ESRGAN zip from the wrong repository, which has the executable and no `models/`.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts
pytest -q
```

The tests need FFmpeg and FFprobe, which build their own clips, and nothing else:
no GPU, and no NCNN binaries, which are replaced by stand-ins. The same two
commands run on every push and pull request in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), on Python 3.10 and 3.14.

Project conventions and the full design live in [CLAUDE.md](CLAUDE.md).
