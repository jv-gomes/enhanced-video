# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project overview

A Python command-line tool that takes a video and outputs an enhanced version:

1. **Super-resolution** (upscaling) with Real-ESRGAN.
2. **Frame interpolation** (higher FPS, e.g. 30 → 60) with RIFE.
3. Reassembly of the final video **preserving the original audio**.

## Key constraint: the GPU is AMD

The user's machine has an **AMD graphics card**. This defines the entire stack:

- **Do NOT use CUDA**, or any package that depends on it (`torch` with CUDA, `cupy`, `basicsr` via standard pip, `realesrgan` via pip, PyTorch-based `practical-rife`, `tensorrt`, CUDA `onnxruntime-gpu`).
- **Use the NCNN + Vulkan builds** of the models, which run on AMD, Intel and NVIDIA without PyTorch:
  - `realesrgan-ncnn-vulkan` — https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan
  - `rife-ncnn-vulkan` — https://github.com/nihui/rife-ncnn-vulkan
- The Python code is only an **orchestrator**: it calls FFmpeg and these executables via `subprocess`.
- ROCm + PyTorch should only be considered if the user explicitly asks for it, and only on Linux with a ROCm-supported card. It is not the default path.
- Before any heavy processing, verify that Vulkan sees the GPU (e.g. run `realesrgan-ncnn-vulkan` once and check the detected GPU in the log; on Linux, `vulkaninfo --summary`).

## Stack

- Python 3.10+ (standard library whenever possible; `tqdm` for progress is acceptable).
- FFmpeg and FFprobe on PATH.
- NCNN binaries in `bin/` (not committed to git; see "Setup").
- No PyTorch, no CUDA.

## Suggested structure

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── bin/                      # NCNN executables + model folders (gitignored)
│   ├── realesrgan/
│   └── rife/
├── src/videoenhance/
│   ├── __init__.py
│   ├── cli.py                # argparse, entry point
│   ├── config.py             # binary paths, defaults
│   ├── probe.py              # ffprobe: fps, resolution, duration, audio, VFR
│   ├── extract.py            # video -> frames
│   ├── upscale.py            # calls realesrgan-ncnn-vulkan
│   ├── interpolate.py        # calls rife-ncnn-vulkan
│   ├── encode.py             # frames + audio -> final video, concat of parts
│   ├── pipeline.py           # orchestrates the steps and the work directory
│   ├── chunk.py              # splits long videos, runs the pipeline per chunk
│   └── state.py              # the saved job: --jobs, --resume, --restart
├── tests/
└── work/                     # temporary frames (gitignored)
```

## Pipeline

Default order: **extract → upscale → interpolate → encode**.

Reason: upscaling is the most expensive step, so it runs before the frame count is multiplied. Keep the order configurable (`--order interpolate-first`) for those who prefer interpolating at the original resolution.

Videos longer than `--chunk` seconds (default 30, `0` disables) run that whole
sequence **once per chunk**, in `chunk.py`: cut the source with
`-f segment -c copy`, process one chunk into an encoded part, delete that
chunk's frames, then concatenate the parts and mux the original audio once at
the end. Peak disk is one chunk rather than the whole film. Two rules there are
load-bearing:

- **Never give a part its own audio.** A `-shortest` per part accumulates into
  audible drift; the audio is muxed a single time, at the concatenation.
- **Pin the encoder** to whatever the first part used, since the parts are
  joined by stream copy and the encoder chain may fall back at run time.

Every run records itself in `work/<job>/state.json` (`state.py`), which is what
`--jobs` lists and `--resume <id>` continues. The rule for that file: **it holds
only what cannot be derived from the files.** Settings, input, output and the
chunked cut go in; progress stays on disk, because a recorded count can
disagree with reality and a resume that lies is worse than one that is slow.

Its real job is to refuse a resume whose options changed. The stage guards are
frame counts (`stage_is_done`), and a frame count does not move when `--scale`
or `--model` does, so without the check a job restarted at a different scale
silently skips the upscale. `PIXEL_OPTIONS` in `state.py` is the list that may
not change; anything not in it (`--gpu`, `--tile`, `--keep-temp`) may.

### 1. Probe
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames \
  -of json input.mp4
```
- If `r_frame_rate` ≠ `avg_frame_rate`, the video is probably **VFR** (common in phone recordings). In that case, convert to CFR before extracting, otherwise the audio ends up out of sync.
- Check whether an audio stream exists (`-select_streams a`).

### 2. Extract frames
```bash
ffmpeg -i input.mp4 -fps_mode passthrough work/frames_in/%08d.png
```

### 3. Upscale
```bash
bin/realesrgan/realesrgan-ncnn-vulkan -i work/frames_in -o work/frames_up \
  -n realesr-animevideov3 -s 2 -f png -g 0 -t 0 -j 1:2:2
```
- Models: `realesrgan-x4plus` (real-world video, scale 4 only), `realesrgan-x4plus-anime` (anime), `realesr-animevideov3` (anime/video, scales 2/3/4, faster).
- `-t` = tile size. `0` = automatic. On out-of-VRAM errors, reduce to `256` or `128`.
- `-g` = GPU ID. If the PC has both an integrated and a dedicated GPU, make sure the dedicated AMD card is selected.

### 4. Interpolate
```bash
bin/rife/rife-ncnn-vulkan -i work/frames_up -o work/frames_out \
  -m bin/rife/rife-v4.6 -n <target_frame_count> -f %08d.png -g 0 -j 1:2:2
```
- `-n` = total number of output frames. Compute as `input_frames * (target_fps / original_fps)`.
- For 4K or larger output, add `-u` (UHD mode).

### 5. Encode
Re-attach the audio from the original file:
```bash
ffmpeg -framerate <target_fps> -i work/frames_out/%08d.png -i input.mp4 \
  -map 0:v -map 1:a? -c:a copy -shortest <ENCODER> output.mp4
```
Choose the encoder based on the OS:
- **Windows (AMD AMF):** `-c:v hevc_amf -quality quality -rc cqp -qp_i 18 -qp_p 20` (or `h264_amf`)
- **Linux (VAAPI):** `-vaapi_device /dev/dri/renderD128 -vf format=nv12,hwupload -c:v hevc_vaapi -qp 20`
- **Fallback (CPU):** `-c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p`

Detect availability with `ffmpeg -hide_banner -encoders` and fall back if the hardware encoder fails.

## Setup

1. Install FFmpeg (on Windows, a "full" build that includes AMF).
2. Download the `realesrgan-ncnn-vulkan` and `rife-ncnn-vulkan` releases for the correct OS and extract them into `bin/realesrgan/` and `bin/rife/` along with their model folders.
3. Update the AMD driver (Adrenalin on Windows; recent Mesa/RADV on Linux) to ensure Vulkan support.
4. `pip install -r requirements.txt`

## Commands

```bash
# run
python -m videoenhance.cli input.mp4 -o output.mp4 --scale 2 --fps 60

# tests
pytest -q

# generate a short test video (no real file needed)
ffmpeg -f lavfi -i testsrc=size=320x240:rate=30 -f lavfi -i sine=frequency=440 \
  -t 3 -c:v libx264 -c:a aac tests/fixtures/sample.mp4
```

## Code conventions

- Every external call goes through a single helper (e.g. `run(cmd: list[str])`) that logs the command, captures stderr, and raises a clear exception on a non-zero exit code.
- Commands are always argument lists, never `shell=True`.
- Use `pathlib.Path` for all paths (must work on both Windows and Linux).
- Binary paths come from `config.py` and can be overridden by environment variables (`REALESRGAN_BIN`, `RIFE_BIN`).
- Every step must be **resumable**: if the output folder already contains all expected frames, skip the step. Long videos take hours and the user should not lose progress if something crashes. A new option that changes the pixels must be added to `PIXEL_OPTIONS` in `state.py`, or resuming with it changed will silently reuse work made the old way.
- Provide a `--keep-temp` option to keep `work/`; by default, clean up after a successful run.

## Known pitfalls

- **Disk space:** PNG frames take up a lot of space. One minute of 1080p at 60 FPS after upscaling can exceed tens of GB. Chunked mode is the answer and is on by default, so the cost is one chunk rather than the whole film; `--chunk 0` opts out. Estimate the required space at startup and warn the user. Offer `--frame-format jpg` as an option (smaller, slightly lossy).
- **VRAM:** Vulkan allocation errors are almost always solved by lowering `-t` in Real-ESRGAN or disabling `-u` in RIFE.
- **Audio out of sync:** almost always a VFR video. See the probe step.
- **Scene cuts:** RIFE can produce "ghost" frames on hard transitions. Future improvement: detect cuts with FFmpeg's `scdet` filter and duplicate the frame instead of interpolating at those points.
- **Windows:** NCNN executables end in `.exe`; resolve the name based on `sys.platform`.

## What to avoid

- Do not add CUDA or PyTorch dependencies without an explicit request.
- Do not load all frames into memory with OpenCV; always work with files on disk and let the binaries process the folders.
- Do not overwrite the input video.