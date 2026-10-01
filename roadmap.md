# Roadmap

Implementation plan for `videoenhance`, the AMD/Vulkan video enhancement CLI
described in [CLAUDE.md](CLAUDE.md).

Tasks are deliberately small: **one task = one commit**. Milestones are ordered
so that something runnable exists as early as possible, and so the expensive
GPU work (M3, M4) only lands after the cheap end-to-end plumbing (M2) is proven.

## Conventions

- **Commit style:** [Conventional Commits](https://www.conventionalcommits.org/),
  English, scoped by module: `feat(probe):`, `fix(encode):`, `test(pipeline):`,
  `chore:`, `docs:`.
- **One commit per completed task.** Each checkbox below carries the commit
  message to use.
- **Definition of done** for a task: the code runs, `pytest -q` is green, and
  nothing in `bin/` or `work/` has been committed.
- **No new heavy dependencies.** Standard library first; `tqdm` is the only
  runtime dependency. No PyTorch, no CUDA (see CLAUDE.md, "What to avoid").

---

## M0 — Repo foundation

- [x] `chore: add gitignore and project guidelines` — ignore `bin/`, `work/`,
      generated fixtures and the usual Python/editor noise; rename
      `Claude.md` to `CLAUDE.md`.
- [x] `docs: add implementation roadmap` — this file.
- [x] `docs: add README with setup and usage` — stub covering install,
      binary download and the basic command; expanded in M6.

## M1 — Core scaffolding & probe

Goal: a runnable CLI that can inspect a video and tell the user whether their
machine is ready, before any GPU code exists.

- [x] `chore: add src layout package and pyproject` — `src/videoenhance/`,
      `pyproject.toml` (setuptools, src layout, `requires-python = ">=3.10"`,
      `tqdm`), `requirements.txt`. Makes
      `python -m videoenhance.cli` work after `pip install -e .`.
- [x] `feat(process): add central subprocess run helper` — `process.py` with
      `run(cmd: list[str])` and `ToolError`. Logs the command, captures stderr,
      raises with the stderr tail on a non-zero exit. Never `shell=True`.
      Every external call in the project goes through this.
- [x] `feat(config): add binary paths and platform resolution` — `config.py`
      with the `bin/realesrgan/` and `bin/rife/` defaults, `.exe` suffix
      resolution via `sys.platform`, environment overrides
      (`REALESRGAN_BIN`, `RIFE_BIN`, `FFMPEG_BIN`, `FFPROBE_BIN`) and the
      default model / tile / thread settings.
- [x] `feat(probe): add ffprobe metadata reader` — `probe.py` with
      `probe(path) -> VideoInfo` (width, height, `r_frame_rate`,
      `avg_frame_rate`, fps, frame count, duration, `has_audio`, `is_vfr`).
      VFR is detected by comparing the two frame rates as fractions.
- [x] `feat(cli): add entry point with --probe-only` — `cli.py` argparse
      surface (`input`, `-o/--output`, `--scale`, `--fps`, `--order`,
      `--frame-format`, `--keep-temp`, `--gpu`, `-v`) plus `--probe-only`,
      which prints the probe result and exits.
- [x] `feat(cli): add doctor preflight check` — `doctor.py` and `--doctor`:
      check ffmpeg/ffprobe, list the usable encoders, report the Vulkan GPU
      and report whether the NCNN binaries are installed, with actionable
      messages and a non-zero exit when something is missing.
- [x] `test(probe): add fixture video and probe tests` — `tests/conftest.py`
      generating `tests/fixtures/sample.mp4` with the `lavfi` command from
      CLAUDE.md (skipping if ffmpeg is unavailable), plus tests for `probe()`
      and for `run()` raising `ToolError`.
- [x] `test(cli): add config, doctor and cli tests` — cover the environment
      overrides and platform suffix, the doctor report's exit code and
      missing-binary hints, and the CLI's `--probe-only`, `--doctor` and
      argument validation paths.

## M2 — Extract & encode

Goal: a full round trip with no model in the loop. This is where audio sync
and disk usage problems surface, so it comes before the GPU steps.

- [x] `feat(pipeline): add work directory and resume helper` — the `work/`
      layout plus `frames_complete(dir, expected)`, so any stage whose output
      directory already holds every expected frame is skipped.
- [x] `feat(extract): add frame extraction with VFR handling` — extract with
      `-fps_mode passthrough`; when the probe reports VFR, convert to CFR
      first so the audio does not drift.
- [x] `feat(encode): add encoder detection with fallback chain` — parse
      `ffmpeg -hide_banner -encoders` and pick VAAPI on Linux, AMF on
      Windows, `libx264` as the fallback; fall back automatically when a
      hardware encoder fails at runtime.
- [x] `feat(encode): assemble frames with original audio` — mux the frame
      sequence with the original audio stream (`-map 1:a? -c:a copy
      -shortest`), never overwriting the input file.
- [x] `feat(cli): warn on insufficient disk space` — estimate the frame
      footprint from resolution, frame count and scale; warn before starting
      and mention `--frame-format jpg`. Adds `--dry-run`, which prints the
      planned run, its encoder chain and its disk cost, then exits.
- [x] `test(e2e): add extract-encode round-trip test` — extract and re-encode
      the fixture, then assert duration and audio presence survive.

## M3 — Upscale (Real-ESRGAN NCNN)

- [x] `feat(upscale): add realesrgan ncnn wrapper` — call
      `realesrgan-ncnn-vulkan` on the input frame directory.
- [x] `feat(upscale): add model and tile-size options` — `--model`
      (`realesr-animevideov3`, `realesrgan-x4plus`, `realesrgan-x4plus-anime`),
      `--tile`, `--gpu`, with scale values validated per model.
- [x] `feat(upscale): retry with smaller tile on VRAM error` — detect Vulkan
      allocation failures and retry at 256 then 128 before giving up.
- [ ] `test(upscale): add wrapper tests with a stub binary` — assert the
      generated argument list without touching the GPU.

## M4 — Interpolate (RIFE NCNN)

- [ ] `feat(interpolate): add rife ncnn wrapper` — call `rife-ncnn-vulkan`
      with the model directory and `-f %08d.png`.
- [ ] `feat(interpolate): compute target frame count` —
      `input_frames * target_fps / original_fps`, as the `-n` argument.
- [ ] `feat(interpolate): enable UHD mode above 4K` — pass `-u` when the
      output resolution warrants it.
- [ ] `test(interpolate): add frame count and argument tests`.

## M5 — Pipeline, resume & UX

- [ ] `feat(pipeline): wire full extract-upscale-interpolate-encode` — the
      default order, upscaling before the frame count is multiplied.
- [ ] `feat(pipeline): support --order interpolate-first` — the alternative
      order, interpolating at the original resolution.
- [ ] `feat(cli): add tqdm progress reporting` — per-stage progress derived
      from the frame counts on disk.
- [ ] `feat(pipeline): clean work dir unless --keep-temp` — clean up only
      after a successful run.
- [ ] `test(pipeline): add stage-skipping resume tests` — a pre-filled output
      directory must skip its stage.

## M6 — Docs, packaging & polish

- [ ] `docs: document bin/ installation` — which release to download for each
      OS and where the model folders go.
- [ ] `docs: add troubleshooting for VRAM and audio sync` — the pitfalls from
      CLAUDE.md, written as symptom to fix.
- [ ] `chore: add scripts/fetch-bins.sh` — optional helper that downloads and
      unpacks the NCNN releases into `bin/`.
- [ ] `chore(ci): add lint and test workflow` — run `ruff` and `pytest` on
      push, without a GPU.

## Backlog / future

- **Scene-cut detection:** use FFmpeg's `scdet` filter to find hard cuts and
  duplicate the frame there instead of interpolating, avoiding RIFE's ghost
  frames.
- **Optional ROCm path:** only if explicitly requested, and only on Linux with
  a ROCm-supported card.
- **Batch mode:** accept a directory of videos and process them in sequence.
- **HDR / 10-bit handling:** currently out of scope; PNG frames are 8-bit.
