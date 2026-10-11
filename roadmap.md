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
- [x] `test(upscale): add wrapper tests with a stub binary` — assert the
      generated argument list without touching the GPU.

## M4 — Interpolate (RIFE NCNN)

- [x] `feat(interpolate): add rife ncnn wrapper` — call `rife-ncnn-vulkan`
      with the model directory and `-f %08d.png`.
- [x] `feat(interpolate): compute target frame count` —
      `input_frames * target_fps / original_fps`, as the `-n` argument.
- [x] `feat(interpolate): enable UHD mode above 4K` — pass `-u` when the
      output resolution warrants it.
- [x] `test(interpolate): add frame count and argument tests`.

## M5 — Pipeline, resume & UX

- [x] `feat(pipeline): wire full extract-upscale-interpolate-encode` — the
      default order, upscaling before the frame count is multiplied. The work
      directory, the resume rule and the disk estimate moved to `workdir.py`
      so `pipeline.py` can import the stages without a cycle.
- [x] `feat(pipeline): support --order interpolate-first` — the alternative
      order, interpolating at the original resolution.
- [x] `feat(cli): add tqdm progress reporting` — per-stage progress derived
      from the frame counts on disk, in `progress.py`; off outside a terminal
      and with `--no-progress`.
- [x] `feat(pipeline): clean work dir unless --keep-temp` — clean up only
      after a successful run.
- [x] `test(pipeline): add stage-skipping resume tests` — a pre-filled output
      directory must skip its stage.
- [x] `fix(pipeline): never resume from another order's frames` — the GPU
      stages require an exact frame count to skip, and clear their output
      directory before re-running. Found while testing the two orders: running
      one after the other doubled the length of the video with no error.

## M6 — Docs, packaging & polish

- [x] `docs: document bin/ installation` — which release to download for each
      OS and where the model folders go.
- [x] `docs: add troubleshooting for VRAM and audio sync` — the pitfalls from
      CLAUDE.md, written as symptom to fix.
- [x] `chore: add scripts/fetch-bins.sh` — optional helper that downloads and
      unpacks the NCNN releases into `bin/`.
- [x] `chore(ci): add lint and test workflow` — run `ruff` and `pytest` on
      push, without a GPU.

## M7 — Chunked processing

Goal: peak disk that does not grow with the length of the video. Found by
running the pipeline on real footage: a 3-second clip already wanted 128 MB of
frames, which puts a feature film in the hundreds of gigabytes.

- [x] `feat(chunk): cut the source into chunks` — `chunk.py` with `split()`,
      cutting with `-f segment -c copy` into video-only chunks plus a
      manifest, normalising a VFR source to CFR once beforehand, and re-cutting
      losslessly any chunk that came out more than twice the requested length
      because the source's keyframes were too sparse.
- [x] `feat(chunk): run the pipeline once per chunk` — `run_chunked()`, which
      drives `run_pipeline` into one part per chunk, reuses a single
      `current/` scratch directory (guarded by a marker file so a resume
      cannot mix two chunks' frames) and deletes each chunk's frames and
      source as soon as its part exists.
- [x] `feat(encode): concatenate the parts and mux the audio` — `concat()`,
      joining the parts by stream copy and muxing the original audio exactly
      once, with the encoder pinned to the first part's so the copy is valid.
- [x] `feat(cli): add --chunk and --encoder` — chunking on by default at 30 s,
      `--chunk 0` for the single-pass path, the disk estimate built from one
      chunk, and `--encoder` to force one when the chain would otherwise
      differ between parts.
- [x] `test(chunk): cover the cut, the resume and the cleanup` — the chunks
      adding back up to the source, no audio in a chunk, the re-cut fallback,
      the encoder pin, and that a finished run leaves no frames behind.

## M8 — Saved jobs

Goal: stop and continue days later, without remembering the command and without
corrupting the result. Found while testing M7: resuming with a different
`--scale` reuses the old frames, because every stage guard is a frame count and
the count does not change with the scale.

- [x] `feat(state): record the job in state.json` — `state.py` with the record,
      an atomic write, lookup by id or unambiguous prefix, disk-derived progress
      and the option comparison. Only what cannot be derived is stored.
- [x] `refactor(chunk): keep the cut in the job record` — `manifest.json` folded
      into `state.json`, and `ChunkPlan` now carries the updated record so a
      later save cannot wipe the cut back out.
- [x] `feat(cli): add --jobs, --resume and --restart` — the listing, resume by
      id, and the refusal when the options or the input file changed. The
      pixel-affecting flags lose their argparse defaults so that "asked for" and
      "left alone" can be told apart.
- [x] `test(state): cover the record, the lookup and the refusal` — including
      the regression test for the silent `--scale` change, in both modes.

## M9 — Generate each frame once

Goal: stop paying the GPU twice for the same pixels. The expensive stage is
deterministic, so a repeated frame is repeated work.

- [x] `feat(dedup): hash frames and upscale each one once` — `dedup.py` with
      pixel hashes from `ffmpeg -f framemd5`, hard-linked duplicates, and a
      cache keyed by hash plus the settings that change the pixels, wrapping
      `upscale()` rather than replacing it so the VRAM retry still applies.
- [x] `feat(cli): add --no-dedup and --cache-size` — on by default, with the
      saving reported per run because it depends on the source, and a ceiling on
      the cache so it cannot grow with the length of the video.
- [x] `test(dedup): cover the hashing, the cache and the equivalence` —
      including a flat-artwork fixture whose held frames survive a lossy encode,
      and a run compared frame by frame against `--no-dedup`.

Measured on 90 s of 640x360 at 2x/60 fps with each drawing held three frames:
3 min 45 s to 2 min 32 s, `frames_up/` from 36 MB to 12 MB, every output frame
identical. With no repeats, +1%.

## M10 — Generate each block once

Goal: reuse the upscaled parts of a frame that only partly changed, such as a
mouth moving over a static background, where M9 finds no repeat at all. The
grid and the margin are set in a config file, `videoenhance.toml`, which this
milestone introduces, and can be overridden on the command line.

The design below is decided; the tasks implement it in order. Read all of it
before the first task, because the later tasks depend on choices made in the
earlier ones.

### Design

**Off by default.** `blocks = 1` means whole frames only, which is exactly
M9's behaviour. Block mode can do *more* GPU work than whole frames, because
every block carries a margin (on the 320x240 fixture a 3x3 grid did 3% more
work and a 4x4 grid 28% less). The user turns it on after measuring with
`scripts/measure_tile_reuse.py`.

**Every frame goes through blocks, not only the changed ones.** The upscaler's
output at a pixel depends on its neighbours, so a block upscaled on its own
differs at its edges from the same pixels upscaled inside the whole frame. If
fresh frames were upscaled whole and reused ones stitched from blocks, the
picture would shift slightly between the two and flicker. When every frame is
stitched from blocks, a reused block is bit-identical to a recomputed one, so
reuse stays lossless *within block mode*. The output does differ slightly from
`blocks = 1`, which is why both settings are pixel options (task 4).

**The cache key of a block is the hash of the block with its margin.** A block
whose neighbour changed has different context, so it must be redone, or the
seam shows. `blocks.grid_tiles()` (task 2) already computes the padded region,
clamped at the frame edges.

**Blocks are cached by content, not by position.** The same padded pixels
anywhere in any frame are one cache entry. Edge blocks have a different shape
from middle ones, so their hashes cannot collide with them.

**Blocks are always PNG, whatever `--frame-format` says.** A block is cropped,
upscaled, cropped again and stitched. With JPEG every one of those steps would
add loss. Only the stitched frame is written in the requested format.

**Whole-frame dedup still runs first.** A frame whose hash is already in the
frame cache is linked as today. Only the distinct frames that miss it are split
into blocks. The stitched result then goes into the frame cache, so the next
chunk finds it whole.

**Few processes, no Python pixel handling.** Hashing, cropping and stitching
are each one FFmpeg run over a sequence, never one per frame: thousands of
FFmpeg starts would cost more than the GPU saves. No frame is read into Python,
as CLAUDE.md requires.

**Defaults:** `margin = 16`. On the fixture, margin 0 measured 47.7 dB against
the whole-frame upscale, with seams along every block edge, and 8 and above
measured over 60 dB.

### Algorithm, for `upscale_deduplicated` when `blocks > 1`

1. `index_frames(frames_in)`, as today. The distinct frame hashes missing from
   the frame cache are the frames to build.
2. Hard-link one representative of each such frame into `scratch/frames/`,
   numbered from 1.
3. `block_hashes()` over `scratch/frames/` gives, for each of the N² blocks,
   one hash per frame. This is one FFmpeg pass.
4. Walk frames x blocks. Each hash that is neither in the block cache nor
   already scheduled is scheduled once, with the frame and block it came from.
5. `crop_blocks()` writes every padded block of every frame in
   `scratch/frames/`, one sequence per block position, in one FFmpeg pass.
   Move the scheduled ones into `scratch/todo/` numbered 1..K in schedule
   order, and delete the rest. Cropping everything and discarding is the
   simple version: a `select` expression per block grows with the frame count.
6. One `upscale(scratch/todo, scratch/todo_up, frame_format="png", force=True)`
   call. Mixed block sizes in one directory are fine, because the binary
   processes each image on its own. Then `block_cache.put(hash, todo_up/rank)`.
7. For each block position k, build a link farm `scratch/stitch/<k>/%08d.png`,
   where frame i points at the block cache entry for that frame's hash at
   position k. Then one `stitch_blocks()` run reads the N² sequences, crops
   the margin off each and assembles the frames with `xstack` into
   `scratch/stitched/` in `frame_format`. Then
   `frame_cache.put(frame_hash, stitched/i)`.
8. Link every frame of `frames_out` from the frame cache, exactly as today.
9. Trim both caches against `--cache-size` together, **only at the end**: a
   trim between steps 4 and 7 could delete a block that step 7 needs.

### Tasks

- [x] `fix(dedup): cache frames under the tile that made them` — a
      pre-existing bug, fixed first because block mode inherits it.
      `FrameCache` is built with `options.tile`, but `upscale()` can lower the
      tile on a VRAM retry and return it as `result.tile`. The frames are then
      stored in the slot of a tile that did not make them, which is the one
      thing the tile in the key exists to prevent. Store new results in the
      slot of `result.tile`, for example via a `FrameCache.with_tile(tile)`
      that returns the cache for another slot, and link `frames_out` from
      wherever each frame actually is. Hits from the requested tile stay valid,
      because they were made at that tile. Test: the existing VRAM-retry stub
      in `tests/test_upscale.py` that only succeeds at tile 128, run through
      `upscale_deduplicated`, must leave its frames in the `t128` slot and
      nothing in `t0`.
- [x] `refactor(blocks): move the block geometry into the package` — new
      `src/videoenhance/blocks.py` holding `Tile` and `grid_tiles()` from the
      script, plus three FFmpeg helpers that take a list of FFmpeg *input
      arguments*, so that the same code serves a video window
      (`["-ss", "5", "-t", "30", "-i", video]`) and a frame directory
      (`["-framerate", "1", "-i", dir / "%08d.png"]`):
      `block_hashes(inputs, tiles, bins) -> (frame_hashes, per_block_hashes)`,
      `crop_blocks(inputs, tiles, out_dirs, bins)`, which writes one
      `%08d.png` sequence per block, and
      `stitch_blocks(block_dirs, tiles, scale, out_pattern, bins)`. Keep the
      `format=rgb24` before the crop, so a block can start at an odd pixel
      without chroma rounding. `scripts/measure_tile_reuse.py` then imports
      all of it and keeps only `summarize()` and the printing; its seam test
      is the 1-frame case of the same functions. Move the geometry and
      stitching tests from `tests/test_tile_reuse.py` to
      `tests/test_blocks.py`, and add one for a sequence: crop 5 frames,
      stitch at scale 1, every frame back with PSNR `inf`. The script's output
      must not change.
- [x] `feat(config): read options from videoenhance.toml` — new
      `src/videoenhance/configfile.py`. Lookup order: `--config PATH` (an
      error if missing), then `$VIDEOENHANCE_CONFIG`, then
      `PROJECT_ROOT / "videoenhance.toml"` if it exists, else no file. Parse
      it with `tomllib`. On Python 3.10, which has no `tomllib`, try `tomli`
      and otherwise fail only when a file was actually found, with a message
      saying so; do **not** add a dependency. The only table for now is:

      ```toml
      [dedup]
      # Blocks per side. 1 = whole frames only (default); 3 = a 3x3 grid.
      blocks = 3
      # Pixels of neighbours upscaled with each block, so stitching leaves no
      # seams. Measure with scripts/measure_tile_reuse.py --seam-test.
      margin = 16
      ```

      `load()` returns `{"blocks": 3, "block_margin": 16}`, keyed by argparse
      dest. Reject unknown tables and keys by name (a typo must not be
      silently ignored), non-integers, `blocks < 1` and `margin < 0`, each with
      the file path in the message. Commit `videoenhance.example.toml` with
      the comments above and gitignore `videoenhance.toml`, which is personal.
      Tests use `tmp_path` files and `monkeypatch` for the variable and for
      `config.PROJECT_ROOT`.
- [x] `feat(state): record blocks and margin as pixel options` — add
      `blocks` and `block_margin` to `PIXEL_OPTIONS` and `OPTION_FLAGS`
      (`--blocks`, `--block-margin`). Two rules make the comparison honest:
      - **Normalise:** `options_of()` records `block_margin = 0` whenever
        `blocks == 1`, because the margin does nothing then, and changing it
        must not refuse a resume.
      - **Legacy values:** a `state.json` written before M10 has neither key,
        and `compare()` currently skips absent keys. That would let a
        whole-frame job be resumed in block mode with no warning, which is the
        exact silent mix the record exists to stop. Add
        `LEGACY_VALUES = {"blocks": 1, "block_margin": 0}` and compare an
        absent key against it.

      Tests: an old record without the keys refuses `blocks=3` and accepts
      `blocks=1`; with `blocks=1` the margin is free to change.
- [x] `feat(dedup): upscale only the blocks that changed` — the algorithm
      above, as a block branch inside `upscale_deduplicated`, which gains
      `blocks: int = 1` and `margin: int = 0` parameters. `blocks == 1` must
      take the existing code path untouched. Other changes:
      - **Options:** `Options` gains `blocks` and `block_margin`, and
        `_upscale_stage` passes them through.
      - **Frame cache:** the slot name gains `-b{blocks}m{margin}` whenever
        `blocks > 1`. A stitched frame is not the same pixels as a whole one,
        and the M9 slot names must not change, or existing caches stop hitting.
      - **Block cache:** a second `FrameCache` under the same root, slot
        `blocks-{model}-{scale}x-t{tile}-m{margin}`. Both slots are trimmed
        against one combined budget. Generalise `trim()` into a function over
        several caches rather than giving each its own half.
      - **Scratch:** the new directories live under `work.dedup` and are
        cleared before use, as the existing `in`/`out` are.
      - **Stats:** `DedupStats` gains `blocks` (N, 1 meaning off),
        `blocks_total` and `blocks_upscaled`, and `work`, which is the
        upscaled area divided by one frame's area, as the script measures it.
        `render()` in block mode reads like `3x3 blocks: 412 of 2700
        generated, 61.4 frames of work for 300 (80% saved)`. The saving is
        against no dedup, like M9's figure.

      Tests need a stub that really scales: the copying stub in `conftest.py`
      cannot be stitched at scale 2. Add `STUB_SCALING`, which runs
      `ffmpeg -i <in> -vf scale=iw*S:ih*S:flags=neighbor <out>` per file,
      reading `S` from `-s`. Nearest-neighbour is purely local, so it gives
      the strongest test available: **block mode must reproduce the
      whole-frame stub output exactly**, PSNR `inf` on every frame. Add a
      fixture written straight to PNG (no lossy encode): 240x240 gray, with a
      small box moving only inside the top-left 80x80 block and staying more
      than the margin away from its edges. With `blocks=3, margin=8` the first
      frame sends 9 blocks and every later distinct frame exactly 1, which can
      be asserted from the stub's argument log and the `todo/` count. Also
      test: a second run of the same frames sends nothing to the stub and
      gives identical frames; `frame_format="jpg"` still upscales PNG blocks
      (assert on the stub's `-f`); `blocks=1` runs the M9 path byte for byte.
- [x] `feat(cli): add --config, --blocks and --block-margin` — `--blocks`
      and `--block-margin` join `RESOLVED_ARGS` and `ARG_TO_OPTION`, with
      defaults 1 and 16 in `config.py` (`DEFAULT_BLOCKS`,
      `DEFAULT_BLOCK_MARGIN`). Precedence: typed flag > recorded job (on
      `--resume`) > config file > default. In `resolve_args`, a value taken
      from the config file is **not** "given", so a resume after editing the
      file uses the recorded value instead of conflicting with it. When a
      plain re-run finds a job whose blocks differ because the file changed,
      the mismatch message names the file as the source of the new value.
      Other changes:
      - **Validation:** after probing, reject a grid whose blocks would be
        under 32 px on a side, and warn when the margin is under 8 (the seams
        measured at 0) or when a frame that changes everywhere would cost more
        than +50%. `grid_tiles()` gives that figure.
      - **No dedup:** `--no-dedup` with `blocks > 1` prints that blocks need
        dedup and runs with `blocks = 1`. Normalise it before the job is
        recorded, so the record says what actually ran.
      - **Plan:** `describe_plan` shows the grid, the margin, and the config
        file when one was read.
      - **Disk estimate:** `estimate_disk_usage` adds the block scratch, an
        upper bound of one chunk's frames times the margin overhead, at both
        resolutions.

      Test the precedence chain, the resume after a config edit, and the
      validation messages.
- [x] `docs: document block dedup and the config file` — the README gets a
      section on `videoenhance.toml`, and turns "Would reusing blocks of a
      frame save more?" into how to measure, then enable it. CLAUDE.md gets
      `blocks.py` and `configfile.py` in the structure, and the three rules
      that are easy to break: every frame goes through blocks; blocks are PNG
      whatever the frame format; trim only at the end. Measure a real anime
      clip at 3x3 and 4x4 against
      `blocks = 1`, for wall time and GPU work, and record the figures under
      this milestone, as M9 did.

### Results

Measured on a synthetic clip, not on anime: 10 s of 960x540 at 30 fps, a
spatially varying background and one 60x60 box crossing the top of the picture,
encoded losslessly so that nothing but the box differs between frames and all
300 are distinct. `--scale 2 --fps 30 --chunk 0 --block-margin 16`, on the AMD
card, wall time of the whole run:

| | upscale | wall time |
|---|---|---|
| whole frames (`--blocks 1`) | 300 frames | 36.8 s |
| 3x3 | 392 of 2700 blocks, 51.0 frames of work | **23.7 s** (-36%) |
| 4x4 | 796 of 4800 blocks, 65.3 frames of work | 28.4 s (-23%) |

A clip with a static background is the case blocks are for, so this is close to
a best case, and 3x3 beat 4x4 because the margins grow faster than the
saving. A real anime clip has not been measured here and is the first thing to
do before turning it on: `scripts/measure_tile_reuse.py` does that without a
GPU. On `tests/fixtures/held.mp4` the same script says 3x3 loses to whole frames
(+3%) and 4x4 wins (-28%), so the answer depends on the footage and the frame
size, which is why it is off by default.

### Notes for the implementer

- **Definition of done:** run before every commit, all green:
  `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check src tests scripts`
  and `.venv/bin/ruff format --check src tests scripts`.
- **External calls:** every one goes through `process.run`, as an argument
  list. Pass `load=` through to `upscale()` too, so `--gpu-load` keeps
  applying.
- **Error checks:** check that FFmpeg returned one hash per frame per block,
  as `pixel_hashes` does for frames, and raise `DedupError` with the counts
  if not. A silent miscount would stitch the wrong block into a frame.
- **Escaping:** `xstack` needs `inputs=` and a `layout` of numeric `X_Y`
  offsets at the *upscaled* size. The `crop` that removes the margin is at
  `(x0 - px0) * scale`. `stitch()` in the script already does this right;
  move it, do not rewrite it.
- **Scope:** do not change the default (`blocks = 1`), the M9 slot names, or
  `pixel_hashes`.

## M11 — Encode while the GPU generates

Goal: stop the GPU idling while each part is encoded. Asked for after M10: the
GPU stages and the encode use different parts of the machine, so a chunked run
can encode part N while chunk N+1 is generated. Recorded after the fact.

- [x] `feat(process): cancel commands from another thread` — `cancel_all()`,
      which terminates every child `run()` started and makes it raise
      `Cancelled` instead of `ToolError`, so a fallback chain does not take a
      stopped tool for a broken one.
- [x] `refactor(pipeline): split run_pipeline into generate and finish` — the
      GPU half and the encode half, so the second can run beside the next first.
- [x] `feat(chunk): encode one chunk while the next is generated` — a single
      encode thread taking parts in order, two scratch slots, failures raised at
      the next chunk boundary, the pin read when an encode starts.
- [x] `feat(cli): add --no-overlap` — the estimate and the plan count two chunks
      when overlapping.

Measured on 10 s of 960x540 at 2x/60 fps in 2-second chunks: 89.2 s to 83.4 s
with `libx265`, 82.6 s to 81.0 s with `hevc_vaapi`, identical output. The ceiling
is the encode time, about 6 s for the whole result with `libx265` here, since
the GPU stages dominate on this machine.

## Backlog / future

- **Dedup for RIFE.** Interpolation reuses nothing yet. The unit there is the
  pair, not the frame, and `rife-ncnn-vulkan` takes a whole directory and
  reloads its model per invocation, so per-pair calls would likely cost more
  than they save. Needs two measurements first: the split of time between the
  two stages, and the model load cost per invocation.
- **Scene-cut detection:** use FFmpeg's `scdet` filter to find hard cuts and
  duplicate the frame there instead of interpolating, avoiding RIFE's ghost
  frames.
- **Optional ROCm path:** only if explicitly requested, and only on Linux with
  a ROCm-supported card.
- **Batch mode:** accept a directory of videos and process them in sequence.
- **Extract ahead too.** M11 overlaps the encode with the GPU, but the
  extraction of chunk N+1 still waits for the GPU to finish chunk N. It is
  CPU work as well and could run a chunk ahead, at the cost of a third slot.
- **HDR / 10-bit handling:** currently out of scope; PNG frames are 8-bit.
