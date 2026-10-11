#!/usr/bin/env bash
# Download the two NCNN executables and their models into bin/.
#
#   scripts/fetch-bins.sh                 # this machine's OS, into ./bin
#   scripts/fetch-bins.sh --dry-run       # say what would be fetched, touch nothing
#   scripts/fetch-bins.sh --only rife --force
#
# The two zips are laid out differently, which is the reason this exists:
#
#   Real-ESRGAN   flat: the binary and models/ at the top. It comes from the
#                 xinntao/Real-ESRGAN repository, because the releases of
#                 Real-ESRGAN-ncnn-vulkan hold the binary alone, without models.
#   RIFE          one wrapper folder (rife-ncnn-vulkan-<date>-<os>/) holding the
#                 binary and every model folder, about 430 MB for all of them.
#
# Both end up as bin/realesrgan/ and bin/rife/ with the binary beside its models,
# which is where config.py looks. Windows users without bash can follow the manual
# steps in the README; Git Bash runs this script as it is.

set -euo pipefail

REALESRGAN_VERSION="v0.2.5.0"
REALESRGAN_BUILD="20220424"
RIFE_VERSION="20221029"
#: The only RIFE model the command line uses; the rest of the zip is optional.
RIFE_DEFAULT_MODEL="rife-v4.6"

REALESRGAN_URL_BASE="${REALESRGAN_URL_BASE:-https://github.com/xinntao/Real-ESRGAN/releases/download/${REALESRGAN_VERSION}}"
RIFE_URL_BASE="${RIFE_URL_BASE:-https://github.com/nihui/rife-ncnn-vulkan/releases/download/${RIFE_VERSION}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/bin"
TARGET_OS=""
ONLY=""
FORCE=0
DRY_RUN=0
ALL_RIFE_MODELS=0

usage() {
  cat <<'EOF'
Usage: scripts/fetch-bins.sh [options]

  --os ubuntu|windows|macos   release to fetch (default: detected from this machine)
  --dest DIR                  where bin/ goes (default: bin/ in the project)
  --only realesrgan|rife      fetch just one of the two
  --all-rife-models           keep every RIFE model (default: only rife-v4.6)
  --force                     fetch again even if it is already there
  --dry-run                   print what would happen and change nothing
  -h, --help                  this text
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --os) TARGET_OS="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --all-rife-models) ALL_RIFE_MODELS=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

if [[ -z "$TARGET_OS" ]]; then
  case "$(uname -s)" in
    Linux) TARGET_OS="ubuntu" ;;
    Darwin) TARGET_OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) TARGET_OS="windows" ;;
    *) die "cannot tell which release to fetch for $(uname -s); pass --os" ;;
  esac
fi
case "$TARGET_OS" in ubuntu|windows|macos) ;; *) die "--os must be ubuntu, windows or macos" ;; esac
case "$ONLY" in ""|realesrgan|rife) ;; *) die "--only must be realesrgan or rife" ;; esac

EXE=""
[[ "$TARGET_OS" == "windows" ]] && EXE=".exe"

REALESRGAN_ZIP="realesrgan-ncnn-vulkan-${REALESRGAN_BUILD}-${TARGET_OS}.zip"
RIFE_ZIP="rife-ncnn-vulkan-${RIFE_VERSION}-${TARGET_OS}.zip"

have_realesrgan() {
  [[ -f "$DEST/realesrgan/realesrgan-ncnn-vulkan${EXE}" && -d "$DEST/realesrgan/models" ]]
}
have_rife() {
  [[ -f "$DEST/rife/rife-ncnn-vulkan${EXE}" && -d "$DEST/rife/${RIFE_DEFAULT_MODEL}" ]]
}

TMP=""
cleanup() { [[ -n "$TMP" ]] && rm -rf "$TMP"; return 0; }
trap cleanup EXIT

download() {  # url file
  echo "  downloading $1"
  # A progress bar redraws one line over and over, which fills a log with noise
  # unless a person is looking at a terminal.
  local quiet=(--silent --show-error)
  [[ -t 2 ]] && quiet=(--progress-bar)
  curl --fail --location --retry 3 "${quiet[@]}" --output "$2" "$1" \
    || die "could not download $1"
  [[ -s "$2" ]] || die "the download of $1 is empty"
}

unpack() {  # zip dir
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$1" -d "$2"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$1" "$2"
  else
    die "need unzip or python3 to open $1"
  fi
}

fetch_realesrgan() {
  if have_realesrgan && [[ $FORCE -eq 0 ]]; then
    echo "realesrgan: already in $DEST/realesrgan (use --force to fetch again)"
    return
  fi
  echo "realesrgan: ${REALESRGAN_ZIP} -> $DEST/realesrgan"
  [[ $DRY_RUN -eq 1 ]] && { echo "  from ${REALESRGAN_URL_BASE}/${REALESRGAN_ZIP}"; return; }
  download "${REALESRGAN_URL_BASE}/${REALESRGAN_ZIP}" "$TMP/realesrgan.zip"
  unpack "$TMP/realesrgan.zip" "$TMP/realesrgan"
  [[ -f "$TMP/realesrgan/realesrgan-ncnn-vulkan${EXE}" && -d "$TMP/realesrgan/models" ]] \
    || die "the Real-ESRGAN zip does not have the binary and models/ at its top; layout changed?"
  rm -rf "$DEST/realesrgan"
  mkdir -p "$DEST/realesrgan"
  # The zip also carries demo images and a video; only these two are needed.
  mv "$TMP/realesrgan/realesrgan-ncnn-vulkan${EXE}" "$TMP/realesrgan/models" "$DEST/realesrgan/"
  chmod +x "$DEST/realesrgan/realesrgan-ncnn-vulkan${EXE}"
}

fetch_rife() {
  if have_rife && [[ $FORCE -eq 0 ]]; then
    echo "rife: already in $DEST/rife (use --force to fetch again)"
    return
  fi
  echo "rife: ${RIFE_ZIP} -> $DEST/rife   (about 430 MB)"
  [[ $DRY_RUN -eq 1 ]] && { echo "  from ${RIFE_URL_BASE}/${RIFE_ZIP}"; return; }
  download "${RIFE_URL_BASE}/${RIFE_ZIP}" "$TMP/rife.zip"
  unpack "$TMP/rife.zip" "$TMP/rife"
  # One wrapper folder, named after the release; its contents are what we want.
  local inner
  inner="$(find "$TMP/rife" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [[ -n "$inner" && -f "$inner/rife-ncnn-vulkan${EXE}" && -d "$inner/${RIFE_DEFAULT_MODEL}" ]] \
    || die "the RIFE zip does not hold rife-ncnn-vulkan and ${RIFE_DEFAULT_MODEL} in one folder; layout changed?"
  rm -rf "$DEST/rife"
  mkdir -p "$DEST/rife"
  if [[ $ALL_RIFE_MODELS -eq 1 ]]; then
    mv "$inner"/* "$DEST/rife/"
  else
    mv "$inner/rife-ncnn-vulkan${EXE}" "$inner/${RIFE_DEFAULT_MODEL}" "$DEST/rife/"
    [[ -f "$inner/LICENSE" ]] && mv "$inner/LICENSE" "$DEST/rife/"
  fi
  chmod +x "$DEST/rife/rife-ncnn-vulkan${EXE}"
}

if [[ $DRY_RUN -eq 0 ]]; then
  command -v curl >/dev/null 2>&1 || die "curl is needed to download the releases"
  TMP="$(mktemp -d)"
  mkdir -p "$DEST"
fi

[[ -z "$ONLY" || "$ONLY" == "realesrgan" ]] && fetch_realesrgan
[[ -z "$ONLY" || "$ONLY" == "rife" ]] && fetch_rife

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry run: nothing was downloaded"
else
  echo
  echo "done. Check the setup with:  python -m videoenhance.cli --doctor"
fi
