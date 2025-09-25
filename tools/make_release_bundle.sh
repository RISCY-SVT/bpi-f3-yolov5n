#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------------------------
#  bpi-f3-yolov5n — release bundle maker (v2)
# ------------------------------------------------------------------------------
#  Packs a redistributable bundle with:
#   - stripped riscv64 binary (+ optional .debug if objcopy is available)
#   - README.md, CHANGELOG.md, Release_Notes_<version>.md
#   - run scripts: run_pipe.sh, run_live_remote*.sh, run_bench_summary.sh, env.sh
#   - model assets: coco.names (+ optional cpu_model/hhb.bm)
#   - MANIFEST.tsv, SHA256SUMS
#   - optional COMMIT_MESSAGE.md (provided or autodetected)
#
#  Usage:
#    tools/make_release_bundle.sh -v v0.4.2 [--include-model] [-o outdir]
#                                 [--binary build/yolov5n_pipeline]
#                                 [--commit-msg /path/Commit_*.md]
#                                 [--notes Release_Notes_v0.4.2.md]
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

# shellcheck disable=SC1091
if [[ -f "${PROJECT_ROOT}/env.sh" ]]; then
  # non-fatal; used only for PATH/toolchain convenience
  source "${PROJECT_ROOT}/env.sh" || true
fi

# --- args ---------------------------------------------------------------------
VERSION=""
OUTDIR=""
BINARY="build/yolov5n_pipeline"
INCLUDE_MODEL="false"
COMMIT_MSG_PATH=""
RELEASE_NOTES_PATH=""

usage() {
  cat <<EOF
Usage:
  $0 -v vX.Y.Z [--include-model] [-o outdir]
               [--binary build/yolov5n_pipeline]
               [--commit-msg /path/Commit_*.md]
               [--notes Release_Notes_vX.Y.Z.md]

Options:
  -v, --version        Release version tag, e.g. v0.4.2 (required)
  -o, --outdir         Output directory for bundle (default: <project>/release)
      --binary         Path to built binary (default: build/yolov5n_pipeline)
      --include-model  Include cpu_model/hhb.bm into bundle (default: off)
      --commit-msg     Path to commit message markdown to include
      --notes          Path to release notes (default: Release_Notes_<version>.md
                        or Release_Notes_v<version>.md)
  -h, --help           Show this help
EOF
}

# --- parse args ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)      VERSION="$2"; shift 2;;
    -o|--outdir)       OUTDIR="$2"; shift 2;;
    --binary)          BINARY="$2"; shift 2;;
    --include-model)   INCLUDE_MODEL="true"; shift 1;;
    --commit-msg)      COMMIT_MSG_PATH="$2"; shift 2;;
    --notes)           RELEASE_NOTES_PATH="$2"; shift 2;;
    -h|--help)         usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

# --- validation ---------------------------------------------------------------
if [[ -z "${VERSION}" ]]; then
  echo "ERROR: --version is required (e.g., -v v0.4.2)" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="${PROJECT_ROOT}/release"
fi

if [[ ! -f "${BINARY}" ]]; then
  echo "ERROR: binary not found: ${BINARY}. Build first (e.g., make -s pipeline ENABLE_SDL=1)." >&2
  exit 1
fi

# Resolve release notes path:
if [[ -z "${RELEASE_NOTES_PATH}" ]]; then
  # accept both with and without leading 'v'
  for cand in "Release_Notes_${VERSION}.md" "Release_Notes_v${VERSION#v}.md"; do
    if [[ -f "${cand}" ]]; then RELEASE_NOTES_PATH="${cand}"; break; fi
  done
  if [[ -z "${RELEASE_NOTES_PATH}" ]]; then
    echo "ERROR: release notes not found (tried Release_Notes_${VERSION}.md and Release_Notes_v${VERSION#v}.md)." >&2
    exit 1
  fi
fi

# Autodetect commit message if not provided
if [[ -z "${COMMIT_MSG_PATH}" ]]; then
  COMMIT_MSG_PATH="$(ls -1t "${PROJECT_ROOT}"/Commit_*.md 2>/dev/null | head -n1 || true)"
fi

PKG_DIR="${OUTDIR}/${VERSION}"
BIN_DIR="${PKG_DIR}/bin"
DOC_DIR="${PKG_DIR}/docs"
SCR_DIR="${PKG_DIR}/scripts"
MOD_DIR="${PKG_DIR}/model"

rm -rf "${PKG_DIR}"
mkdir -p "${BIN_DIR}" "${DOC_DIR}" "${SCR_DIR}" "${MOD_DIR}"

# --- binary (strip + optional .debug) -----------------------------------------
cp -f "${BINARY}" "${BIN_DIR}/yolov5n_pipeline"
chmod +x "${BIN_DIR}/yolov5n_pipeline"

have_objcopy="false"
for c in riscv64-unknown-linux-gnu-objcopy objcopy; do
  if command -v "$c" >/dev/null 2>&1; then have_objcopy="true"; OC="$c"; break; fi
done

if [[ "${have_objcopy}" == "true" ]]; then
  "${OC}" --only-keep-debug "${BIN_DIR}/yolov5n_pipeline" "${BIN_DIR}/yolov5n_pipeline.debug" || true
  strip_cmd="$(command -v riscv64-unknown-linux-gnu-strip || command -v strip || true)"
  if [[ -n "${strip_cmd}" ]]; then
    "${strip_cmd}" --strip-debug --strip-unneeded "${BIN_DIR}/yolov5n_pipeline" || true
  fi
fi

# --- docs ---------------------------------------------------------------------
cp -f "${PROJECT_ROOT}/README.md"        "${DOC_DIR}/README.md"
cp -f "${PROJECT_ROOT}/CHANGELOG.md"     "${DOC_DIR}/CHANGELOG.md"
cp -f "${RELEASE_NOTES_PATH}"            "${DOC_DIR}/Release_Notes_${VERSION}.md"

# --- scripts ------------------------------------------------------------------
cp -f "${PROJECT_ROOT}/run_pipe.sh"                  "${SCR_DIR}/run_pipe.sh"
cp -f "${PROJECT_ROOT}/tools/run_bench_summary.sh"   "${SCR_DIR}/run_bench_summary.sh" || true
cp -f "${PROJECT_ROOT}/tools/run_live_remote.sh"     "${SCR_DIR}/run_live_remote.sh" || true
cp -f "${PROJECT_ROOT}/tools/run_live_remote_inner.sh" "${SCR_DIR}/run_live_remote_inner.sh" || true
cp -f "${PROJECT_ROOT}/tools/memsnap.sh"             "${SCR_DIR}/memsnap.sh" || true
cp -f "${PROJECT_ROOT}/env.sh"                       "${SCR_DIR}/env.sh"

chmod +x "${SCR_DIR}/"*.sh 2>/dev/null || true

# --- model assets -------------------------------------------------------------
cp -f "${PROJECT_ROOT}/coco.names"    "${MOD_DIR}/coco.names"
if [[ "${INCLUDE_MODEL}" == "true" ]]; then
  mkdir -p "${MOD_DIR}/cpu_model"
  cp -f "${PROJECT_ROOT}/cpu_model/hhb.bm" "${MOD_DIR}/cpu_model/hhb.bm"
fi

# --- optional commit message --------------------------------------------------
if [[ -n "${COMMIT_MSG_PATH}" && -f "${COMMIT_MSG_PATH}" ]]; then
  cp -f "${COMMIT_MSG_PATH}" "${PKG_DIR}/COMMIT_MESSAGE.md"
fi

# --- MANIFEST + SHA256 --------------------------------------------------------
(
  cd "${PKG_DIR}"
  # MANIFEST.tsv: <sha256>\t<size>\t<path>
  : > MANIFEST.tsv
  while IFS= read -r -d '' f; do
    rel="${f#./}"
    sz="$(stat -c '%s' "$rel")"
    sha="$(sha256sum "$rel" | awk '{print $1}')"
    printf "%s\t%s\t%s\n" "$sha" "$sz" "$rel" >> MANIFEST.tsv
  done < <(find . -type f -print0 | LC_ALL=C sort -z)

  awk -F'\t' -v ORS='\0' '{print $3}' MANIFEST.tsv | xargs -0 sha256sum -- > SHA256SUMS
)

# --- archives -----------------------------------------------------------------
(
  cd "${OUTDIR}"
  tar_name="bpi-f3-yolov5n_${VERSION}_riscv64.tar.gz"
  zip_name="bpi-f3-yolov5n_${VERSION}_riscv64.zip"
  tar -czf "${tar_name}" "${VERSION}"
  (command -v zip >/dev/null 2>&1 && zip -r "${zip_name}" "${VERSION}") || true
)

echo "Release bundle created at: ${PKG_DIR}"
echo "Archives: ${OUTDIR}/bpi-f3-yolov5n_${VERSION}_riscv64.tar.gz (and .zip if zip is available)"