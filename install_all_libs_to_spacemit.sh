#!/usr/bin/env bash
# install_all_libs_to_spacemit.sh (final, SSH alias 'banana')
# Purpose: Populate SpaceMIT sysroot with shared libs, headers, and pkg-config files
#          required for OpenCV + FFmpeg + SDL2 + V4L2 + CSI-NN2 based cross-builds.
# Notes:
#   - Copies from a target RISCV device via SSH alias from ~/.ssh/config (default: 'banana').
#   - Idempotent: safe to run multiple times.
#   - Keep this script as the single place to add missing deps (libs/headers/*.pc).
#   - To ensure SDL2 development files (sdl2.pc) exist on device, run with REMOTE_INSTALL_DEVEL=on.
#   - Comments MUST be in English.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/env.sh

# --- Configuration (env-overridable) ---
SSH_TARGET="${DEVICE_SSH_NAME}"
SYSROOT="${SPACEMIT_SYSROOT:-/opt/spacemit/sysroot}"
REMOTE_INSTALL_DEVEL="${REMOTE_INSTALL_DEVEL:-auto}"

REMOTE_LIB="/usr/lib/riscv64-linux-gnu"
REMOTE_LIB_ALT="/lib/riscv64-linux-gnu"
REMOTE_INC="/usr/include"
REMOTE_PKG_RV64="${REMOTE_LIB}/pkgconfig"
REMOTE_PKG_USR="/usr/lib/pkgconfig"
REMOTE_PKG_SHARE="/usr/share/pkgconfig"

LOG_FILE="/tmp/spacemit_sysroot_install_$(date +%Y%m%d_%H%M%S).log"

# SSH options (non-interactive by default; override via SSH_OPTS)
SSH_OPTS_DEFAULT="-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new"
SSH_OPTS="${SSH_OPTS:-$SSH_OPTS_DEFAULT}"

run_ssh() { ssh ${SSH_OPTS} "${SSH_TARGET}" "$@"; }

# SUDO helper: prefer passwordless sudo; otherwise fall back to no sudo (many sysroots are user-writable)
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  SUDO="sudo"
else
  SUDO=""
fi

# --- Helpers ---
info() { echo -e "\033[0;32m[$(date +'%H:%M:%S')]\033[0m $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*" | tee -a "$LOG_FILE"; }
fail() { echo -e "\033[0;31m[ERR]\033[0m  $*" | tee -a "$LOG_FILE"; exit 1; }

require_ssh() {
  if ! run_ssh "echo ok" >/dev/null 2>&1; then
    fail "Cannot SSH to ${SSH_TARGET} (check keys or ~/.ssh/config alias)"
  fi
}

mkdirs() {
  ${SUDO} mkdir -p \
    "${SYSROOT}${REMOTE_LIB}" \
    "${SYSROOT}/lib/riscv64-linux-gnu" \
    "${SYSROOT}${REMOTE_INC}/opencv4" \
    "${SYSROOT}${REMOTE_INC}/riscv64-linux-gnu" \
    "${SYSROOT}${REMOTE_INC}/SDL2" \
    "${SYSROOT}${REMOTE_INC}/libdrm" \
    "${SYSROOT}${REMOTE_INC}/libv4l2" >/dev/null 2>&1 || true

  # pkg-config dirs inside sysroot
  ${SUDO} mkdir -p \
    "${SYSROOT}${REMOTE_PKG_RV64}" \
    "${SYSROOT}/usr/lib/pkgconfig" \
    "${SYSROOT}/usr/share/pkgconfig"
}

# Copy files and/or directories matched by shell globs on the remote side.
# $1: remote dir; $2: space-separated glob patterns; $3: local dest
copy_globs() {
  local rdir="$1"; local globs="$2"; local ldest="$3"
  # Build a tiny remote script to expand globs and tar them (if any).
  local script="
    set -Eeuo pipefail
    cd '$rdir'
    shopt -s nullglob dotglob
    arr=( $globs )
    if [ \${#arr[@]} -gt 0 ]; then
      tar -czf - \"\${arr[@]}\"
    fi
  "
  # If nothing matches, tar receives no input; we swallow the error with '|| true'.
  printf '%s' "$script" | run_ssh 'bash -s' | ${SUDO} tar xzf - -C "$ldest" 2>/dev/null || true
}

# Copy a whole subdirectory from remote dir into local dest
# $1: remote dir; $2: subdir; $3: local dest
copy_dir() {
  local rdir="$1"; local sub="$2"; local ldest="$3"
  run_ssh "cd '$rdir' && tar -czf - '$sub' 2>/dev/null" | ${SUDO} tar xzf - -C "$ldest" 2>/dev/null || true
}

# Copy pkg-config files for critical libs
copy_pkgconfig() {
  local -a PC_SETS=(
    "${REMOTE_PKG_RV64}:libav*.pc libsw*.pc libpostproc.pc"
    "${REMOTE_PKG_RV64}:opencv4.pc"
    "${REMOTE_PKG_RV64}:sdl2.pc"
    "${REMOTE_PKG_RV64}:libv4l2.pc"
    "${REMOTE_PKG_RV64}:drm.pc"
    "${REMOTE_PKG_RV64}:x11.pc xcb.pc xkbcommon.pc wayland-client.pc wayland-egl.pc"
    "${REMOTE_PKG_USR}:sdl2.pc"
    "${REMOTE_PKG_SHARE}:opencv4.pc"
  )
  for entry in "${PC_SETS[@]}"; do
    local from="${entry%%:*}"; local globs="${entry#*:}"
    if run_ssh "[ -d '${from}' ]"; then
      copy_globs "${from}" "${globs}" "${SYSROOT}${from}"
    fi
  done
}

link_if_missing() {
  # $1: dir; $2: target; $3: linkname
  if [ -e "$1/$2" ] && [ ! -e "$1/$3" ]; then
    (cd "$1" && ${SUDO} ln -sf "$2" "$3")
  fi
}

# --- Start ---
info "=== SpaceMIT sysroot dependency installer ==="
info "Target sysroot: ${SYSROOT}"
info "Source device : ${SSH_TARGET}"
info "Log file      : ${LOG_FILE}"

[ -d "${SYSROOT}" ] || fail "Sysroot not found: ${SYSROOT}"
require_ssh
mkdir -p "$(dirname "$LOG_FILE")"
mkdirs

# Optionally ensure -dev packages exist on device (Debian/Ubuntu only, passwordless sudo)
ensure_remote_devel_packages() {
  if run_ssh "[ -d '${REMOTE_INC}/opencv4' ]" && \
     run_ssh "[ -d '${REMOTE_PKG_RV64}' ]" && \
     run_ssh "ls '${REMOTE_PKG_RV64}' | grep -Eq '^(opencv4|sdl2|libavcodec|libavformat|libavutil|libswscale)\\.pc$'"; then
    return 0
  fi
  case "${REMOTE_INSTALL_DEVEL}" in
    0|off|false) info "Skipping remote -dev installation (REMOTE_INSTALL_DEVEL=${REMOTE_INSTALL_DEVEL})"; return 0 ;;
    auto|1|on|true) : ;; # proceed
    *) warn "Unknown REMOTE_INSTALL_DEVEL='${REMOTE_INSTALL_DEVEL}', proceeding with auto" ;;
  esac
  if ! run_ssh "command -v apt-get >/dev/null 2>&1"; then
    warn "apt-get not found on device; cannot install -dev packages automatically"
    return 0
  fi
  if ! run_ssh 'sudo -n true' >/dev/null 2>&1 && ! run_ssh 'test "$(id -u)" -eq 0'; then
    warn "No root or passwordless sudo on device; cannot install -dev packages automatically"
    return 0
  fi
  info "Installing -dev packages on device (requires network on device)"
  local pkgs="build-essential pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libpostproc-dev libavdevice-dev libavfilter-dev libsdl2-dev libv4l-dev libdrm-dev libx11-dev libxcb1-dev libxkbcommon-dev libwayland-dev libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev libopencv-dev"
  run_ssh 'DEBIAN_FRONTEND=noninteractive sudo -n apt-get update -y' || warn "apt-get update failed on device"
  run_ssh "DEBIAN_FRONTEND=noninteractive sudo -n apt-get install -y ${pkgs}" || warn "apt-get install (dev packages) failed on device"
}

ensure_remote_devel_packages

# 1) Core & math
info "[1/8] Core system + math"
copy_globs "${REMOTE_LIB}" "libz.so* libbz2.so* liblzma.so* libatomic.so* libpthread*.so* libdl*.so* libm*.so* libgomp.so* libgcc_s.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libz.so* libbz2.so* liblzma.so* libatomic.so* libpthread*.so* libdl*.so* libm*.so* libgomp.so* libgcc_s.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
copy_globs "${REMOTE_LIB}" "blas/* lapack/* libgfortran.so* libquadmath.so*" "${SYSROOT}${REMOTE_LIB}"

# 2) Image formats
info "[2/8] Image formats"
copy_globs "${REMOTE_LIB}" "libjpeg*.so* libpng*.so* libtiff*.so* libwebp*.so* libopenjp2*.so* libjbig*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libjpeg*.so* libpng*.so* libtiff*.so* libwebp*.so* libopenjp2*.so* libjbig*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"

# 3) OpenCV
info "[3/8] OpenCV libs + headers"
copy_globs "${REMOTE_LIB}" "libopencv*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libopencv*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
if run_ssh "[ -d '${REMOTE_INC}/opencv4' ]"; then
  copy_dir "${REMOTE_INC}" "opencv4" "${SYSROOT}${REMOTE_INC}"
fi

# 4) FFmpeg
info "[4/8] FFmpeg libs + headers"
copy_globs "${REMOTE_LIB}" "libavcodec*.so* libavformat*.so* libavutil*.so* libavdevice*.so* libavfilter*.so* libswscale*.so* libswresample*.so* libpostproc*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libavcodec*.so* libavformat*.so* libavutil*.so* libavdevice*.so* libavfilter*.so* libswscale*.so* libswresample*.so* libpostproc*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
# headers (Debian-style)
if run_ssh "[ -d '${REMOTE_INC}/riscv64-linux-gnu/libavcodec' ]"; then
  copy_globs "${REMOTE_INC}/riscv64-linux-gnu" "libav* libsw*" "${SYSROOT}${REMOTE_INC}/riscv64-linux-gnu"
fi
# headers (flat /usr/include/ffmpeg)
if run_ssh "[ -d '${REMOTE_INC}/ffmpeg' ]"; then
  copy_dir "${REMOTE_INC}" "ffmpeg" "${SYSROOT}${REMOTE_INC}"
fi

# 5) SDL2, V4L2, graphics stack
info "[5/8] SDL2, V4L2, Wayland/X11/DRM"
copy_globs "${REMOTE_LIB}" "libSDL2*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libSDL2*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
copy_globs "${REMOTE_LIB}" "libv4l2*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libv4l2*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
copy_globs "${REMOTE_LIB}" "libdrm*.so* libX11*.so* libxcb*.so* libXau*.so* libXdmcp*.so* libXext*.so* libXfixes*.so* libXrender*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libdrm*.so* libX11*.so* libxcb*.so* libXau*.so* libXdmcp*.so* libXext*.so* libXfixes*.so* libXrender*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
copy_globs "${REMOTE_LIB}" "libwayland-client*.so* libwayland-egl*.so* libxkbcommon*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libwayland-client*.so* libwayland-egl*.so* libxkbcommon*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"

# headers
[ -d "${SYSROOT}${REMOTE_INC}/SDL2" ] || copy_dir "${REMOTE_INC}" "SDL2" "${SYSROOT}${REMOTE_INC}"
[ -d "${SYSROOT}${REMOTE_INC}/libdrm" ] || copy_dir "${REMOTE_INC}" "libdrm" "${SYSROOT}${REMOTE_INC}"
# V4L2 headers (libv4l-dev or kernel headers)
run_ssh "[ -f '${REMOTE_INC}/libv4l2.h' ]" && copy_globs "${REMOTE_INC}" "libv4l2.h libv4l-plugin.h" "${SYSROOT}${REMOTE_INC}" || true
run_ssh "[ -f '${REMOTE_INC}/linux/videodev2.h' ]" && copy_globs "${REMOTE_INC}" "linux/videodev2.h" "${SYSROOT}${REMOTE_INC}" || true

# 6) System/crypto (common deps)
info "[6/8] System/crypto"
copy_globs "${REMOTE_LIB}" "libbsd*.so* libmd*.so* libglib-2.0*.so* libgobject*.so* libgio*.so* libgmodule*.so* libxml2*.so* libcairo*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libbsd*.so* libmd*.so* libglib-2.0*.so* libgobject*.so* libgio*.so* libgmodule*.so* libxml2*.so* libcairo*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"
copy_globs "${REMOTE_LIB}" "libgcrypt*.so* libgmp*.so* libnettle*.so* libhogweed*.so* libtasn1*.so* libidn2*.so* libp11-kit*.so* libunistring*.so* libffi*.so* libgnutls*.so*" "${SYSROOT}${REMOTE_LIB}"
copy_globs "${REMOTE_LIB_ALT}" "libgcrypt*.so* libgmp*.so* libnettle*.so* libhogweed*.so* libtasn1*.so* libidn2*.so* libp11-kit*.so* libunistring*.so* libffi*.so* libgnutls*.so*" "${SYSROOT}${REMOTE_LIB_ALT}"

# 7) Optional codecs (if present on target)
info "[7/8] Optional codecs (x264/x265/vpx/xvid/opus)"
copy_globs "${REMOTE_LIB}" "libx264*.so* libx265*.so* libvpx*.so* libxvidcore*.so* libopus*.so*" "${SYSROOT}${REMOTE_LIB}"

# 8) pkg-config (*.pc)
info "[8/8] pkg-config files"
copy_pkgconfig

# Create common linker-friendly symlinks
info "Creating common symlinks"
cd "${SYSROOT}${REMOTE_LIB}"
link_if_missing "$(pwd)" "libpng16.so.16" "libpng.so"
link_if_missing "$(pwd)" "libjpeg.so.8"  "libjpeg.so"
link_if_missing "$(pwd)" "libtiff.so.6"  "libtiff.so"
link_if_missing "$(pwd)" "libwebp.so.7"  "libwebp.so"
link_if_missing "$(pwd)" "libz.so.1"     "libz.so"
link_if_missing "$(pwd)" "libbz2.so.1.0" "libbz2.so"
link_if_missing "$(pwd)" "libSDL2-2.0.so.0" "libSDL2.so"
# BLAS/LAPACK (if present)
if target=$(ls blas/libblas.so.3* 2>/dev/null | head -n1); then
  link_if_missing "$(pwd)" "$target" "libblas.so.3"
  link_if_missing "$(pwd)" "libblas.so.3" "libblas.so"
fi
if target=$(ls lapack/liblapack.so.3* 2>/dev/null | head -n1); then
  link_if_missing "$(pwd)" "$target" "liblapack.so.3"
  link_if_missing "$(pwd)" "liblapack.so.3" "liblapack.so"
fi

# Resolve and copy FFmpeg runtime dependencies discovered via ldd (recursive)
info "Resolving FFmpeg shared library dependencies via ldd"
copy_ldd_deps() {
  local so="$1"
  local list
  list=$(run_ssh "ldd '$so' | awk '{for(i=1;i<=NF;i++){if(\$i==\"=>\"&&i+1<=NF){print \$(i+1)} }}' | sed -n 's@^/@@p'" || true)
  [ -z "$list" ] && return 0
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    if run_ssh "[ -f '/$rel' ]"; then
      # Stage to tmp then install with sudo to sysroot
      local stage="/tmp/sysroot_stage"
      mkdir -p "${stage}/$(dirname "$rel")"
      scp ${SSH_OPTS} "${SSH_TARGET}:/$rel" "${stage}/$rel" >/dev/null 2>&1 || true
      if [ -f "${stage}/$rel" ]; then
        ${SUDO} install -D -m 0644 "${stage}/$rel" "${SYSROOT}/$rel" || true
      fi
    fi
  done <<< "$list"
}

for so in \
  "${REMOTE_LIB}/libavformat.so" \
  "${REMOTE_LIB}/libavcodec.so" \
  "${REMOTE_LIB}/libavutil.so" \
  "${REMOTE_LIB}/libswscale.so" \
  "${REMOTE_LIB}/libswresample.so" \
  "${REMOTE_LIB}/libpostproc.so" \
  "${REMOTE_LIB}/libavdevice.so" \
  "${REMOTE_LIB}/libavfilter.so" ; do
  if run_ssh "[ -f '$so' ]"; then
    info "Collecting dependencies for $(basename "$so")"
    copy_ldd_deps "$so"
  fi
done

# Verify pkg-config from sysroot
info "Verifying pkg-config visibility (if available)"
if command -v pkg-config >/dev/null 2>&1; then
  PKG_CONFIG_LIBDIR="${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}${REMOTE_PKG_RV64}:${SYSROOT}/usr/share/pkgconfig" \
  PKG_CONFIG_SYSROOT_DIR="${SYSROOT}" \
  pkg-config --exists sdl2 libavformat libavcodec libavutil libswscale opencv4 && \
  info "pkg-config: OK (sdl2/ffmpeg/opencv4 found)" || warn "pkg-config: some packages not found (check *.pc copies)"
else
  warn "pkg-config not installed on host; skipping validation"
fi

# Tiny compile test (optional)
info "Trying a tiny compile test (optional)"
cat > /tmp/test_sysroot.cpp << 'EOF'
#include <iostream>
#include <opencv2/core.hpp>
extern "C" {
#include <libavutil/version.h>
}
int main() {
  std::cout << "OpenCV " << CV_VERSION << " | libavutil "
            << LIBAVUTIL_VERSION_MAJOR << "." << LIBAVUTIL_VERSION_MINOR << "\n";
  return 0;
}
EOF

if /opt/spacemit/bin/riscv64-unknown-linux-gnu-g++ \
  -march=rv64gcv_zvfh -mabi=lp64d -O2 \
  /tmp/test_sysroot.cpp \
  --sysroot="${SYSROOT}" \
  $(PKG_CONFIG_LIBDIR="${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}${REMOTE_PKG_RV64}" \
    PKG_CONFIG_SYSROOT_DIR="${SYSROOT}" \
    pkg-config --cflags opencv4 libavutil 2>/dev/null || true) \
  -L"${SYSROOT}${REMOTE_LIB}" -Wl,-rpath-link="${SYSROOT}${REMOTE_LIB}" \
  -lopencv_core -lavutil \
  -latomic -lpthread -ldl -lm -static-libgcc -static-libstdc++ \
  -o /tmp/test_sysroot 2>/tmp/test_sysroot.log; then
  info "✓ Tiny compile test succeeded"
  rm -f /tmp/test_sysroot
else
  warn "Tiny compile test failed (see /tmp/test_sysroot.log)"
fi

info "=== Done. Sysroot is populated. ==="
info "Tip: export PKG_CONFIG_LIBDIR='${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}${REMOTE_PKG_RV64}'"
info "     export PKG_CONFIG_SYSROOT_DIR='${SYSROOT}'"
