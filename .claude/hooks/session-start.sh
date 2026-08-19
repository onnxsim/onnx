#!/bin/bash
# Ensures ccache + Ninja are available and wired in as the default CMake
# compiler launcher / generator, so C++ extension builds (pip install -e .,
# via scikit-build-core) are cached and parallelized by default in web
# sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if ! command -v ccache >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq ccache ninja-build
fi

ccache --max-size=5G >/dev/null 2>&1 || true

{
  echo 'export CMAKE_GENERATOR=Ninja'
  echo 'export CMAKE_C_COMPILER_LAUNCHER=ccache'
  echo 'export CMAKE_CXX_COMPILER_LAUNCHER=ccache'
  echo "export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)"
  # scikit-build-core filters CMAKE_ARGS to a known-safe allowlist, so also
  # set SKBUILD_CMAKE_ARGS (its unfiltered equivalent, semicolon-separated)
  # to guarantee the launcher flags reach cmake.
  echo 'export CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"'
  echo 'export SKBUILD_CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache;-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"'
} >> "$CLAUDE_ENV_FILE"
