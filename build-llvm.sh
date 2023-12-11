#!/usr/bin/env bash

set -eu

die() {
    echo "$@" >&2
    exit 1
}

## Change into the LLVM directory.
cd llvm || die "Could not find LLVM directory."

## Use Clang and LLD to build LLVM. Build in Release mode because Debug mode
## produces a 2GB library, requires enormous amounts of RAM to link, and 30
## seconds to load into the debugger. Enable assertions so we know why it’s
## crashing on use, as well as assertions for llvm_unreachable(). Also enable
## dump(), so we can print pretty much anything for debugging. Don’t forget
## to enable MLIR and the other subprojects we want to build.
CC=clang CXX=clang++ cmake -G "Ninja" \
  -S llvm \
  -B out \
  -DCMAKE_INSTALL_PREFIX="$(realpath .)/llvm-install" \
  -DGCC_INSTALL_PREFIX=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;compiler-rt;mlir' \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_C_COMPILER=clang \
  -DLLVM_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=mold \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_APPEND_VC_REV=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_UNWIND_TABLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_UNREACHABLE_OPTIMIZE=OFF \
  -DLLVM_ENABLE_DUMP=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_DOXYGEN=ON \
  -DLLVM_ENABLE_FFI=ON \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF

## Build LLVM.
cmake --build out -- -j $((`nproc` - 2))
