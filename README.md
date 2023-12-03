## The Source Programming Language

Documentation forthcoming. See the `tests/` directory for examples, particularly `tests/aoc23/`.

## Building
Currently, only Linux is supported.

To avoid rebuilding LLVM every time, we do not just
use `add_subdirectory()` for it because that’s expensive; rather, you have to build LLVM first, and only then this project.

1. Get LLVM
```bash
$ git submodule update --init --recursive
```

2. Build LLVM
```bash
$ cd ..
$ ./build-llvm.sh
```

Alternatively, you can instead run `./build-llvm-debug.sh` to build LLVM in debug mode, but doing
so is *not* recommended unless you really need to debug some internal LLVM issue that you believe
you’ve run into. In that case, you’ll also have to point `SOURCE_LLVM_BUILD_DIR` in the `CMakeLists.txt`
to point to `out-debug` instead of `out`.

3. Build this project
```bash
$ cmake -G Ninja -S . -B out
$ cmake --build out
```

This will build the `srcc` executable in the project directory, as well as the standard library. Note that
the `cpptrace` library that is used for stack traces is a bit janky at times, so if you get compilation errors
about `CPPTRACE_EXPORT` not being defined or some header related to it not being found, manually edit
`libs/cpptrace/include/cpptrace/cpptrace.hpp` and replace the line that includes the header in question
with `#define CPPTRACE_EXPORT ` to define the macro to expand to nothing. 

## Testing
Currently, you have to build `fchk` manually. Go into the `fchk/` directory and use CMake to build it. Then
you can try running the tests.

