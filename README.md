## Notes
To avoid rebuilding LLVM every time, we do not just
use `add_subdirectory()` for it because that’s expensive; rather, you have to build LLVM first, and only then this project.

1. Get LLVM
```bash
$ git submodule update --init --recursive
```

2. Fetch the version you want to use
```bash
$ cd llvm
$ git fetch origin llvmorg-VERSION --depth 1 # e.g. `llvmord-16.0.6`
```

3. Checkout that version.
```bash
$ git checkout FETCH_HEAD
```

4. Build LLVM
```bash
$ cd ..
$ ./build-llvm.sh
```

