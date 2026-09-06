# Agent Instructions for FBGEMM

## What this is

Fork of the PyTorch FBGEMM project: highly-optimized low-precision matrix-multiplication and convolution kernels for server-side inference, plus `fbgemm_gpu` PyTorch GPU operator libraries for training/recommendation systems and generative AI.

This repo is a C++/CUDA/ROCm project with Python packaging, Bazel, and CMake builds. Most agents will interact with the CPU library (`src/`, `include/`, `test/`) or the Python GPU packages under `fbgemm_gpu/`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/` | CPU kernel implementations (AVX2/AVX512 intrinsics, inline assembly) |
| `include/fbgemm/` | Public C++ headers |
| `test/` | CPU C++ unit tests (Google Test) |
| `bench/` | CPU microbenchmarks |
| `fbgemm_gpu/` | PyTorch GPU operators (CUDA / ROCm) |
| `fbgemm_gpu/experimental/gen_ai/` | GenAI-specific GPU kernels (FP8, collectives) |
| `cmake/` | CMake modules and compiler setup |
| `defs.bzl` | Bazel source file lists consumed by CMake via Python |
| `.github/scripts/setup_env.bash` | CI prelude with helper functions |

## Build commands (verified)

CPU library, CMake (minimum version 3.21):

```bash
# out-of-tree debug build
mkdir build && cd build
cmake .. -DFBGEMM_BUILD_TESTS=ON -DFBGEMM_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

CMake options of interest:
- `FBGEMM_BUILD_TESTS=ON` — build `test/` unit tests
- `FBGEMM_BUILD_BENCHMARKS=ON` — build `bench/`
- `FBGEMM_BUILD_STATIC=OFF` — produce shared library instead of static

`fbgemm_gpu` requires a full PyTorch/CUDA/ROCm environment and is built by the CI scripts under `.github/scripts/`. It is not expected to build on a clean developer machine.

## Test commands (verified)

After a CPU CMake build with tests enabled:

```bash
cd build
ctest --output-on-failure -j$(nproc)
```

Individual test binaries live in `build/test/` (e.g. `test/PackedMatrixTest`).

## Lint / code quality

- C++ code must pass the project's `.clang-tidy` rules. CI runs clang-tidy via `fbgemm_gpu_lint.yml`.
- Python code in `fbgemm_gpu/` is formatted/linted by the same CI workflow (black/isort/flake8 configuration is embedded in the workflow).
- Keep formatting consistent with the surrounding files; do not reformat whole files.

## Conventions and gotchas

1. **Compiler requirements** — C++20, CMake 3.21+. CI tests GCC 11.4.0, GCC 14.1.0, and Clang 16.0.6 on Amazon Linux 2023.
2. **Intrinsics-heavy** — `src/` contains hand-written AVX2/AVX512 kernels. Preserve alignment and `constexpr` tile-size assumptions.
3. **`defs.bzl` is authoritative** — source lists are declared there and parsed by CMake via Python. When adding/removing files, update `defs.bzl` first.
4. **Tests must be added for new APIs** — follow the existing Google Test patterns in `test/`.
5. **Documentation** — user-facing docs are under `docs/` (Sphinx) and `fbgemm_gpu/docs/`. Update the relevant README if you change public APIs.
6. **CLA** — upstream PyTorch/FBGEMM requires a completed CLA for PRs. Fork-only changes here can be reviewed and promoted by a human when appropriate.
7. **No large refactors** — this is a performance kernel library. Prefer minimal, targeted patches with benchmark numbers when performance-sensitive code changes.

## Common tasks

| Task | Entry point |
|------|-------------|
| Build CPU library | `mkdir build && cd build && cmake .. && make -j$(nproc)` |
| Run CPU tests | `cd build && ctest --output-on-failure` |
| Build GPU wheels | `.github/scripts/setup_env.bash` then `build_fbgemm_gpu_package` helpers (requires CUDA/ROCm + conda) |
| Update source lists | edit `defs.bzl` then re-run CMake |

## CI

GitHub Actions workflows in `.github/workflows/`:
- `fbgemm_ci.yml` — CPU builds and tests for `main` and PRs (runs only in the `pytorch` upstream org).
- `fbgemm_gpu_ci_*.yml` — GPU builds/tests for CUDA, ROCm, and CPU-only variants.
- `fbgemm_gpu_lint.yml` — Python/C++ lint.
- `fbgemm_gpu_docs.yml` — documentation build.
