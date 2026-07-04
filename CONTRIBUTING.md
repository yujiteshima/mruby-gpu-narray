# Contributing to mruby-gpu-narray

Thanks for your interest! This is an **mrbgem** that computes on the GPU via
**Vulkan Compute**, so building it needs a Vulkan loader and a shader compiler.
This guide covers how to build, test, and contribute.

## Requirements

- **mruby 3.x** — you build this gem *inside* an mruby checkout (there is no
  standalone build).
- A **C compiler** (gcc or clang).
- **Vulkan loader + headers** (`libvulkan`, `vulkan/vulkan.h`)
  - Debian / Ubuntu / Raspberry Pi OS: `sudo apt install libvulkan-dev`
  - macOS: `brew install vulkan-loader vulkan-headers molten-vk`
- **glslangValidator** (from `glslang`) — compiles the compute shaders to
  SPIR-V at build time.
  - Debian / Ubuntu: `sudo apt install glslang-tools`
  - macOS: `brew install glslang`
- A **Vulkan compute-capable device** at runtime:
  - Raspberry Pi 5 (VideoCore VII / Mesa V3DV) — the primary target
  - macOS (Apple GPU via MoltenVK; portability is auto-detected)
  - any desktop GPU, or **lavapipe** (software Vulkan) for headless / CI

## Building & running

Add the gem to an mruby `build_config.rb`:

```ruby
MRuby::Build.new do |conf|
  conf.toolchain :gcc          # use :clang on macOS
  conf.gembox 'default'
  conf.gem '/path/to/mruby-gpu-narray'

  # macOS only (Homebrew Vulkan) — omit these two lines on Linux / Pi:
  # conf.cc.include_paths     << '/opt/homebrew/include'
  # conf.linker.library_paths << '/opt/homebrew/lib'

  conf.enable_test
end
```

Build and run:

```sh
cd /path/to/mruby
MRUBY_CONFIG=/path/to/build_config.rb rake
./build/host/bin/mruby /path/to/mruby-gpu-narray/test/narray_test.rb      # expect: ALL TESTS PASSED
./build/host/bin/mruby /path/to/mruby-gpu-narray/examples/narray_basics.rb
```

The compute shaders are compiled to SPIR-V **automatically** during the build
(via `glslangValidator`; override the binary with `GLSLANG=`). `make -C shader`
is a manual fallback. The `.spv` files and `src/shader_dir.h` are generated and
git-ignored — do not commit them.

## Tests

`test/narray_test.rb` uses known-value assertions (there is no CPU-backend
parity yet). Please add cases there for any behavior you change or add, and make
sure it prints `ALL TESTS PASSED` before opening a PR.

## Reporting issues

GPU behavior is environment-specific, so please include:

- Device / driver — paste `GPU.info` output, or `vulkaninfo --summary`
- OS and version, and the board (e.g. Raspberry Pi 5)
- mruby version
- the failing script, with expected vs. actual output

## Pull requests

- Keep the **only link dependency `libvulkan`** — please don't add new external
  libraries to `mrbgem.rake`.
- Match the surrounding style (C: the existing `src/*.c`; Ruby: `mrblib`).
- Add or update tests and keep `test/narray_test.rb` green.
- One focused change per PR. In the description, say **what you verified and on
  which device** (e.g. "tested on Raspberry Pi 5 / V3D and macOS / MoltenVK").

## Scope (current)

`v0.1.0` is the **L1 foundation**: FP32, 1-D `GPU::SFloat`, element-wise and
scalar arithmetic, and `sum` / `mean`. Work in progress: N-dimensional (2-D)
support, FFT, and a kernel DSL. Dense linear algebra (GEMM / SVD) is
intentionally **out of scope on Vulkan** (see the README roadmap for why). If
you're unsure whether a change fits, please open an issue first.

## License

By contributing, you agree that your contributions are licensed under the
**MIT License** (see [LICENSE](LICENSE)).
