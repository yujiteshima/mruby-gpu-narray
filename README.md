# mruby-gpu-narray

> **Numo-like N-dimensional array for mruby, computed on the GPU via Vulkan Compute.**
> The code you prototype with is the code you deploy to the edge — no Python→C++ rewrite.

🚧 **Work in progress** — an early but working GPU array library for mruby
(FP32, 1-D today; N-dimensional planned). Built on the Vulkan compute core of
[mruby-gpu](https://github.com/yujiteshima/mruby-gpu).

```ruby
a = GPU::SFloat.new(1024).seq   # like Numo::SFloat.new(1024).seq
b = a * 2 + 1                    # runs on the GPU (no host copy)
b.to_a                           # only now is data copied back to the host
b.sum                            # GPU reduction -> Float
```

## Motivation

Python has CuPy: change `import numpy` to `import cupy` and your array code runs on the
GPU. Ruby's small sibling, mruby, has nothing like it — and CuPy itself is CUDA-only, so
it never reaches the GPUs on boards like the Raspberry Pi.

Two things make this a real gap, not a quick port:

- **mruby is not "small CRuby."** Its C API, build system, and GC differ completely, so
  CRuby's arrays (Numo, and its CUDA cousin Cumo) don't run on mruby at all.
- **Vulkan has no cuBLAS/cuFFT-class library to lean on.** CuPy is easy because it wraps
  NVIDIA's mature libraries; on Vulkan you build the pieces yourself.

So `mruby-gpu-narray` builds the missing foundation natively: a Numo-like array whose
data lives in GPU memory from the first line, on Vulkan Compute — the *same* mruby code
runs from a workstation down to a Raspberry Pi 5. Moving a prototype to the edge no
longer means a Python→C++ rewrite.

## Status — what works today

This is the **L1 (array foundation)** layer, FP32 and 1-D:

| Area | API |
|---|---|
| Construct | `GPU::SFloat.new(n)`, `.cast(array)`, `GPU::SFloat[…]`, `.zeros(n)`, `.ones(n)` |
| Initialize | `#seq(start=0, step=1)`, `#fill(v)` |
| Element-wise (array ⊗ array) | `+  -  *  /` |
| Scalar (array ⊗ number) | `+  -  *  /`, unary `-@` |
| Reduction | `#sum`, `#mean` |
| Host transfer | `#to_a`, `#head(k)` |
| Metadata | `#size` / `#length`, `#shape`, `#ndim` |
| Device | `GPU.info`, `GPU.device_name`, `GPU.init(dir)` |

Data lives in a `VkBuffer` the whole time; the only host copies happen in `#to_a` /
`#head`. Arithmetic and reduction are Vulkan compute dispatches.

### Verified on

- **Raspberry Pi 5** — VideoCore VII, Mesa V3DV (`V3D 7.1.10.2`, Vulkan 1.3): all 28 tests pass.
- **macOS** — Apple M-series GPU via MoltenVK: all 28 tests pass (development host).

The same source runs on both; the only difference is the Homebrew include/lib paths in
`build_config.rb` on macOS.

Scalar arithmetic is supported with the **array on the left** (`a * 2`). The reverse
(`2 * a`) raises `TypeError` — full numeric coercion is future work (see Roadmap).

## Requirements

- A Vulkan 1.1+ loader and a compute-capable device.
  - **Target:** Raspberry Pi 5 / VideoCore VII (Mesa V3DV).
  - **Dev:** also runs on macOS via MoltenVK (portability is auto-detected).
- `glslangValidator` (from the Vulkan SDK / `glslang`) to compile the shaders.
- mruby 3.x.

The gem's only link dependency is the Vulkan loader (`-lvulkan`).

## Build

`mruby-gpu-narray` is an mrbgem. Add it to your `build_config.rb`:

```ruby
MRuby::Build.new do |conf|
  conf.toolchain :gcc          # use :clang on macOS
  conf.gembox 'default'
  conf.gem '/path/to/mruby-gpu-narray'
  # On macOS, point at Homebrew's Vulkan (skip on Linux/Pi, where it's standard):
  # conf.cc.include_paths     << '/opt/homebrew/include'
  # conf.linker.library_paths << '/opt/homebrew/lib'
end
```

Then build mruby:

```sh
cd /path/to/mruby && MRUBY_CONFIG=build_config.rb rake
```

The compute shaders are compiled to SPIR-V automatically during the build (via
`glslangValidator`, found on `PATH` or set `GLSLANG=`). No manual step is needed;
`make -C shader` remains available as a fallback if `glslangValidator` is absent
at build time.

`mrbgem.rake` also bakes the absolute shader directory into a generated header, so
`GPU.init` is optional — the first GPU operation initializes lazily. You can still
call `GPU.init("/some/shader/dir")` explicitly to override it.

Try it:

```sh
./build/host/bin/mruby /path/to/mruby-gpu-narray/examples/narray_basics.rb
./build/host/bin/mruby /path/to/mruby-gpu-narray/test/narray_test.rb   # ALL TESTS PASSED
```

## How it works

```
GPU::SFloat (mruby)                         mrblib/gpu_narray.rb  (shape, mean, inspect, zeros/ones/[])
   │  a * 2 + 1
   ▼
src/gpu_narray.c    dtype methods; picks a pipeline; keeps results on the GPU
   ▼
src/gpu_vulkan.c    dispatch_compute(): descriptor set → command buffer → submit → fence
   ▼
shader/*.comp       add / sub / mul / div / scale / adds / sum  (GLSL → SPIR-V)
   ▼
Vulkan driver → GPU (VideoCore VII on the Pi, or Metal via MoltenVK on macOS)
```

`sum` is a two-level reduction: each workgroup reduces 256 elements in shared memory
to one partial, and the host adds the partials in double precision.

## Roadmap

- **L3 — VkFFT** ([VkFFT](https://github.com/DTolm/VkFFT)): FFT sharing the same
  `VkBuffer`s. The highest-value domain piece (signal/vibration processing).
- **L2 — kernel DSL**: turn a Ruby block into a compute shader by tracing operator
  overloads → GLSL → SPIR-V, so `na.map { |x| x * 2 + sin(x) }` runs on the GPU.
- **L1 growth**: 2-D + minimal broadcast, more element-wise ops (`sqrt`/`exp`/`sin`),
  `min`/`max`, scalar-on-left coercion.
- **Backends**: a CUDA backend for Jetson behind the same mruby API.

Out of scope for now: dense linear algebra (GEMM/SVD/eigen) — Vulkan has no cuBLAS-class
library, so it isn't worth the effort yet.

## Relationship to mruby-gpu

[mruby-gpu](https://github.com/yujiteshima/mruby-gpu) is the predecessor: a Vulkan
compute mrbgem (plus camera / face detection / display) for the Pi. This project lifts
its proven Vulkan core (context init, host-visible buffers with a GC finalizer, generic
compute dispatch) and builds a Numo-like numeric array on top, dropping the
image/ML-specific parts.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to build,
test, and submit changes (the build needs a Vulkan loader and `glslangValidator`), and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

MIT © 2026 Yuji Teshima
