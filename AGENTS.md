# DEM-Engine Agent Guide

This file is a living guide for coding agents working in DEM-Engine. Keep it practical: add rules here when a mistake would be expensive, subtle, or easy to repeat.

## Project Shape

DEM-Engine is a performance-sensitive CUDA/C++ DEM solver with a Chrono-like public API, runtime-compiled kernels, dual worker threads (`dT` and `kT`), and a demo/test suite that doubles as usage documentation. Changes should respect that mix: public APIs should stay approachable, implementation code should stay explicit, and GPU code should be treated as a place where small assumptions can become hard-to-debug runtime failures.

## General Philosophy

- Prefer local patterns over new abstractions. If an existing class, helper, `DualArray`, timer, error macro, or CMake pattern already solves the problem, use it.
- Keep changes narrow. DEM-Engine has many tightly coupled host/device data paths; avoid broad refactors unless the task explicitly calls for them.
- Make data movement obvious. When adding simulation state, follow the full path deliberately: API/cache data, dT/kT storage, `DEMData*` pointer binding, host-to-device transfer, JIT substitution if needed, and runtime refresh/update behavior.
- Preserve the setup/runtime boundary. Template/cache data is usually resolved during `Initialize()` or `Update()`, not rebuilt inside high-frequency `DoDynamics()` paths.
- Keep CPU-side convenience separate from GPU hot paths. Public APIs can be friendly; kernels and per-step loops should be boring, direct, and allocation-free.
- Prefer explicit names over cleverness. Names like `COMBINED_OWNER_REIMPOSITION_BLOCK` are good because they describe the kernel context and the tuning purpose.
- Use comments to explain non-obvious physics, synchronization, ownership, or CUDA/JIT constraints. Do not narrate simple assignments.

## Style

- Follow `.clang-format` for C++/CUDA formatting. The repo uses Chromium style with 4-space indentation and a 120-column limit.
- Match existing CMake style in the directory being edited. Do not modernize unrelated CMake while adding a target or option.
- Keep includes, namespace use, and helper placement consistent with nearby files.
- Prefer `DEME_ERROR`, `DEME_WARNING`, and existing diagnostic patterns over ad hoc exceptions or `std::cout` in library code.
- Demos and modular tests may print concise PASS/FAIL information, but core library code should avoid noisy output unless controlled by existing verbosity mechanisms.

## CUDA And JIT Kernels

- Treat CUDA launch configuration as part of correctness, not just performance.
- For added new kernel calls' block size, prefer a reasonable `constexpr` value defined near the kernel call, for example `COMBINED_OWNER_REIMPOSITION_BLOCK`. Use `DEME_MAX_THREADS_PER_BLOCK` only when a 1024-thread block is known to be fine. Oversized blocks can make larger CUDA kernels fail silently, which is especially dangerous.
- Kernel call arguments must avoid implicit type conversion. The type passed at launch must match the kernel call pattern exactly. For example, do not pass a `size_t` value to a kernel argument declared as `unsigned int`; assume CUDA jitified kernels will not perform smart type conversion for launch arguments.
- When adding a kernel argument, update all related launch sites, instantiation/prewarm paths, and any JIT call patterns together.
- Keep kernel argument lists and `DEMData*` structs type-stable. If a host value must be narrowed, validate it and cast explicitly at the boundary.
- Avoid adding host/device synchronization in hot paths unless it is required for correctness. Prefer existing events, stream usage, and worker-thread coordination patterns.
- If a kernel can run in both contact and no-contact scenarios, make sure it is not accidentally nested under `nContactPairs > 0`.
- When adding per-step GPU work, add NVTX ranges and timers only where they help separate meaningful phases.

## Simulation Semantics

- Respect owner/member distinctions. Clumps, meshes, templates, combined owners, and tracked objects have different lifetimes and ownership rules.
- Combined-owner behavior should preserve rigid relative motion among members unless a feature explicitly says otherwise.
- Contact suppression, family prescriptions, user-added acceleration, and custom force models can interact. Check the order in `calculateForces()`, integration, and re-imposition paths before changing behavior.
- Keep units, frames, and quaternion conventions explicit. Many angular quantities are local-frame internally and global-frame in user-facing helpers.

## Public API Changes

- Follow existing API naming and overload patterns. New APIs should look like the surrounding `DEMSolver`, `DEMTracker`, `DEMMesh`, and template-loading methods.
- Validate user inputs early with useful messages. Prefer failing at setup time over letting invalid state reach kernels.
- Keep Python bindings in sync when changing public C++ APIs that should be exposed to Python.
- Preserve backward compatibility unless the requested change explicitly breaks it.

## Tests And Demos

- Prefer modular tests for targeted behavior changes. They should be deterministic, self-checking, and small enough to run as normal build targets.
- Demos should remain useful examples, not only regression tests. If adding assertions to a demo-style file, keep the printed output readable.
- For CUDA-dependent tests, at least verify the target builds locally. If runtime execution is blocked by driver/toolkit availability, report the exact CUDA error.
- When fixing a bug, add a test that would have failed before the fix whenever feasible.

## Build And Verification

- Use focused builds first, for example `cmake --build build --target DEMTest_CombinedOwners -- -j1`, then broaden only if the touched surface warrants it.
- Do not assume generated runtime data has refreshed after changing kernel source or force-model text. Clean rebuilds or copying generated assets may be required for some issues.
- Be careful with files under `build/`; they may contain generated outputs or local experiment artifacts and usually should not be edited.

## Git Hygiene

- Work on a topic branch when asked. Keep unrelated user changes intact.
- Do not revert or reformat files outside the task scope.
- Before handing off, check `git status --short` and summarize the touched files and verification.
