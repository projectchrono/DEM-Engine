Troubleshooting
===============

For numeric contact, geometry, and owner codes seen in logs or a debugger,
see `Internal type codes for debugging <developer/type-codes.rst>`__.

CUDA or PTX mismatch
--------------------

Errors such as ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` usually indicate that the
installed NVIDIA driver cannot execute code produced by the selected CUDA
Toolkit. Verify the driver/toolkit compatibility and rebuild with a supported
toolkit.

Runtime JIT compilation cannot find headers
-------------------------------------------

For binary wheels, install ``"deme[cuda12]"`` to supply matching libraries and
headers. A complete pip CUDA installation takes precedence over system CUDA
discovery inside the Python extension only. It does not export CUDA paths to
child processes. If another GPU package has already loaded a different NVRTC
version, use compatible dependencies in a fresh Python process or separate
environment.

DEME compiles kernels at runtime. Confirm that the CUDA Toolkit headers and the
``share/DEME/kernel`` and ``include`` resources installed with DEME are
available. The CUDA header major and minor version must match the loaded NVRTC
library. If several CUDA Toolkits are installed, set ``CUDA_HOME`` to the one
providing that NVRTC version, for example:

.. code-block:: console

   export CUDA_HOME=/usr/local/cuda-12.8

Errors in CUDA or CURAND headers involving undefined internal identifiers often
mean that an unversioned ``/usr/local/cuda`` link selected headers from a newer
Toolkit than the loaded ``libnvrtc``. Current DEME releases reject that mismatch
and report both the required NVRTC version and how to select matching headers.

Import works but initialization fails
-------------------------------------

``import deme`` verifies only that the extension and its immediate shared
libraries can load. A meaningful installation test must construct a solver and
run a small simulation through ``Initialize()`` so NVRTC, kernel resources, and
the GPU driver are exercised.

Conda ``GLIBCXX`` errors
------------------------

Build with compilers compatible with the target Conda environment. A wheel
built against a newer system ``libstdc++`` may import on the build host but fail
inside another environment.

Stale runtime kernels
---------------------

After changing kernel or force-model sources, use a clean build or ensure the
runtime kernel assets have been refreshed. Old copied text sources can make the
runtime behavior disagree with the compiled host code.

Persistent Jitify startup cache
-------------------------------

DEME uses Jitify and NVRTC to discover CUDA headers and compile runtime kernels during solver initialization. Workloads
that repeatedly start fresh DEME processes can persist the discovered header sources and avoid repeating much of that
startup work. Set ``DEME_PERSISTENT_JITIFY_CACHE`` to a writable cache-file path:

.. code-block:: console

   export DEME_PERSISTENT_JITIFY_CACHE="$HOME/.cache/deme/jitify_header_cache.bin"
   mkdir -p "$(dirname "$DEME_PERSISTENT_JITIFY_CACHE")"


Values such as ``1``, ``true``, ``on``, and ``yes`` select an automatic temporary path. Values such as ``0``, ``false``, ``off``, and
``no``, as well as an unset variable, disable the persistent cache. The cache is opt-in because an automatic shared
temporary path can be unsafe on multi-user systems, concurrent writers can contend, and header contents can change
without their paths or CUDA version changing. Prefer an explicit, user-owned path and remove the file after changing
CUDA toolchains or headers. An unreadable, incompatible, or unwritable cache falls back to normal Jitify discovery.


Custom force models
-------------------

If a custom force kernel fails to launch, try a smaller block size such as
``SetForceCalcThreadsPerBlock(128)`` during setup. Large kernels can exceed
register limits. Check CUDA errors before treating missing contacts or a
subsequent too-many-geometries-in-bin error as a geometry problem.

For JIT errors involving material arrays or missing wildcard variables,
compare your model declarations with ``DEMdemo_Electrostatic``. Declare pairwise
material properties with ``SetMustPairwiseMatProp`` and contact wildcards with
``SetPerContactWildcards`` before initialization.
