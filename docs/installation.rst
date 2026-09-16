Installation
============

System requirements
-------------------

DEME requires:

* a 64-bit Linux system for the currently supported Python package;
* an NVIDIA GPU;
* an NVIDIA driver compatible with the selected CUDA Toolkit;
* CUDA runtime libraries, NVRTC, and headers (installed by the ``cuda12`` extra,
  or provided by a compatible system CUDA Toolkit);
* CMake 3.18 or newer and a CUDA-compatible C++ compiler when building from
  source.

The default source build also includes the interactive visualizer. On Linux,
install the X11 and OpenGL development headers listed in
`Interactive visualization <visualization.rst>`__. Set ``DEME_BUILD_VISUALIZER=OFF`` only for an explicitly
headless build; published Python wheels build the visualizer by default.

The exact Python, CUDA, compiler, driver, and GPU architecture matrix is being
validated for DEM-Engine 3. A wheel should not be assumed portable across
CUDA major versions until that matrix is published.

The release-wheel policy currently covers 64-bit x86 Linux with glibc 2.28 or
newer, CPython 3.9 through 3.14, and CUDA 12.9. The installed machine must
provide an NVIDIA driver compatible with CUDA 12.9. Source builds can continue
to use other supported CUDA Toolkit versions, but those builds are outside the
binary-wheel compatibility policy.

Python package
--------------

Install a released wheel with:

.. code-block:: console

   python -m pip install "deme[cuda12]"

This installation flavor is for the Python package only. It configures the
private Python extension without changing CUDA environment variables or the
header discovery used by standalone C++ programs. C++ builds and applications
continue to use their normal system CUDA Toolkit.

The ``cuda12`` extra installs CUDA 12.9 runtime/compiler libraries and headers
from NVIDIA wheels. No system CUDA Toolkit is required for binary-wheel users.
On WSL2, install a compatible NVIDIA driver on Windows; do not install a Linux
GPU driver inside WSL. A working GPU driver is still required and is not
installed by pip. Building from source still requires a CUDA development toolkit.

Use plain ``pip install deme`` if you provide a system toolkit yourself.

The canonical import is:

.. code-block:: python

   import deme

The historical ``import DEME`` spelling remains available as a compatibility
alias. New code should use the canonical lowercase ``deme`` distribution and
import name.

Build a wheel from a checkout
-----------------------------

Build prerequisites
~~~~~~~~~~~~~~~~~~~

The wheel contains a native CUDA/C++ extension and is compiled on the machine
that creates it. Before building, verify that the intended Python interpreter,
CMake, CUDA compiler, and NVIDIA driver are available:

.. code-block:: console

   python3 --version
   cmake --version
   nvcc --version
   nvidia-smi

Initialize the bundled Git dependencies:

.. code-block:: console

   git submodule update --init --recursive

Run the environment-creation commands from the repository root. Start with an
empty ``dist/`` directory, or move artifacts from earlier builds elsewhere, so
that validation and installation cannot accidentally select an older wheel.

Create and validate the wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a dedicated virtual environment for packaging:

.. code-block:: console

   python3 -m venv .venv-wheel
   source .venv-wheel/bin/activate

   python -m pip install --upgrade pip
   python -m pip install build twine

   cd ..
   python -m build --wheel --outdir DEM-Engine/dist DEM-Engine
   cd DEM-Engine
   python -m twine check dist/*

The build command deliberately runs from the checkout's parent directory. A
pre-existing local ``build/`` directory in the repository root can otherwise
shadow the PyPA package named ``build`` and cause
``No module named build.__main__``. Replace ``DEM-Engine`` with the checkout
directory name if it differs.

``python -m build --wheel`` invokes the ``scikit-build-core`` backend from
``pyproject.toml``. That backend configures CMake with
``DEME_BUILD_PYTHON=ON``, compiles the native ``deme._deme`` extension in
Release mode, and places the resulting wheel under ``dist/``.

The repository-root ``VERSION`` file is the authoritative DEM-Engine release
version. Update only that file when preparing a release; CMake, Python package
metadata, the Conda recipe, runtime ``__version__``, and these rendered
documentation examples derive their versions from it.

``twine check`` validates the wheel metadata and the rendering of its package
description. A successful build should produce a platform-specific file whose
name resembles:

.. parsed-literal::

   dist/deme-|release|-<python-tag>-<abi-tag>-linux_<architecture>.whl

This is not a pure-Python or universal wheel. Its filename tags determine which
Python interpreter and operating-system ABI pip will accept, while CUDA and
driver compatibility must also be validated separately.

Test the wheel in a clean environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leave the packaging environment, create a separate test environment, and
install the wheel there:

.. parsed-literal::

   deactivate

   python3 -m venv /tmp/deme-wheel-test
   source /tmp/deme-wheel-test/bin/activate

   python -m pip install --upgrade pip
   python -m pip install dist/deme-|release|-\*.whl
   python -m pip check

Run import checks from outside the source tree. Otherwise, files in the
checkout could hide missing wheel contents:

.. code-block:: console

   cd /tmp
   python -c "import deme; print(deme.__version__, deme.__file__)"
   python -c "import DEME; print(DEME.__version__)"

The first command should report version |release| and a module path inside
``/tmp/deme-wheel-test``. The second verifies the compatibility import; new
applications should continue to use lowercase ``import deme``.

On a host with a supported visible NVIDIA GPU, also construct a solver to
exercise CUDA initialization and confirm the selected logical device:

.. code-block:: console

   python -c "import deme; s = deme.DEMSolver([0]); print(s.GetGPUDeviceIDs())"

Expected output is ``[0, 0]``. Import checks alone do not exercise solver
construction, CUDA device selection, or worker allocation.

Return to the checkout when testing is complete:

.. code-block:: console

   deactivate
   cd /path/to/DEM-Engine

C++ build
---------

.. code-block:: console

   git submodule update --init --recursive
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel

Use a focused demo or modular-test target first when validating a change.

On native Windows, configure with CMake GUI or the command line using a
CUDA-compatible Visual Studio toolchain, then build the Release configuration:

.. code-block:: console

   cmake --build build --config Release

Executables from multi-configuration generators are normally under
``build/bin/Release``. Linux and WSL use ``build/bin``. WSL follows the Linux
instructions; graphical output additionally needs the display setup in
`Interactive visualization <visualization.rst>`__.

Install the C++ library
-----------------------

Select an installation prefix during configuration, then install after building:

.. code-block:: console

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/deme-install
   cmake --build build --config Release --parallel
   cmake --install build --config Release

For a consuming CMake project, point ``DEME_DIR`` to the installed directory
containing ``DEMEConfig.cmake`` (under ``lib/cmake/DEME`` or
``lib64/cmake/DEME``, depending on the installation).

Development and release packaging
---------------------------------

For a local source installation use ``python -m pip install .``; use
``python -m pip install -e .`` for an editable installation. These still build
a native extension and need the source-build prerequisites. Select a specific
interpreter for a manual CMake build with
``-DPython_EXECUTABLE=/path/to/python`` and ``-DDEME_BUILD_PYTHON=ON``.

The Conda recipe is under ``recipe/``. To build it locally, install
``conda-build`` and run ``conda build recipe/ -c conda-forge``. Use compilers
and runtime libraries compatible with the target environment; see
`Troubleshooting <troubleshooting.rst>`__ for ``GLIBCXX`` errors.

For the supported wheel matrix, CI, portability checks, and PyPI publishing,
see `Python wheel maintenance <developer/packaging.rst>`__.
