Python wheel maintenance
========================

For prerequisites, a single wheel build, and runtime validation, see
`Installation <../installation.rst>`__. The commands below run from the repository root unless
a step explicitly changes directory.

Build the complete supported Python matrix
------------------------------------------

One native wheel must be built by each targeted CPython interpreter. Building
with Python 3.13, for example, creates only the ``cp313`` wheel; it does not
also create wheels for the other supported Python versions. DEM-Engine 3
currently targets CPython 3.9 through 3.14.

The following reproducible Conda workflow creates a separate build environment
for every target. Run the environment-creation commands from the repository
root:

.. code-block:: console

   conda create --yes --name deme-wheel-py39 python=3.9 pip
   conda create --yes --name deme-wheel-py310 python=3.10 pip
   conda create --yes --name deme-wheel-py311 python=3.11 pip
   conda create --yes --name deme-wheel-py312 python=3.12 pip
   conda create --yes --name deme-wheel-py313 python=3.13 pip
   conda create --yes --name deme-wheel-py314 python=3.14 pip

   for env in deme-wheel-py39 deme-wheel-py310 deme-wheel-py311 deme-wheel-py312 deme-wheel-py313 deme-wheel-py314; do
       conda run --name "$env" python -m pip install --upgrade pip build twine auditwheel
   done

Build once with each interpreter. As in the single-version workflow, run the
builder from the checkout's parent directory to prevent the local ``build/``
directory from shadowing the PyPA ``build`` package:

.. code-block:: console

   cd ..
   for env in deme-wheel-py39 deme-wheel-py310 deme-wheel-py311 deme-wheel-py312 deme-wheel-py313 deme-wheel-py314; do
       conda run --name "$env" python -m build --wheel --outdir DEM-Engine/dist DEM-Engine
   done
   cd DEM-Engine

   conda run --name deme-wheel-py313 python -m twine check dist/*.whl

The resulting directory should contain six distinct wheels with ``cp39``,
``cp310``, ``cp311``, ``cp312``, ``cp313``, and ``cp314`` tags.
Confirm that explicitly:

.. parsed-literal::

   ls -1 dist/deme-|release|-cp3\*-linux\_\*.whl
   for wheel in dist/\*.whl; do
       conda run --name deme-wheel-py313 python -m auditwheel show "$wheel"
   done

Generating all six files is only the build step. Each wheel must still be
installed and exercised with its matching Python version before that version
is considered validated. CUDA, Linux ABI, and GPU compatibility also require
separate testing; ``auditwheel show`` reports the native shared-library and
``glibc`` requirements but does not prove runtime compatibility.

Build release wheels with cibuildwheel
--------------------------------------

The Conda commands above are useful for native development builds. Release
wheels use ``cibuildwheel`` and PyPA's CUDA-enabled
``manylinux_2_28_x86_64_cuda12_9`` container so the result does not inherit the
Linux ABI of the maintainer's workstation. The configuration is stored in
``pyproject.toml``.

With Docker available, build the same complete matrix locally from the parent
of the checkout:

.. parsed-literal::

   python3 -m venv .venv-cibuildwheel
   source .venv-cibuildwheel/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "cibuildwheel==4.1.1" twine auditwheel

   cd ..
   python -m cibuildwheel --platform linux --output-dir DEM-Engine/wheelhouse DEM-Engine
   cd DEM-Engine

   python -m twine check wheelhouse/\*.whl
   for wheel in wheelhouse/\*.whl; do
       python -m auditwheel show "$wheel"
   done

This release process uses ``auditwheel repair`` to copy ordinary redistributable
native dependencies into each wheel and assign the
``manylinux_2_28_x86_64`` tag. It explicitly excludes ``libcuda.so.1``,
``libcudart.so.12``, and ``libnvrtc.so.12``. DEME runtime-compiles CUDA kernels,
so NVRTC, its builtins, and headers must be supplied by the ``cuda12`` extra
or a compatible system CUDA 12.9 toolkit. The NVIDIA driver remains a host
requirement. Bundling a driver
stub is incorrect, while bundling NVRTC without all of its dynamically loaded
resources produces an incomplete runtime. Before publishing, inspect the
repaired wheel and ``auditwheel show`` output to confirm that CUDA is the only
non-system external dependency.

Automated wheel builds
----------------------

``.github/workflows/python-wheels.yml`` runs the same policy as six parallel
jobs, one for each CPython ABI. It runs on relevant pull requests, release tags,
or manual dispatch. Every job:

* checks out Git submodules recursively;
* builds in the CUDA 12.9 manylinux 2.28 container;
* repairs the wheel with ``auditwheel``;
* checks package metadata and the expected Python/platform filename tags;
* installs the ``cuda12`` extra in a clean Python container and compiles CUDA,
  CURAND, and CCCL headers with pip-provided NVRTC, without a system toolkit; and
* uploads the wheel as a workflow artifact for testing or release assembly.

The hosted build runners do not provide a usable NVIDIA GPU. Consequently this
workflow validates compilation, repair, metadata, and tags but deliberately
does not claim GPU runtime validation. Install each artifact on a compatible
GPU host and run the tests below before publishing it.

Publish the deme distribution
-----------------------------

Publishing changes external package state and is only for authorized
maintainers. ``python-wheels.yml`` uses PyPI Trusted Publishing, so it does not
store a long-lived PyPI token in GitHub.

Before the first upload, create a GitHub environment named ``pypi`` under
``Settings`` then ``Environments``. Configure required reviewers so that the
publication job always pauses for approval. Then sign in to PyPI, open the
account-level ``Publishing`` page, and add a pending GitHub publisher with:

* PyPI project name: ``deme``;
* GitHub owner: ``Ruochun``;
* repository: ``DEM-Engine``;
* workflow filename: ``python-wheels.yml``; and
* environment: ``pypi``.

The pending publisher creates ``deme`` on the first successful upload. It
does not reserve the name before that upload. The project name must exactly
match ``name = "deme"`` in ``pyproject.toml``.

To build without publishing, open the repository's ``Actions`` tab, select
``Build Python wheels``, choose ``Run workflow``, leave
``publish_to_pypi`` disabled, and run it from the intended commit or branch.
Download and test all six artifacts after the jobs succeed.

To publish the already-reviewed source commit, dispatch the same workflow
again with ``publish_to_pypi`` enabled. The six build jobs run again; only if
all succeed does the ``Publish deme wheels to PyPI`` job enter the protected
``pypi`` environment. Approve that deployment after checking the commit and
wheel jobs. The publishing job downloads the six artifacts and uploads them
with a short-lived PyPI OIDC credential.

PyPI does not allow replacing a file or reusing an existing release version.
If any ``deme`` version |release| file has already been uploaded, increment the
project version and rebuild the complete wheel set rather than retrying with
different bytes under the same version.

Wheel portability
-----------------

Before distributing a wheel, record and test at least:

* the Python and ABI tag in the wheel filename;
* the Linux distribution and minimum compatible ``glibc`` baseline;
* the CUDA Toolkit used for compilation;
* the minimum NVIDIA driver version;
* the GPU architectures included by the CUDA build; and
* imported shared-library dependencies.

Until the supported compatibility matrix is published, build and validate
wheels on the oldest intended deployment platform and test them on each
supported Python, CUDA/driver, and GPU configuration.

Release distribution
--------------------

The wheel workflow publishes the ``deme`` distribution through the ``pypi``
environment on ``main`` and ``Mesh_Particles``. Publication remains an explicit
manual dispatch.

Python-only CUDA configuration
------------------------------

The ``cuda12`` extra supplies NVIDIA component wheels. Before importing the
extension, ``deme._cuda`` loads the required libraries by absolute path and
returns their header directories. A private binding configures these directories
in the Python core variant (``core_python``), before any solver workers start.
The native ``core`` target does not compile that configuration hook and retains
its original CUDA header search order. No CUDA environment variable is set, so
executables launched from Python retain their normal native configuration.

PyPI project description
------------------------

``pyproject.toml`` sets ``readme = "README.md"``. Each wheel embeds that README
as its Markdown description, so publishing a new DEME 3 release built from this
source also publishes the DEME 3 front page. Editing GitHub's README alone does
not update metadata in an already uploaded release; rebuild and publish a new
version through the release workflow. Older version pages retain their release
metadata.

Before building wheels, ``python-wheels.yml`` runs
``docs/prepare_pypi_readme.py`` in its disposable checkout. The script changes
relative README links into absolute GitHub URLs pinned to the build commit.
This keeps the installation guides and demo links usable on PyPI while leaving
the repository README's relative links and remote cover images intact. No
separate PyPI README needs to be maintained. The workflow checks that each
wheel's description matches the prepared README and runs ``twine check``.

For a release built outside this workflow, run the same preparation script in
a disposable source checkout, supplying its repository URL and revision, before
building. The script rewrites that checkout's README in place. Inspect the wheel
metadata and rendered description before publication; the documentation-site
workflow does not update PyPI.
