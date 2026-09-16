Building the documentation
==========================

The concise build, Python-reference regeneration, and local/remote preview
commands are collected in `docs/README.md <../README.md>`__.

Prerequisites
-------------

Install Doxygen through the operating system or Conda, then create an isolated
Python environment:

.. code-block:: console

   python3 -m venv .venv-docs
   . .venv-docs/bin/activate
   python -m pip install -r docs/requirements.txt

Standalone build
----------------

.. code-block:: console

   make -C docs html

Open ``docs/_build/html/index.html`` in a browser.

Regenerating the Python API reference
-------------------------------------

After building and installing the current ``deme`` extension into the active
Python environment, regenerate the static reference from pybind11 signatures
and docstrings:

.. code-block:: console

   make -C docs python-reference
   make -C docs html

The generated ``docs/python/reference.rst`` is committed so normal
documentation builds do not import the native extension or require a GPU.
Improve descriptions in ``src/DEM/python/bindings.cpp`` and regenerate rather
than editing the reference page directly.

CMake target
------------

When configuring the complete project:

.. code-block:: console

   cmake -S . -B build-docs -DDEME_BUILD_DOCS=ON
   cmake --build build-docs --target docs

Documentation policy
--------------------

* Doxygen parses the public C++ headers and emits XML.
* Sphinx and Breathe render the C++ API alongside authored guides.
* Worker internals and CUDA kernels belong in architecture pages rather than
  the public API reference.
* New public functions should describe intent, parameters, units, frames,
  setup/runtime restrictions, and return values.
* The HTML build uses warnings-as-errors so broken API references do not
  silently enter published documentation.

See `Hosting the documentation <hosting.rst>`__ for publication options and source-tree organization, and
`Python wheel maintenance <packaging.rst>`__ for Python wheel maintenance.

Links in authored guides
------------------------

Use standard reStructuredText relative hyperlinks to source files, with a
readable label and the actual extension (for example, ``.rst``). These work
when readers browse the repository on GitHub. Prefer anonymous hyperlinks
(two trailing underscores) so repeated labels do not create duplicate targets.
Sphinx-specific document and download roles are not understood by GitHub.

The local ``repository_links`` extension converts links to ``.rst`` pages into
Sphinx document references and links to ``.md``, ``.cpp``, and ``.py`` files into
downloads during the site build. Keep these relative file links free of queries
and fragments; the adapter leaves external URLs and same-page anchors alone.
The generated API pages and toctrees still require the Sphinx build for their
full rendering.
