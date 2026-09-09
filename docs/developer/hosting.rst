Hosting the documentation
=========================

Recommended: GitHub Pages
-------------------------

GitHub Pages fits this repository's existing static documentation build.
One site contains the authored guides, the Doxygen/Breathe C++ reference, and
the committed Python API reference. A normal HTML build does not import the
native extension, compile the solver, or need a GPU. See
:doc:`documentation` for the build and reference-regeneration steps.

Pages is free for public repositories on GitHub Free. Published sites have a
1 GB size limit and a soft bandwidth limit of 100 GB/month. These should suit
API documentation; keep large simulation videos and datasets elsewhere.
Check `GitHub's current Pages limits
<https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits>`_
before choosing it for a larger site.

Suggested setup
~~~~~~~~~~~~~~~

#. In the repository's **Settings → Pages**, select **GitHub Actions** as the
   publishing source.
#. Create a workflow that checks out the intended documentation branch,
   installs Doxygen and ``docs/requirements.txt``, and runs ``make -C docs html``.
#. Upload ``docs/_build/html`` with ``actions/upload-pages-artifact`` and deploy
   it using ``actions/deploy-pages``. The deployment job needs ``pages: write``
   and ``id-token: write`` permissions and the ``github-pages`` environment.
#. Initially deploy by manual dispatch from the reviewed documentation branch.
   Once the documentation is merged, automatic publishing can track the chosen
   release branch. Pull requests should build for validation without deploying.
#. Add the resulting site URL to the root README once the site is live.

Follow `GitHub's custom workflow instructions
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>`_
for the current action versions and deployment configuration. Upload the
**generated HTML directory**, not the Sphinx source directory ``docs/``.

A project site normally has the form ``https://<owner>.github.io/DEM-Engine/``.
For a repository owned by ``Ruochun``, that would be
``https://ruochun.github.io/DEM-Engine/`` after Pages is configured and deployed.
This is an example address, not a claim that the site is already published.

Already-compiled documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To publish an existing local build, copy the contents of ``docs/_build/html/``
to a dedicated publishing branch or a separate documentation repository. Add
an empty ``.nojekyll`` file alongside ``index.html`` and configure Pages to
serve that branch's root. Keep generated files out of the development branch.
A workflow build is easier to reproduce and keep synchronized with releases.

The existing Doxyfile emits XML for Breathe. If you also want Doxygen's own
standalone HTML layout, enable ``GENERATE_HTML`` in a separate Doxygen
configuration and place its output in a subdirectory of the published site.
The current combined Sphinx site already includes the C++ reference, so this
extra output is optional.

Alternative: Read the Docs Community
------------------------------------

`Read the Docs Community <https://docs.readthedocs.com/platform/latest/index.html>`_
hosts open-source documentation at no cost. It is a good alternative when
built-in documentation versions and pull-request previews are priorities.
`Community hosting <https://about.readthedocs.com/pricing/>`_ is advertising-supported.

Use ``docs/conf.py`` and ``docs/requirements.txt`` in a ``.readthedocs.yaml``
configuration. Install Doxygen in the build environment and run
``make -C docs doxygen`` before Sphinx so Breathe has its XML input. Keep the
committed Python reference for hosted builds; regenerate it on a suitable
native-extension build host when bindings change. See the official
`Sphinx setup guide <https://docs.readthedocs.com/platform/stable/intro/sphinx.html>`_.

Maintaining the source tree
---------------------------

Keep the repository README short and link to maintained pages under ``docs/``.
Use reStructuredText for pages that should appear in the compiled site, and
add each page to a toctree. Markdown is reserved for repository-facing indexes,
credits, and historical notes; it is not parsed by the current Sphinx setup.
The root ``LICENSE.md`` and ``AGENTS.md`` remain in place for license discovery
and agent tooling.

Earlier implementation reports are preserved in
:download:`the historical notes <../archive/implementation-notes.md>`.
They are not current API or installation instructions; update the maintained
guides and references when behavior changes instead of adding another report.
