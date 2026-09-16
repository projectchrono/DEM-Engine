Hosting the documentation
=========================

Recommended: GitHub Pages
-------------------------

GitHub Pages fits this repository's existing static documentation build.
One site contains the authored guides, the Doxygen/Breathe C++ reference, and
the committed Python API reference. A normal HTML build does not import the
native extension, compile the solver, or need a GPU. See
`Building the documentation <documentation.rst>`__ for the build and reference-regeneration steps.

Pages is free for public repositories on GitHub Free. Published sites have a
1 GB size limit and a soft bandwidth limit of 100 GB/month. These should suit
API documentation; keep large simulation videos and datasets elsewhere.
Check `GitHub's current Pages limits
<https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits>`_
before choosing it for a larger site.

Publish with the included workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.github/workflows/docs.yml`` builds both API references and publishes the
combined site. The repository variable ``DOCS_PUBLISH_BRANCH`` selects the
publishing branch. Pushes to other branches skip the build and deployment;
relevant pull requests build an HTML artifact without deploying. An unset
variable disables publication.

One-time GitHub setup
^^^^^^^^^^^^^^^^^^^^^

#. Open **Settings → Secrets and variables → Actions → Variables** and create
   a repository variable named ``DOCS_PUBLISH_BRANCH`` with value
   ``DEME3_docs``. Use a repository variable, not an environment variable or
   secret, because the build job checks it before entering an environment.
#. In **Settings → Pages → Build and deployment → Source**, select
   **GitHub Actions**.
#. In **Settings → Environments**, create or open ``github-pages``. Under
   deployment branches and tags, allow the branch ``DEME3_docs``. If required
   reviewers are configured, each deployment waits for their approval; omit
   required reviewers if publication should be fully automatic.
#. Push the commit containing the workflow and documentation to ``DEME3_docs``.
   Every subsequent push to that branch builds and publishes the site.
#. Open **Actions → Build and publish documentation** and inspect the build
   and deploy jobs. The deployment links to the live site, normally
   ``https://ruochun.github.io/DEM-Engine/`` for this repository.

If the branch was already pushed before setup, use **Re-run all jobs** on its
workflow run after configuring Pages and the variable, or push a new commit.
Keep generated HTML out of Git; the workflow uploads ``docs/_build/html`` as
an artifact and publishes it only after a successful build.

The workflow also supports **Run workflow**. GitHub requires the workflow file
to exist on the repository's default branch before manual dispatch is available.
Until then, use the push trigger on ``DEME3_docs``; changing the repository's
default branch is unnecessary. Once manual dispatch is available, select the
branch matching ``DOCS_PUBLISH_BRANCH`` in the branch dropdown.

Switch publication to main later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Merge the documentation and workflow into ``main``.
#. Allow ``main`` in the ``github-pages`` environment's deployment branch rules.
#. Change the repository variable ``DOCS_PUBLISH_BRANCH`` to ``main``.
#. Push a new commit to ``main``, or manually run the workflow on ``main``.
   Changing the variable alone does not trigger a build.
#. Remove ``DEME3_docs`` from the environment's allowed deployment branches
   when it is no longer used for publication.

No workflow edit is needed. The selected branch must contain the workflow and
its documentation sources. Builds check out the triggering commit, so a push
publishes documentation corresponding to that commit rather than a moving
branch head. The Python API reference must still be regenerated and committed
when bindings change; see `Building the documentation <documentation.rst>`__.

See GitHub's official instructions for
`Pages publishing sources
<https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>`_,
`custom Pages workflows
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>`_, and
`manual workflow runs
<https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow>`_.

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
`the historical notes <../archive/implementation-notes.md>`__.
They are not current API or installation instructions; update the maintained
guides and references when behavior changes instead of adding another report.
