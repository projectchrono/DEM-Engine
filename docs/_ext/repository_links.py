"""Resolve GitHub-readable relative source links in the compiled documentation."""

from pathlib import PurePosixPath
from urllib.parse import urlsplit

from docutils import nodes
from sphinx import addnodes


def resolve_repository_links(app, doctree):
    """Promote local source links before Sphinx collects downloads and resolves pages.

    Authored RST uses ordinary links so GitHub can render them. Sphinx needs
    document references for .rst pages and download references for source files;
    otherwise the HTML would point to files absent from the published site.
    """
    for node in list(doctree.findall(nodes.reference)):
        uri = node.get("refuri", "")
        target = urlsplit(uri)
        if target.scheme or target.netloc or not target.path or target.query or target.fragment:
            continue
        suffix = PurePosixPath(target.path).suffix
        if suffix == ".rst":
            replacement = addnodes.pending_xref(
                node.rawsource,
                refdomain="std",
                reftype="doc",
                reftarget=target.path[:-4],
                refdoc=app.env.docname,
                refexplicit=True,
                refwarn=True,
            )
        elif suffix in {".md", ".cpp", ".py"}:
            replacement = addnodes.download_reference(
                node.rawsource, reftarget=target.path, refdoc=app.env.docname
            )
        else:
            continue
        replacement.source = node.source
        replacement.line = node.line
        replacement.extend(node.children)
        node.replace_self(replacement)


def setup(app):
    """Run before the download collector so Sphinx copies linked source files."""
    app.connect("doctree-read", resolve_repository_links, priority=100)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
