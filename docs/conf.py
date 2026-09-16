"""Sphinx configuration for the DEM-Engine documentation."""

from pathlib import Path
import sys


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = DOCS_DIR.parent
DEME_RELEASE = (REPOSITORY_DIR / "VERSION").read_text(encoding="utf-8").strip()

project = "DEM-Engine"
author = "DEM-Engine contributors"
copyright = "2021–2026, DEM-Engine contributors"
version = DEME_RELEASE.split(".", maxsplit=1)[0]
release = DEME_RELEASE

sys.path.insert(0, str(DOCS_DIR / "_ext"))

extensions = [
    "repository_links",
    "breathe",
    "sphinx.ext.autosectionlabel",
]

breathe_projects = {
    "DEME": str(DOCS_DIR / "_build" / "doxygen" / "xml"),
}
breathe_default_project = "DEME"
breathe_default_members = ("members", "undoc-members")

autosectionlabel_prefix_document = True
# Breathe sees implementation types that are deliberately outside the public
# Doxygen input set. Keep authored directive and document warnings fatal while
# allowing those signature types to render as plain identifiers.
nitpicky = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = f"DEM-Engine {version}"
