SPHINXBUILD ?= sphinx-build
DOXYGEN ?= doxygen
PYTHON ?= python3
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help examples python-reference doxygen html clean

help:
	@echo "make html     Build Doxygen XML and the Sphinx HTML site"
	@echo "make examples Check Python documentation example syntax"
	@echo "make python-reference  Regenerate API RST from an installed deme module"
	@echo "make doxygen  Build only the Doxygen XML"
	@echo "make clean    Remove generated documentation"

examples:
	$(PYTHON) -m compileall -q python/examples

python-reference:
	$(PYTHON) generate_python_reference.py

doxygen:
	@mkdir -p $(BUILDDIR)/doxygen
	$(DOXYGEN) Doxyfile

html: examples doxygen
	$(SPHINXBUILD) -W --keep-going -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Documentation: $(CURDIR)/$(BUILDDIR)/html/index.html"

clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)
