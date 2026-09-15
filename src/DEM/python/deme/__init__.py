"""Python interface for DEM-Engine.

The implementation lives in a private extension module so this package can
grow Python-level helpers without changing the public import path.
"""

from ._cuda import configure_cuda as _configure_cuda

_cuda_include_paths = _configure_cuda()
from ._deme import _set_cuda_include_paths

_set_cuda_include_paths(_cuda_include_paths)
del _configure_cuda, _cuda_include_paths, _set_cuda_include_paths

from ._deme import *  # noqa: F401,F403
from ._deme import __version__
