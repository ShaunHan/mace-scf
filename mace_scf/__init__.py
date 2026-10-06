from packaging.version import Version
from .__version__ import __version__

import mace
if Version(mace.__version__) < Version("0.3.14"):
    raise ImportError("mace_scf requires mace-torch >= 0.3.14")

import graph_longrange
assert Version(graph_longrange.__version__) >= Version("0.3.0"), "please update your graph_longrange version to >= 0.3.0"
