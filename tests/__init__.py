import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
