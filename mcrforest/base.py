
# THIS FILE WAS AUTOMATICALLY GENERATED BY deprecated_modules.py
import sys
# mypy error: Module X has no attribute y (typically for C extensions)
from . import _base  # type: ignore
from ._pep562 import Pep562
from sklearn.utils.deprecation import _raise_dep_warning_if_not_pytest

#deprecated_path = 'mcrforest.base'
#correct_import_path = 'mcrforest'

#_raise_dep_warning_if_not_pytest(deprecated_path, correct_import_path)

def __getattr__(name):
    return getattr(_base, name)

if not sys.version_info >= (3, 7):
    Pep562(__name__)
