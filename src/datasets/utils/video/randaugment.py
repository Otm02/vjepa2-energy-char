"""Compatibility shim for JEPA RandAugment.

``from ... import *`` does not re-export names starting with ``_``. DataLoader
workers unpickle callables such as ``_rotate_level_to_arg`` from this module
path, so we forward every non-dunder attribute from the vendored implementation.
"""

from importlib import import_module

_v = import_module("src.models.vjepa2.jepa.src.datasets.utils.video.randaugment")
for _k, _val in _v.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _val
