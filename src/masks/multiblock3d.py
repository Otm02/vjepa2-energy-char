"""Shim: vendored JEPA 3-D multiblock masks (see ``src.models.vjepa2.jepa``)."""

from importlib import import_module

_v = import_module("src.models.vjepa2.jepa.src.masks.multiblock3d")
for _k, _val in _v.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _val
