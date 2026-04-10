"""Shim: vendored JEPA tube masks (see ``src.models.vjepa2.jepa``)."""

from importlib import import_module

_v = import_module("src.models.vjepa2.jepa.src.masks.random_tube")
for _k, _val in _v.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _val
