"""Shim: mask helper functions from vendored JEPA."""

from importlib import import_module

_v = import_module("src.models.vjepa2.jepa.src.masks.utils")
for _k, _val in _v.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _val
