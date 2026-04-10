"""Compatibility shims for vendored JEPA dataset modules.

These modules exist so multiprocessing dataloader workers can import the same
`src.datasets.*` names that the vendored JEPA code uses internally.
"""

