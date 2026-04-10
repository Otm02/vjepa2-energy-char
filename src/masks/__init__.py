"""Course-visible re-exports of vendored JEPA mask utilities.

DataLoader workers (spawn/forkserver) start a fresh interpreter and cannot rely on
``sys.modules`` injections from ``vjepa2_init``. Importing ``src.masks.*`` must
work from the normal project layout.
"""
