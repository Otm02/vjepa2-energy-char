"""Compatibility shim for the vendored JEPA VideoDataset module."""

from src.models.vjepa2.jepa.src.datasets.video_dataset import (  # noqa: F401
    VideoDataset,
    make_videodataset,
)

