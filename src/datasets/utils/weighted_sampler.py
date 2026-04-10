"""Compatibility shim for the vendored JEPA weighted sampler utilities."""

from src.models.vjepa2.jepa.src.datasets.utils.weighted_sampler import (  # noqa: F401
    CustomWeightedRandomSampler,
    DatasetFromSampler,
    DistributedSamplerWrapper,
    DistributedWeightedSampler,
)
