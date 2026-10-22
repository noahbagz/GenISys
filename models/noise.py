"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import torch


class NoiseStream:
    """One isolated CUDA RNG, with hashes proving equality of actual draws."""

    def __init__(self, seed, device):
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.digests = {name: hashlib.sha256() for name in ("initial", "innovation", "lock")}
        self.counts = dict.fromkeys(self.digests, 0)

    def draw(self, template, role):
        # randn_like preserves a tensor's memory layout. Some historical HCDDPM
        # templates are column-major views; a contiguous randn(shape) assigns
        # their draws to different coordinates even with the same seed.
        values = torch.empty_like(template).normal_(generator=self.generator)
        digest = self.digests[role]
        digest.update(str(tuple(values.shape)).encode())
        digest.update(values.detach().cpu().numpy().tobytes())
        self.counts[role] += 1
        return values

    def report(self):
        return {name: {"draw_count": self.counts[name], "sha256": value.hexdigest()}
                for name, value in self.digests.items()}


