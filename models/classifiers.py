"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import torch


from models import diffusion as TDT


def load_classifier(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    classifier = TDT.ConstraintClassifier(saved["signature"]["config"], device=torch.device(device)).to(device)
    classifier.load_state_dict(saved["model"])
    classifier.guidance_output_mask = torch.as_tensor(saved["guidance_output_mask"], device=device)
    classifier.guidance_temperature = torch.as_tensor(saved["guidance_temperature"], device=device)
    classifier.eval()
    return classifier


