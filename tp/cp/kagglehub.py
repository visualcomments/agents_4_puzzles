"""Minimal stub for vendored CayleyPy in offline environments.

CayleyPy imports kagglehub only for optional model weight downloads.
The Megaminx hybrid pipeline in this repository does not use those pretrained models,
so a tiny stub is enough to let the core graph/search modules import successfully.
"""

from __future__ import annotations


def model_download(*args, **kwargs):
    raise RuntimeError(
        'kagglehub is not available in this offline bundle. '
        'Pretrained CayleyPy model downloads are disabled, but graph/search features still work.'
    )
