#!/usr/bin/env python3
"""
Histogram utilities for quantizing and visualizing tensor distributions.
"""

import math
import torch


def quantize_histogram(t: torch.Tensor, bins: int = 16) -> list:
    """
    Convert a tensor into a quantized histogram representation.

    Args:
        t: Input tensor to histogram
        bins: Number of histogram bins (default: 16)

    Returns:
        List of bin counts (length = bins)

    Notes:
        - NaN values are replaced with 0.0 before histogramming
        - Non-finite values result in uniform zero histogram
        - Empty ranges (min == max) return zero histogram
    """
    if torch.isnan(t).any():
        # Clean before histogramming
        t = torch.nan_to_num(t, nan=0.0)

    tmin = float(t.min())
    tmax = float(t.max())

    if not math.isfinite(tmin) or not math.isfinite(tmax) or tmin == tmax:
        # Fallback to uniform histogram
        return [0] * bins

    hist = torch.histc(t, bins=bins, min=tmin, max=tmax)
    return hist.cpu().tolist()
