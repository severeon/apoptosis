#!/usr/bin/env python3
"""
Normalization utilities for tensor processing.
"""

import torch


def z_score_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Z-score normalization (standardization) of a tensor.

    Transforms the tensor to have mean=0 and std=1.

    Args:
        x: Input tensor
        eps: Small constant to prevent division by zero (default: 1e-8)

    Returns:
        Normalized tensor: (x - mean) / (std + eps)

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> z_score_normalize(x)
        tensor([-1.4142, -0.7071,  0.0000,  0.7071,  1.4142])
    """
    return (x - x.mean()) / (x.std() + eps)


def min_max_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalization to [0, 1] range.

    Args:
        x: Input tensor
        eps: Small constant to prevent division by zero (default: 1e-8)

    Returns:
        Normalized tensor: (x - min) / (max - min + eps)

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> min_max_normalize(x)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def robust_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Robust normalization using median and IQR.

    More resistant to outliers than z-score normalization.

    Args:
        x: Input tensor
        eps: Small constant to prevent division by zero (default: 1e-8)

    Returns:
        Normalized tensor: (x - median) / (IQR + eps)
    """
    median = x.median()
    q75 = x.quantile(0.75)
    q25 = x.quantile(0.25)
    iqr = q75 - q25
    return (x - median) / (iqr + eps)


def log_normalize(x: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    """
    Log normalization for skewed distributions.

    Uses log1p (log(1 + x)) to handle zero values gracefully.

    Args:
        x: Input tensor (should be non-negative)
        offset: Offset to add before taking log (default: 1.0, uses log1p)

    Returns:
        Log-normalized tensor: log(x + offset)

    Example:
        >>> x = torch.tensor([0.0, 1.0, 10.0, 100.0])
        >>> log_normalize(x)
        tensor([0.0000, 0.6931, 2.3979, 4.6151])
    """
    if offset == 1.0:
        return torch.log1p(x)
    return torch.log(x + offset)
