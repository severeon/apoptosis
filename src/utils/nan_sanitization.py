#!/usr/bin/env python3
"""
NaN and infinite value sanitization utilities for tensors.
"""

import torch


def sanitize_tensor(
    t: torch.Tensor,
    nan: float = 0.0,
    posinf: float = 1e6,
    neginf: float = -1e6
) -> torch.Tensor:
    """
    Replace NaN and infinite values in a tensor with safe defaults.

    Args:
        t: Input tensor
        nan: Replacement value for NaN (default: 0.0)
        posinf: Replacement value for +inf (default: 1e6)
        neginf: Replacement value for -inf (default: -1e6)

    Returns:
        Sanitized tensor (same shape and device as input)

    Example:
        >>> x = torch.tensor([1.0, float('nan'), float('inf'), -float('inf')])
        >>> sanitize_tensor(x)
        tensor([1.0, 0.0, 1e6, -1e6])
    """
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)


def has_nan(t: torch.Tensor) -> bool:
    """
    Check if tensor contains any NaN values.

    Args:
        t: Input tensor

    Returns:
        True if tensor contains at least one NaN
    """
    return torch.isnan(t).any().item()


def has_inf(t: torch.Tensor) -> bool:
    """
    Check if tensor contains any infinite values.

    Args:
        t: Input tensor

    Returns:
        True if tensor contains at least one +inf or -inf
    """
    return torch.isinf(t).any().item()


def has_nonfinite(t: torch.Tensor) -> bool:
    """
    Check if tensor contains any non-finite values (NaN or inf).

    Args:
        t: Input tensor

    Returns:
        True if tensor contains NaN or infinite values
    """
    return not torch.isfinite(t).all().item()


def sanitize_if_needed(t: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Sanitize tensor only if it contains non-finite values.

    This can be more efficient than always calling sanitize_tensor.

    Args:
        t: Input tensor
        **kwargs: Arguments to pass to sanitize_tensor

    Returns:
        Sanitized tensor if non-finite values found, otherwise original tensor
    """
    if has_nonfinite(t):
        return sanitize_tensor(t, **kwargs)
    return t
