"""
Neural Apoptosis Package

Core modules for implementing neuron-level apoptosis in neural networks.
"""

from .neuron_apoptosis_fixed import NeuronApoptosisManager
from .smooth_apoptosis import (
    FunctionalPreservationApoptosis,
    GradualFadeApoptosis,
    ContinuousTurnoverApoptosis,
    DistillationApoptosis
)
from .growth_only_strategy import (
    GrowthOnlyManager,
    HybridGrowthAndDeath
)
from .crossover_strategy import CrossoverApoptosis

__all__ = [
    'NeuronApoptosisManager',
    'FunctionalPreservationApoptosis',
    'GradualFadeApoptosis',
    'ContinuousTurnoverApoptosis',
    'DistillationApoptosis',
    'GrowthOnlyManager',
    'HybridGrowthAndDeath',
    'CrossoverApoptosis',
]

__version__ = '0.1.0'
