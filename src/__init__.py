"""
Apoptosis-v2: Neural Apoptosis for Transformer Models

A framework for implementing neuron-level apoptosis and neurogenesis in transformer
architectures, enabling dynamic network evolution during training.

Core Components:
- ApoptoticTransformer: Main transformer model with apoptosis support
- NeuronApoptosisManager: Manages neuron lifecycle and fitness-based pruning/regrowth
- CharTokenizer: Character-level tokenization
- CharDataset: Character-level dataset for language modeling
"""

# Main model
from src.apoptotic_transformer import ApoptoticTransformer

# Apoptosis system
from src.neuron_apoptosis_manager import NeuronApoptosisManager
from src.neuron_metadata import NeuronMetadata

# Data utilities
from src.char_tokenizer import CharTokenizer
from src.char_dataset import CharDataset

# Building blocks (for advanced users)
from src.transformer_block import TransformerBlock

__all__ = [
    # Main model
    'ApoptoticTransformer',

    # Apoptosis system
    'NeuronApoptosisManager',
    'NeuronMetadata',

    # Data utilities
    'CharTokenizer',
    'CharDataset',

    # Building blocks
    'TransformerBlock',
]

__version__ = '2.0.0'
