#!/usr/bin/env python3
"""
Centralized Configuration System for Apoptosis Experiments

This module provides a single source of truth for all hyperparameters
across training, apoptosis, and experiment management.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import argparse
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    seq_len: int = 128


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_steps: int = 2000
    batch_size: int = 256
    lr: float = 2e-4
    seed: int = 42
    device: str = 'mps'  # 'cpu', 'cuda', or 'mps'

    # Validation and logging intervals
    val_interval: int = 500
    log_interval: int = 250
    checkpoint_interval: int = 2500


@dataclass
class ApoptosisConfig:
    """Neural apoptosis hyperparameters."""
    strategy: str = 'functional'  # 'baseline', 'functional', 'hybrid', 'standard'
    prune_rate: float = 0.15
    apoptosis_interval: int = 125
    mutation_strength: float = 0.3

    # Fitness function parameters (3-term model)
    fitness_alpha: float = 1.0  # Plasticity weight (grad_norm)
    fitness_beta: float = 1.0   # Usefulness weight (activation_variance)
    fitness_gamma: float = 2.0  # Stagnation penalty weight (cosine similarity)
    activation_ema_decay: float = 0.9


@dataclass
class SenescenceConfig:
    """Senescence lifecycle daemon parameters."""
    enable_senescence: bool = False
    senescence_low_pct: float = 0.1
    senescence_patience: int = 20
    senescence_age_factor: float = 0.01
    phase_durations: str = '10,5,1,5'  # Comma-separated durations for phases 1-4
    lifecycle_warmup_steps: int = 200
    max_escalations_per_step: int = 5
    max_kills_per_layer: int = 3
    slope_window: int = 5
    slope_threshold: float = 2e-4
    min_to_escalate: int = 2


@dataclass
class LoggingConfig:
    """Logging and output parameters."""
    log_dir: str = "logs"
    run_name: Optional[str] = None
    verbose: bool = False
    output_json: bool = False
    output_buffer_size: int = 10

    # Unused/deprecated
    svd_rank: int = 0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    apoptosis: ApoptosisConfig = field(default_factory=ApoptosisConfig)
    senescence: SenescenceConfig = field(default_factory=SenescenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (flattened for backwards compatibility)."""
        result = {}
        for section in [self.model, self.training, self.apoptosis, self.senescence, self.logging]:
            result.update(asdict(section))
        return result

    def to_nested_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'apoptosis': asdict(self.apoptosis),
            'senescence': asdict(self.senescence),
            'logging': asdict(self.logging)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from flat or nested dictionary."""
        # Try nested format first
        if 'model' in d and isinstance(d['model'], dict):
            return cls(
                model=ModelConfig(**d.get('model', {})),
                training=TrainingConfig(**d.get('training', {})),
                apoptosis=ApoptosisConfig(**d.get('apoptosis', {})),
                senescence=SenescenceConfig(**d.get('senescence', {})),
                logging=LoggingConfig(**d.get('logging', {}))
            )

        # Otherwise assume flat format (backwards compatibility)
        model_keys = set(ModelConfig.__dataclass_fields__.keys())
        training_keys = set(TrainingConfig.__dataclass_fields__.keys())
        apoptosis_keys = set(ApoptosisConfig.__dataclass_fields__.keys())
        senescence_keys = set(SenescenceConfig.__dataclass_fields__.keys())
        logging_keys = set(LoggingConfig.__dataclass_fields__.keys())

        return cls(
            model=ModelConfig(**{k: v for k, v in d.items() if k in model_keys}),
            training=TrainingConfig(**{k: v for k, v in d.items() if k in training_keys}),
            apoptosis=ApoptosisConfig(**{k: v for k, v in d.items() if k in apoptosis_keys}),
            senescence=SenescenceConfig(**{k: v for k, v in d.items() if k in senescence_keys}),
            logging=LoggingConfig(**{k: v for k, v in d.items() if k in logging_keys})
        )

    @classmethod
    def from_json_file(cls, path: Path) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: Path, nested: bool = True):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            data = self.to_nested_dict() if nested else self.to_dict()
            json.dump(data, f, indent=2)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExperimentConfig':
        """Create config from argparse Namespace (backwards compatibility)."""
        args_dict = vars(args)
        return cls.from_dict(args_dict)

    def to_args_namespace(self) -> argparse.Namespace:
        """Convert config to argparse Namespace (backwards compatibility)."""
        return argparse.Namespace(**self.to_dict())


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all configuration options."""
    parser = argparse.ArgumentParser(
        description='Train transformer with neural apoptosis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Logging (placed first for convenience)
    parser.add_argument('--run_name', type=str, default=f"run_{str(time.time())}",
                        help='Name of run')
    parser.add_argument('--svd_rank', type=int, default=0, help='<UNUSED>')

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--d_model', type=int, default=128, help='Model dimension')
    model_group.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    model_group.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    model_group.add_argument('--seq_len', type=int, default=128, help='Sequence length')

    # Training
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--num_steps', type=int, default=2000, help='Number of training steps')
    training_group.add_argument('--batch_size', type=int, default=256, help='Batch size')
    training_group.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    training_group.add_argument('--seed', type=int, default=42, help='Random seed')
    training_group.add_argument('--device', type=str, default='mps',
                               choices=['cpu', 'cuda', 'mps'], help='Device')
    training_group.add_argument('--val_interval', type=int, default=500,
                               help='Steps between validation checks')

    # Apoptosis
    apoptosis_group = parser.add_argument_group('Neural Apoptosis')
    apoptosis_group.add_argument('--strategy', type=str, default='functional',
                                choices=['baseline', 'functional', 'hybrid', 'standard'],
                                help='Apoptosis strategy')
    apoptosis_group.add_argument('--prune_rate', type=float, default=0.15,
                                help='Prune/turnover rate')
    apoptosis_group.add_argument('--apoptosis_interval', type=int, default=125,
                                help='Steps between apoptosis')
    apoptosis_group.add_argument('--mutation_strength', type=float, default=0.3,
                                help='Mutation strength')

    # Fitness function
    fitness_group = parser.add_argument_group('Fitness Function (3-term model)')
    fitness_group.add_argument('--fitness_alpha', type=float, default=1.0,
                              help='Plasticity weight (grad_norm) in fitness')
    fitness_group.add_argument('--fitness_beta', type=float, default=1.0,
                              help='Usefulness weight (activation_variance) in fitness')
    fitness_group.add_argument('--fitness_gamma', type=float, default=2.0,
                              help='Stagnation penalty weight (cosine similarity) in fitness')
    fitness_group.add_argument('--activation_ema_decay', type=float, default=0.9,
                              help='EMA decay for activation history tracking')

    # Senescence lifecycle
    senescence_group = parser.add_argument_group('Senescence Lifecycle Daemon')
    senescence_group.add_argument('--enable_senescence', action='store_true',
                                 help='Enable lifecycle senescence daemon')
    senescence_group.add_argument('--senescence_low_pct', type=float, default=0.1,
                                 help='Bottom percentile considered low-fitness')
    senescence_group.add_argument('--senescence_patience', type=int, default=20,
                                 help='Steps of low-fitness before at-risk')
    senescence_group.add_argument('--senescence_age_factor', type=float, default=0.01,
                                 help='Age multiplier contributing to risk')
    senescence_group.add_argument('--phase_durations', type=str, default='10,5,1,5',
                                 help='Comma list of durations for phases 1-4 (in steps)')
    senescence_group.add_argument('--lifecycle_warmup_steps', type=int, default=200,
                                 help='Warmup steps before senescence daemon starts')
    senescence_group.add_argument('--max_escalations_per_step', type=int, default=5,
                                 help='Max neurons escalated to at-risk per layer per step')
    senescence_group.add_argument('--max_kills_per_layer', type=int, default=3,
                                 help='Max neurons killed per layer per step')
    senescence_group.add_argument('--slope_window', type=int, default=5,
                                 help='Window (steps) used to compute fitness slope')
    senescence_group.add_argument('--slope_threshold', type=float, default=2e-4,
                                 help='Threshold for negative slope to consider escalation')
    senescence_group.add_argument('--min_to_escalate', type=int, default=2,
                                 help='Minimum neurons to escalate when candidates exist')

    # Logging
    logging_group = parser.add_argument_group('Logging and Output')
    logging_group.add_argument('--log_interval', type=int, default=250,
                              help='Steps between logging')
    logging_group.add_argument('--log_dir', type=str, default="logs",
                              help='Logging directory')
    logging_group.add_argument('--checkpoint_interval', type=int, default=2500,
                              help='Steps between checkpoints')
    logging_group.add_argument('--verbose', action='store_true',
                              help='Print generated samples')
    logging_group.add_argument('--output_buffer_size', type=int, default=10,
                              help='Activation history buffer size')
    logging_group.add_argument("--output_json", action="store_true",
                              help="Print structured JSON summary instead of normal prints")

    return parser


def parse_args_to_config() -> ExperimentConfig:
    """Parse command-line arguments and return ExperimentConfig."""
    parser = create_argument_parser()
    args = parser.parse_args()
    return ExperimentConfig.from_args(args)


# Predefined config presets
PRESET_CONFIGS = {
    'default': ExperimentConfig(),

    'quick_test': ExperimentConfig(
        training=TrainingConfig(
            num_steps=500,
            batch_size=128,
            log_interval=100,
            checkpoint_interval=500
        )
    ),

    'aggressive_apoptosis': ExperimentConfig(
        apoptosis=ApoptosisConfig(
            prune_rate=0.25,
            apoptosis_interval=100,
            mutation_strength=0.5
        )
    ),

    'senescence_enabled': ExperimentConfig(
        senescence=SenescenceConfig(
            enable_senescence=True,
            senescence_patience=15,
            max_kills_per_layer=5
        )
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]
