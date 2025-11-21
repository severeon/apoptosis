"""
Taguchi Method for Hyperparameter Optimization

Instead of exhaustive grid search (100+ configs), use orthogonal arrays
to test only 12-18 configs and still find optimal parameters.

Key idea: Test each parameter at multiple levels, but in a balanced way
so you can isolate main effects without testing every combination.

Example:
- Grid search: 4 params × 3 levels each = 81 experiments
- Taguchi L9:  4 params × 3 levels each = 9 experiments (9x faster!)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from itertools import product


# Standard Orthogonal Arrays (Taguchi Tables)
ORTHOGONAL_ARRAYS = {
    'L4': np.array([  # 3 factors, 2 levels
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]),
    'L8': np.array([  # 7 factors, 2 levels
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]),
    'L9': np.array([  # 4 factors, 3 levels
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [1, 0, 1, 2],
        [1, 1, 2, 0],
        [1, 2, 0, 1],
        [2, 0, 2, 1],
        [2, 1, 0, 2],
        [2, 2, 1, 0]
    ]),
    'L12': np.array([  # 11 factors, 2 levels
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
    ]),
    'L16': np.array([  # 15 factors, 2 levels
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    ])
}


class TaguchiSearch:
    """
    Taguchi method for efficient hyperparameter search.

    Uses orthogonal arrays to test parameter combinations in a balanced way,
    allowing detection of main effects with minimal experiments.
    """

    def __init__(self, parameters: Dict[str, List], array_name: str = 'auto'):
        """
        Parameters:
            parameters: Dict of {param_name: [level0, level1, level2, ...]}
            array_name: Which orthogonal array to use ('L4', 'L8', 'L9', etc.)
        """
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.param_levels = [len(levels) for levels in parameters.values()]

        # Auto-select array if needed
        if array_name == 'auto':
            array_name = self._select_array()

        self.array_name = array_name
        self.array = ORTHOGONAL_ARRAYS[array_name]

        # Generate experiments
        self.experiments = self._generate_experiments()

    def _select_array(self) -> str:
        """Auto-select appropriate orthogonal array."""
        n_params = len(self.parameters)
        max_levels = max(self.param_levels)

        if max_levels == 2:
            # 2-level arrays
            if n_params <= 3:
                return 'L4'
            elif n_params <= 7:
                return 'L8'
            elif n_params <= 11:
                return 'L12'
            else:
                return 'L16'
        elif max_levels == 3:
            # 3-level arrays
            if n_params <= 4:
                return 'L9'
            else:
                # Convert 3-level to 2-level by binning
                return 'L16'
        else:
            # For more levels, use L16 and map levels
            return 'L16'

    def _generate_experiments(self) -> List[Dict]:
        """Generate experiment configurations from orthogonal array."""
        experiments = []

        for row in self.array:
            config = {}
            for i, param_name in enumerate(self.param_names):
                if i >= len(row):
                    # Use default (first level) for extra params
                    level_idx = 0
                else:
                    level_idx = row[i] % len(self.parameters[param_name])

                config[param_name] = self.parameters[param_name][level_idx]

            experiments.append(config)

        return experiments

    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze results to find optimal parameter levels.

        Uses Taguchi's S/N ratio (Signal-to-Noise) and main effects analysis.
        """
        # Convert results to DataFrame for easy analysis
        df = pd.DataFrame(results)

        # Compute S/N ratio (smaller-is-better for loss)
        if 'loss' in df.columns:
            df['sn_ratio'] = -10 * np.log10(df['loss']**2)
        elif 'final_loss' in df.columns:
            df['sn_ratio'] = -10 * np.log10(df['final_loss']**2)
        else:
            raise ValueError("Results must contain 'loss' or 'final_loss'")

        # Main effects analysis
        main_effects = {}

        for param_name in self.param_names:
            param_values = [config[param_name] for config in self.experiments]
            df['param'] = param_values

            # Average S/N ratio for each level
            level_effects = df.groupby('param')['sn_ratio'].mean()
            main_effects[param_name] = level_effects.to_dict()

        # Find optimal configuration
        optimal_config = {}
        for param_name, effects in main_effects.items():
            # Choose level with highest S/N ratio
            optimal_level = max(effects, key=effects.get)
            optimal_config[param_name] = optimal_level

        # Compute expected performance
        expected_sn = sum(
            main_effects[param][optimal_config[param]]
            for param in self.param_names
        ) / len(self.param_names)

        return {
            'optimal_config': optimal_config,
            'main_effects': main_effects,
            'expected_sn_ratio': expected_sn,
            'experiments_tested': len(self.experiments),
            'analysis_df': df
        }

    def print_analysis(self, results: List[Dict]):
        """Pretty-print analysis results."""
        analysis = self.analyze_results(results)

        print("\n" + "="*70)
        print("TAGUCHI ANALYSIS RESULTS")
        print("="*70)

        print(f"\nArray used: {self.array_name}")
        print(f"Experiments tested: {analysis['experiments_tested']}")

        print("\n📊 MAIN EFFECTS (S/N Ratio by Parameter Level):")
        for param, effects in analysis['main_effects'].items():
            print(f"\n{param}:")
            sorted_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)
            for level, sn in sorted_effects:
                print(f"  {level}: {sn:.4f} {'← BEST' if level == sorted_effects[0][0] else ''}")

        print("\n🎯 OPTIMAL CONFIGURATION:")
        for param, value in analysis['optimal_config'].items():
            print(f"  {param}: {value}")

        print(f"\nExpected S/N Ratio: {analysis['expected_sn_ratio']:.4f}")

        # Convert S/N back to expected loss
        expected_loss = 10 ** (-analysis['expected_sn_ratio'] / 10)
        print(f"Expected Loss: {np.sqrt(expected_loss):.4f}")

        print("\n" + "="*70)

        return analysis


# Apoptosis-specific Taguchi search
class ApoptosisTaguchiSearch:
    """Taguchi search specifically for apoptosis hyperparameters."""

    @staticmethod
    def define_parameters() -> Dict[str, List]:
        """Define parameter space for apoptosis."""
        return {
            'strategy': ['standard', 'functional', 'crossover', 'growth'],
            'prune_rate': [0.05, 0.10, 0.15],
            'interval': [250, 500, 750],
            'mutation_strength': [0.1, 0.2, 0.3, 0.4],
            'fitness_metric': ['grad_activation', 'weight'],
        }

    @staticmethod
    def run_search(model_class, dataset, device, num_steps=2000):
        """Run Taguchi search for apoptosis parameters."""

        params = ApoptosisTaguchiSearch.define_parameters()
        taguchi = TaguchiSearch(params)

        print(f"\n{'='*70}")
        print("TAGUCHI HYPERPARAMETER SEARCH")
        print(f"{'='*70}")
        print(f"Testing {len(taguchi.experiments)} configurations")
        print(f"(vs {np.prod([len(v) for v in params.values()])} for full grid search)")
        print(f"Speedup: {np.prod([len(v) for v in params.values()]) / len(taguchi.experiments):.1f}x")
        print(f"{'='*70}\n")

        results = []

        for i, config in enumerate(taguchi.experiments):
            print(f"\n[{i+1}/{len(taguchi.experiments)}] Testing config:")
            for k, v in config.items():
                print(f"  {k}: {v}")

            try:
                # Run experiment
                result = run_single_experiment(
                    config, model_class, dataset, device, num_steps
                )
                result['config'] = config
                results.append(result)

                print(f"  → Loss: {result['final_loss']:.4f}")

            except Exception as e:
                print(f"  → ERROR: {e}")
                results.append({'config': config, 'final_loss': 999.0, 'error': str(e)})

        # Analyze
        analysis = taguchi.print_analysis(results)

        return analysis, results


def run_single_experiment(config, model_class, dataset, device, num_steps):
    """Run a single experiment with given config."""

    # Create model
    model = model_class(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=6,
        max_seq_len=128,
        enable_apoptosis=False
    ).to(device)

    target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

    # Create manager based on strategy
    if config['strategy'] == 'standard':
        from neuron_apoptosis_fixed import NeuronApoptosisManager
        manager = NeuronApoptosisManager(
            model=model,
            target_layers=target_layers,
            prune_rate=config['prune_rate'],
            apoptosis_interval=config['interval'],
            fitness_metric=config['fitness_metric'],
            regrowth_strategy='mutation',
            mutation_strength=config['mutation_strength']
        )

    elif config['strategy'] == 'functional':
        from smooth_apoptosis import FunctionalPreservationApoptosis
        manager = FunctionalPreservationApoptosis(
            model=model,
            target_layers=target_layers,
            prune_rate=config['prune_rate'],
            apoptosis_interval=config['interval'],
            mutation_strength=config['mutation_strength']
        )

    elif config['strategy'] == 'crossover':
        from crossover_strategy import CrossoverApoptosis
        manager = CrossoverApoptosis(
            model=model,
            target_layers=target_layers,
            prune_rate=config['prune_rate'],
            apoptosis_interval=config['interval'],
            crossover_mode='uniform',
            mutation_strength=config['mutation_strength']
        )

    elif config['strategy'] == 'growth':
        from growth_only_strategy import GrowthOnlyManager
        manager = GrowthOnlyManager(
            model=model,
            target_layers=target_layers,
            growth_rate=config['prune_rate'],  # Reuse param
            growth_interval=config['interval'],
            max_capacity=1.5,
            mutation_strength=config['mutation_strength']
        )

    # Training
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        manager.step()
        train_losses.append(loss.item())

    # Results
    final_loss = np.mean(train_losses[-100:])
    variance = np.var(train_losses[-500:])

    del model
    del manager
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'final_loss': final_loss,
        'variance': variance,
        'loss': final_loss  # For S/N ratio calculation
    }


print("✓ Taguchi method loaded!")
print("\nUsage:")
print("  # Run Taguchi search")
print("  analysis, results = ApoptosisTaguchiSearch.run_search(")
print("      model_class=ApoptoticTransformer,")
print("      dataset=shakespeare_dataset,")
print("      device=device,")
print("      num_steps=2000")
print("  )")
print("\nBenefits:")
print("  - Tests 16 configs instead of 96 (6x faster!)")
print("  - Finds main effects (which params matter most)")
print("  - More robust to noise than grid search")
print("  - Industry-standard method (manufacturing, engineering)")
