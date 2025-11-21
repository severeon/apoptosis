"""
Automated Hyperparameter Sweep for Apoptosis Strategies

Tests multiple combinations of:
- Strategies (Standard, Functional, Growth-Only, Hybrid)
- Prune rates (5%, 10%, 15%, 20%)
- Intervals (250, 500, 750 steps)
- Mutation strengths (0.1, 0.2, 0.3, 0.4)

Runs each config for 2000 steps, records metrics, finds best configs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import os

# NOTE: This file expects the following classes to be already loaded in the global namespace:
# - NeuronApoptosisManager (from neuron_apoptosis_fixed.py)
# - FunctionalPreservationApoptosis, GradualFadeApoptosis (from smooth_apoptosis.py)
# - GrowthOnlyManager, HybridGrowthAndDeath (from growth_only_strategy.py)
# - ApoptoticTransformer, tokenizer, shakespeare_dataset, device (from main notebook)
#
# Load them BEFORE running this script:
#   exec(open('neuron_apoptosis_fixed.py').read())
#   exec(open('smooth_apoptosis.py').read())
#   exec(open('growth_only_strategy.py').read())
#   exec(open('hyperparameter_sweep.py').read())


class HyperparameterSweep:
    """Manages and executes hyperparameter sweep experiments."""

    def __init__(self,
                 model_class,
                 train_dataset,
                 val_dataset,
                 device,
                 num_steps: int = 2000,
                 eval_interval: int = 100):

        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_steps = num_steps
        self.eval_interval = eval_interval

        self.results = []

    def define_search_space(self) -> List[Dict]:
        """Define all configurations to test."""

        configs = []

        # Strategy 1: Standard Neuron Apoptosis
        for prune_rate in [0.05, 0.10, 0.15, 0.20]:
            for interval in [250, 500, 750]:
                for mutation_strength in [0.1, 0.2, 0.3, 0.4]:
                    configs.append({
                        'strategy': 'standard',
                        'prune_rate': prune_rate,
                        'interval': interval,
                        'mutation_strength': mutation_strength,
                        'fitness_metric': 'grad_activation',
                        'regrowth_strategy': 'mutation'
                    })

        # Strategy 2: Functional Preservation
        for prune_rate in [0.05, 0.10, 0.15]:
            for interval in [250, 500, 750]:
                for mutation_strength in [0.1, 0.2, 0.3]:
                    configs.append({
                        'strategy': 'functional',
                        'prune_rate': prune_rate,
                        'interval': interval,
                        'mutation_strength': mutation_strength,
                        'preservation_steps': 50
                    })

        # Strategy 3: Gradual Fade
        for prune_rate in [0.10, 0.15]:
            for interval in [500, 750]:
                for fade_duration in [25, 50]:
                    configs.append({
                        'strategy': 'gradual_fade',
                        'prune_rate': prune_rate,
                        'interval': interval,
                        'fade_duration': fade_duration,
                        'mutation_strength': 0.3
                    })

        # Strategy 4: Growth-Only (DISABLED - requires architecture changes)
        # Growth-only changes layer dimensions which breaks subsequent layers
        # TODO: Fix by also updating next layer's input dimensions
        # for growth_rate in [0.05, 0.10, 0.15]:
        #     for interval in [500, 750, 1000]:
        #         for max_capacity in [1.3, 1.5, 1.7]:
        #             configs.append({
        #                 'strategy': 'growth_only',
        #                 'growth_rate': growth_rate,
        #                 'growth_interval': interval,
        #                 'max_capacity': max_capacity,
        #                 'mutation_strength': 0.3
        #             })

        # Strategy 5: Hybrid Growth and Death
        for turnover_rate in [0.05, 0.10, 0.15]:
            for interval in [250, 500, 750]:
                configs.append({
                    'strategy': 'hybrid',
                    'turnover_rate': turnover_rate,
                    'interval': interval,
                    'mutation_strength': 0.3
                })

        print(f"Total configurations to test: {len(configs)}")
        return configs

    def create_manager(self, config: Dict, model):
        """Create appropriate manager based on config."""

        target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

        if config['strategy'] == 'standard':
            return NeuronApoptosisManager(
                model=model,
                target_layers=target_layers,
                prune_rate=config['prune_rate'],
                apoptosis_interval=config['interval'],
                fitness_metric=config['fitness_metric'],
                regrowth_strategy=config['regrowth_strategy'],
                mutation_strength=config['mutation_strength']
            )

        elif config['strategy'] == 'functional':
            # FunctionalPreservationApoptosis should be loaded in global namespace
            return FunctionalPreservationApoptosis(
                model=model,
                target_layers=target_layers,
                prune_rate=config['prune_rate'],
                apoptosis_interval=config['interval'],
                mutation_strength=config['mutation_strength'],
                preservation_steps=config.get('preservation_steps', 50)
            )

        elif config['strategy'] == 'gradual_fade':
            return GradualFadeApoptosis(
                model=model,
                target_layers=target_layers,
                prune_rate=config['prune_rate'],
                apoptosis_interval=config['interval'],
                fade_duration=config['fade_duration'],
                mutation_strength=config['mutation_strength']
            )

        elif config['strategy'] == 'growth_only':
            return GrowthOnlyManager(
                model=model,
                target_layers=target_layers,
                growth_rate=config['growth_rate'],
                growth_interval=config['growth_interval'],
                max_capacity=config['max_capacity'],
                mutation_strength=config['mutation_strength']
            )

        elif config['strategy'] == 'hybrid':
            return HybridGrowthAndDeath(
                model=model,
                target_layers=target_layers,
                turnover_rate=config['turnover_rate'],
                interval=config['interval'],
                mutation_strength=config['mutation_strength']
            )

        else:
            raise ValueError(f"Unknown strategy: {config['strategy']}")

    def run_single_experiment(self, config: Dict) -> Dict:
        """Run single experiment with given config."""

        # Create fresh model
        model = self.model_class(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=6,
            max_seq_len=128,
            enable_apoptosis=False
        ).to(self.device)

        # Create manager
        manager = self.create_manager(config, model)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        # Metrics
        train_losses = []
        val_losses = []

        # Training loop
        train_iter = iter(train_loader)

        for step in range(self.num_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Train step
            model.train()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Apoptosis step
            manager.step()

            train_losses.append(loss.item())

            # Eval
            if step % self.eval_interval == 0 and step > 0:
                model.eval()
                val_loss_sum = 0
                val_count = 0

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                        logits_val = model(x_val)
                        loss_val = criterion(
                            logits_val.reshape(-1, logits_val.size(-1)),
                            y_val.reshape(-1)
                        )
                        val_loss_sum += loss_val.item()
                        val_count += 1

                        if val_count >= 50:  # Quick eval
                            break

                val_losses.append(val_loss_sum / val_count)

        # Compute metrics
        final_train_loss = np.mean(train_losses[-100:])
        final_val_loss = val_losses[-1] if val_losses else float('inf')
        train_loss_variance = np.var(train_losses[-500:])

        # Get stats from manager
        if hasattr(manager, 'apoptosis_events'):
            num_events = len(manager.apoptosis_events)
        elif hasattr(manager, 'growth_events'):
            num_events = len(manager.growth_events)
        elif hasattr(manager, 'apoptosis_mgr'):
            # Hybrid manager wraps another manager
            num_events = len(manager.apoptosis_mgr.apoptosis_events)
        else:
            num_events = 0

        result = {
            'config': config,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'train_loss_variance': train_loss_variance,
            'num_events': num_events,
            'min_train_loss': min(train_losses),
            'max_train_loss': max(train_losses),
        }

        # Clean up
        del model
        del manager
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    def run_sweep(self):
        """Run full hyperparameter sweep."""

        configs = self.define_search_space()

        print(f"\n{'='*70}")
        print("HYPERPARAMETER SWEEP")
        print(f"{'='*70}")
        print(f"Total experiments: {len(configs)}")
        print(f"Steps per experiment: {self.num_steps}")
        print(f"Estimated time: {len(configs) * 5} minutes (~5 min per config)")
        print(f"{'='*70}\n")

        for i, config in enumerate(tqdm(configs, desc="Running experiments")):
            print(f"\n[{i+1}/{len(configs)}] Testing: {config['strategy']}")

            try:
                result = self.run_single_experiment(config)
                self.results.append(result)

                print(f"  Final loss: {result['final_train_loss']:.4f}")
                print(f"  Events: {result['num_events']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                self.results.append({
                    'config': config,
                    'error': str(e),
                    'final_train_loss': float('inf')
                })

        self.save_results()
        self.analyze_results()

    def save_results(self):
        """Save results to JSON file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sweep_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def analyze_results(self):
        """Analyze and print top configurations."""

        print(f"\n{'='*70}")
        print("TOP CONFIGURATIONS")
        print(f"{'='*70}\n")

        # Filter out errors
        valid_results = [r for r in self.results if 'error' not in r]

        # Sort by final loss
        sorted_by_loss = sorted(valid_results, key=lambda x: x['final_train_loss'])

        print("🏆 TOP 10 BY FINAL LOSS:")
        for i, result in enumerate(sorted_by_loss[:10]):
            config = result['config']
            print(f"\n{i+1}. {config['strategy'].upper()}")
            print(f"   Loss: {result['final_train_loss']:.4f}")
            print(f"   Variance: {result['train_loss_variance']:.6f}")
            print(f"   Events: {result['num_events']}")
            print(f"   Config: {config}")

        # Sort by stability (low variance)
        sorted_by_stability = sorted(valid_results, key=lambda x: x['train_loss_variance'])

        print(f"\n{'='*70}")
        print("📊 TOP 10 BY STABILITY (Low Variance):")
        for i, result in enumerate(sorted_by_stability[:10]):
            config = result['config']
            print(f"\n{i+1}. {config['strategy'].upper()}")
            print(f"   Variance: {result['train_loss_variance']:.6f}")
            print(f"   Loss: {result['final_train_loss']:.4f}")
            print(f"   Events: {result['num_events']}")
            print(f"   Config: {config}")

        # Best by strategy
        print(f"\n{'='*70}")
        print("🎯 BEST CONFIG PER STRATEGY:")

        strategies = set(r['config']['strategy'] for r in valid_results)
        for strategy in strategies:
            strategy_results = [r for r in valid_results if r['config']['strategy'] == strategy]
            best = min(strategy_results, key=lambda x: x['final_train_loss'])

            print(f"\n{strategy.upper()}:")
            print(f"   Loss: {best['final_train_loss']:.4f}")
            print(f"   Variance: {best['train_loss_variance']:.6f}")
            print(f"   Events: {best['num_events']}")
            print(f"   Config: {best['config']}")

        print(f"\n{'='*70}\n")


# Quick sweep for testing (fewer combinations)
class QuickSweep(HyperparameterSweep):
    """Faster sweep with fewer configs for testing."""

    def define_search_space(self) -> List[Dict]:
        """Define smaller search space."""

        configs = []

        # Standard - only test a few key combinations
        for prune_rate in [0.10, 0.15]:
            for interval in [500]:
                configs.append({
                    'strategy': 'standard',
                    'prune_rate': prune_rate,
                    'interval': interval,
                    'mutation_strength': 0.3,
                    'fitness_metric': 'grad_activation',
                    'regrowth_strategy': 'mutation'
                })

        # Functional
        for prune_rate in [0.10, 0.15]:
            configs.append({
                'strategy': 'functional',
                'prune_rate': prune_rate,
                'interval': 500,
                'mutation_strength': 0.3,
                'preservation_steps': 50
            })

        # Hybrid (test more since it's winning!)
        for turnover_rate in [0.05, 0.08, 0.10]:
            for interval in [400, 500, 600]:
                configs.append({
                    'strategy': 'hybrid',
                    'turnover_rate': turnover_rate,
                    'interval': interval,
                    'mutation_strength': 0.3
                })

        print(f"Quick sweep - testing {len(configs)} configurations")
        return configs


print("✓ Hyperparameter sweep system loaded!")
print("\nUsage:")
print("  # Full sweep (may take hours)")
print("  sweep = HyperparameterSweep(")
print("      model_class=ApoptoticTransformer,")
print("      train_dataset=shakespeare_dataset,")
print("      val_dataset=shakespeare_dataset,")
print("      device=device,")
print("      num_steps=2000")
print("  )")
print("  sweep.run_sweep()")
print("\n  # Quick sweep (10-15 configs, ~1 hour)")
print("  quick_sweep = QuickSweep(...same args...)")
print("  quick_sweep.run_sweep()")
