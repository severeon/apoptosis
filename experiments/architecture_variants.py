"""
Architecture Variants for Neuron-Level Apoptosis

Tests different layer selection patterns:
1. Standard: First 6 FFN layers (layers 0-5)
2. Deep Only: Last 3 layers only (layers 3-5, death zone)
3. Shallow Only: First 3 layers only (layers 0-2, birth zone)
4. Alternating: Every other layer (layers 0, 2, 4)
5. Attention Only: Target attention projections instead of FFN
6. Both FFN and Attention: Maximum coverage

Idea: Different layer patterns may have different evolutionary dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import json


class ArchitectureExperiment:
    """Test different layer targeting strategies."""

    def __init__(self, model, device, train_dataset, val_dataset):
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_layer_pattern(self, pattern_name: str) -> List[str]:
        """Get layer names for different patterns."""

        patterns = {
            # FFN patterns
            'standard': [f'blocks.{i}.ffn.0' for i in range(6)],
            'deep_only': [f'blocks.{i}.ffn.0' for i in range(3, 6)],  # Layers 3-5
            'shallow_only': [f'blocks.{i}.ffn.0' for i in range(3)],  # Layers 0-2
            'alternating': [f'blocks.{i}.ffn.0' for i in [0, 2, 4]],
            'middle_only': [f'blocks.{i}.ffn.0' for i in [2, 3]],  # Stable core

            # Attention patterns
            'attention_only': [f'blocks.{i}.attention.out_proj' for i in range(6)],
            'attention_shallow': [f'blocks.{i}.attention.out_proj' for i in range(3)],
            'attention_deep': [f'blocks.{i}.attention.out_proj' for i in range(3, 6)],

            # Combined patterns
            'ffn_and_attention': (
                [f'blocks.{i}.ffn.0' for i in range(6)] +
                [f'blocks.{i}.attention.out_proj' for i in range(6)]
            ),
            'shallow_ffn_deep_attention': (
                [f'blocks.{i}.ffn.0' for i in range(3)] +
                [f'blocks.{i}.attention.out_proj' for i in range(3, 6)]
            ),

            # Variable intensity patterns
            'high_turnover_everywhere': [f'blocks.{i}.ffn.0' for i in range(6)],
            'graduated': [f'blocks.{i}.ffn.0' for i in range(6)],  # Will use varying rates
        }

        return patterns.get(pattern_name, patterns['standard'])

    def create_graduated_managers(self, model, base_prune_rate: float = 0.10):
        """
        Create multiple managers with different turnover rates per layer.

        Idea: Shallow layers have high turnover (new concepts)
              Deep layers have low turnover (stable representations)
        """

        managers = []

        # Shallow (high turnover)
        shallow_mgr = NeuronApoptosisManager(
            model=model,
            target_layers=[f'blocks.{i}.ffn.0' for i in [0, 1]],
            prune_rate=base_prune_rate * 1.5,  # 15% turnover
            apoptosis_interval=500,
            fitness_metric='grad_activation',
            regrowth_strategy='mutation',
            mutation_strength=0.4  # Higher mutation for exploration
        )
        managers.append(('shallow_high_turnover', shallow_mgr))

        # Middle (moderate turnover)
        middle_mgr = NeuronApoptosisManager(
            model=model,
            target_layers=[f'blocks.{i}.ffn.0' for i in [2, 3]],
            prune_rate=base_prune_rate,  # 10% turnover
            apoptosis_interval=500,
            fitness_metric='grad_activation',
            regrowth_strategy='mutation',
            mutation_strength=0.3
        )
        managers.append(('middle_moderate_turnover', middle_mgr))

        # Deep (low turnover)
        deep_mgr = NeuronApoptosisManager(
            model=model,
            target_layers=[f'blocks.{i}.ffn.0' for i in [4, 5]],
            prune_rate=base_prune_rate * 0.5,  # 5% turnover
            apoptosis_interval=500,
            fitness_metric='grad_activation',
            regrowth_strategy='mutation',
            mutation_strength=0.2  # Lower mutation for stability
        )
        managers.append(('deep_low_turnover', deep_mgr))

        return managers

    def test_pattern(self, pattern_name: str, num_steps: int = 2000) -> Dict:
        """Test a single layer pattern."""

        print(f"\nTesting pattern: {pattern_name}")

        # Create fresh model
        model = ApoptoticTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=6,
            max_seq_len=128,
            enable_apoptosis=False
        ).to(self.device)

        # Get layer pattern
        target_layers = self.get_layer_pattern(pattern_name)
        print(f"  Target layers: {target_layers}")

        # Create manager
        if pattern_name == 'graduated':
            managers = self.create_graduated_managers(model)
        else:
            manager = NeuronApoptosisManager(
                model=model,
                target_layers=target_layers,
                prune_rate=0.10,
                apoptosis_interval=500,
                fitness_metric='grad_activation',
                regrowth_strategy='mutation',
                mutation_strength=0.3
            )
            managers = [('single', manager)]

        # Training setup
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        # Training
        train_losses = []
        train_iter = iter(train_loader)

        for step in range(num_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            model.train()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Apply all managers
            for name, mgr in managers:
                mgr.step()

            train_losses.append(loss.item())

        # Compute metrics
        final_loss = np.mean(train_losses[-100:])
        variance = np.var(train_losses[-500:])

        total_events = sum(len(mgr.apoptosis_events) for _, mgr in managers)

        result = {
            'pattern': pattern_name,
            'target_layers': target_layers,
            'final_loss': final_loss,
            'variance': variance,
            'num_events': total_events,
            'min_loss': min(train_losses),
            'max_loss': max(train_losses)
        }

        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Events: {total_events}")

        # Cleanup
        del model
        del managers
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    def run_all_patterns(self, num_steps: int = 2000):
        """Test all architecture patterns."""

        patterns = [
            'standard',
            'deep_only',
            'shallow_only',
            'alternating',
            'middle_only',
            'attention_only',
            'attention_shallow',
            'attention_deep',
            'ffn_and_attention',
            'shallow_ffn_deep_attention',
            'graduated'
        ]

        results = []

        print("\n" + "="*70)
        print("ARCHITECTURE PATTERN COMPARISON")
        print("="*70)
        print(f"Testing {len(patterns)} patterns, {num_steps} steps each\n")

        for pattern in patterns:
            try:
                result = self.test_pattern(pattern, num_steps)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'pattern': pattern, 'error': str(e)})

        # Analyze
        self.analyze_patterns(results)

        return results

    def analyze_patterns(self, results: List[Dict]):
        """Analyze and compare patterns."""

        print("\n" + "="*70)
        print("PATTERN COMPARISON RESULTS")
        print("="*70)

        valid = [r for r in results if 'error' not in r]

        # Sort by loss
        sorted_by_loss = sorted(valid, key=lambda x: x['final_loss'])

        print("\n🏆 BEST PATTERNS BY LOSS:")
        for i, r in enumerate(sorted_by_loss[:5]):
            print(f"\n{i+1}. {r['pattern']}")
            print(f"   Loss: {r['final_loss']:.4f}")
            print(f"   Variance: {r['variance']:.6f}")
            print(f"   Events: {r['num_events']}")

        # Sort by stability
        sorted_by_variance = sorted(valid, key=lambda x: x['variance'])

        print("\n📊 MOST STABLE PATTERNS:")
        for i, r in enumerate(sorted_by_variance[:5]):
            print(f"\n{i+1}. {r['pattern']}")
            print(f"   Variance: {r['variance']:.6f}")
            print(f"   Loss: {r['final_loss']:.4f}")
            print(f"   Events: {r['num_events']}")

        print("\n" + "="*70 + "\n")


# Simple test runner
def quick_architecture_test():
    """Quick test of a few key patterns."""

    print("\n" + "="*70)
    print("QUICK ARCHITECTURE TEST")
    print("="*70)

    exp = ArchitectureExperiment(
        model=None,  # Will create fresh models
        device=device,
        train_dataset=shakespeare_dataset,
        val_dataset=shakespeare_dataset
    )

    # Test just a few key patterns
    patterns_to_test = [
        'standard',        # Current approach
        'deep_only',       # Death zone only
        'shallow_only',    # Birth zone only
        'graduated',       # Variable turnover rates
        'ffn_and_attention'  # Maximum coverage
    ]

    results = []
    for pattern in patterns_to_test:
        result = exp.test_pattern(pattern, num_steps=2000)
        results.append(result)

    exp.analyze_patterns(results)

    return results


print("✓ Architecture variants loaded!")
print("\nUsage:")
print("  # Test all patterns")
print("  exp = ArchitectureExperiment(")
print("      model=None,")
print("      device=device,")
print("      train_dataset=shakespeare_dataset,")
print("      val_dataset=shakespeare_dataset")
print("  )")
print("  results = exp.run_all_patterns(num_steps=2000)")
print("\n  # Or quick test of key patterns:")
print("  quick_architecture_test()")
