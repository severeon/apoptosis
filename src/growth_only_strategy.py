"""
Growth-Only Strategy: Neurogenesis Without Apoptosis

Key idea: Add new neurons periodically, never kill old ones.
Let the network naturally downweight less useful neurons.

Advantages:
- No disruption from death
- Network capacity grows over time
- Old knowledge preserved
- Natural "wisdom" accumulation

Constraints:
- Cap at 150% original capacity (memory/compute limit)
- New neurons start with low learning rate (don't disrupt existing)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class GrowthOnlyManager:
    """
    Manages neuron growth without death.

    Timeline:
    - Step 0: Start with 512 neurons per layer
    - Step 500: Add 51 neurons (10% growth)
    - Step 1000: Add 51 more (now 614 total)
    - ...
    - Step 2500: Hit cap at 768 neurons (150% of original)
    - After cap: No more growth, just training
    """

    def __init__(self,
                 model,
                 target_layers: List[str],
                 growth_rate: float = 0.1,  # Grow by 10%
                 growth_interval: int = 500,
                 max_capacity: float = 1.5,  # Cap at 150%
                 mutation_strength: float = 0.3,
                 fitness_metric: str = 'grad_activation'):

        self.model = model
        self.target_layers = target_layers
        self.growth_rate = growth_rate
        self.growth_interval = growth_interval
        self.max_capacity = max_capacity
        self.mutation_strength = mutation_strength
        self.fitness_metric = fitness_metric

        self.step_count = 0
        self.growth_events = []

        # Track original and current sizes
        self.original_sizes = {}
        self.current_sizes = {}

        # Activation hooks
        self.activations = {}

        self._initialize()

    def _initialize(self):
        """Record original layer sizes."""
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            self.original_sizes[layer_name] = layer.out_features
            self.current_sizes[layer_name] = layer.out_features

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            layer.register_forward_hook(hook_fn(layer_name))

    def _get_layer(self, layer_name: str):
        """Get layer by name."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def compute_fitness(self, layer_name: str) -> torch.Tensor:
        """Compute fitness for existing neurons."""
        layer = self._get_layer(layer_name)

        if self.fitness_metric == 'grad_activation':
            if layer.weight.grad is None or layer_name not in self.activations:
                return torch.ones(self.current_sizes[layer_name], device=layer.weight.device)

            grad_norm = torch.norm(layer.weight.grad, dim=1)
            activation = self.activations[layer_name]
            act_norm = torch.abs(activation).mean(dim=(0, 1))

            # Pad if sizes don't match (after growth)
            if len(act_norm) < len(grad_norm):
                pad_size = len(grad_norm) - len(act_norm)
                act_norm = torch.cat([act_norm, torch.ones(pad_size, device=act_norm.device)])

            fitness = grad_norm * act_norm
        else:
            fitness = torch.norm(layer.weight, dim=1)

        return fitness

    def step(self) -> bool:
        """Check for growth events."""
        self.step_count += 1

        if self.step_count % self.growth_interval == 0:
            return self.trigger_growth()

        return False

    def trigger_growth(self) -> bool:
        """Add new neurons to all target layers."""
        growth_occurred = False

        for layer_name in self.target_layers:
            # Check if we've hit capacity
            original_size = self.original_sizes[layer_name]
            current_size = self.current_sizes[layer_name]
            max_size = int(original_size * self.max_capacity)

            if current_size >= max_size:
                continue

            # Calculate growth
            num_to_add = min(
                int(original_size * self.growth_rate),
                max_size - current_size
            )

            if num_to_add == 0:
                continue

            print(f"\n[Growth @ step {self.step_count}] {layer_name}")
            print(f"  Adding {num_to_add} neurons ({current_size} → {current_size + num_to_add})")

            # Grow the layer
            self._grow_layer(layer_name, num_to_add)

            self.current_sizes[layer_name] += num_to_add
            self.growth_events.append((self.step_count, layer_name, num_to_add))
            growth_occurred = True

        return growth_occurred

    def _grow_layer(self, layer_name: str, num_to_add: int):
        """Physically add neurons to a layer."""
        layer = self._get_layer(layer_name)

        # Compute fitness to find good parent neurons
        fitness = self.compute_fitness(layer_name)

        # Create new weight matrix (expanded)
        old_out_features = layer.out_features
        new_out_features = old_out_features + num_to_add
        in_features = layer.in_features

        # New weight tensor
        new_weight = torch.zeros(new_out_features, in_features,
                                device=layer.weight.device,
                                dtype=layer.weight.dtype)

        # Copy old weights
        new_weight[:old_out_features, :] = layer.weight.data

        # Initialize new neurons (mutations of high-fitness neurons)
        if len(fitness) > 0:
            # Sample parents weighted by fitness
            fitness_np = fitness.cpu().numpy()
            probs = fitness_np / (fitness_np.sum() + 1e-8)

            for i in range(num_to_add):
                parent_idx = np.random.choice(len(fitness), p=probs)

                # Mutate parent
                new_weight[old_out_features + i, :] = layer.weight[parent_idx, :].clone()
                noise = torch.randn_like(layer.weight[parent_idx, :]) * self.mutation_strength
                new_weight[old_out_features + i, :] += noise
        else:
            # Fallback: random init
            nn.init.xavier_uniform_(new_weight[old_out_features:, :])

        # Update layer
        layer.weight = nn.Parameter(new_weight)

        # Handle bias
        if layer.bias is not None:
            new_bias = torch.zeros(new_out_features,
                                  device=layer.bias.device,
                                  dtype=layer.bias.dtype)
            new_bias[:old_out_features] = layer.bias.data

            # New biases: copy from parents
            if len(fitness) > 0:
                for i in range(num_to_add):
                    parent_idx = np.random.choice(len(fitness), p=probs)
                    new_bias[old_out_features + i] = layer.bias[parent_idx].clone()

            layer.bias = nn.Parameter(new_bias)

        # Update layer's out_features attribute
        layer.out_features = new_out_features

        print(f"  → Layer expanded: weight {layer.weight.shape}, bias {layer.bias.shape if layer.bias is not None else 'None'}")

    def get_stats(self) -> Dict:
        """Get growth statistics."""
        stats = {}
        for layer_name in self.target_layers:
            original = self.original_sizes[layer_name]
            current = self.current_sizes[layer_name]
            growth_pct = ((current - original) / original) * 100

            stats[layer_name] = {
                'original': original,
                'current': current,
                'growth_pct': growth_pct,
                'at_capacity': current >= int(original * self.max_capacity)
            }
        return stats


class HybridGrowthAndDeath:
    """
    Combination: Grow neurons AND prune weak ones.

    Strategy:
    - Every 500 steps: Add 5% new neurons (mutations of strong)
    - Every 500 steps: Remove 5% weak neurons (bottom fitness)
    - Net effect: Constant capacity, but 10% turnover per event

    This is "steady state" evolution - birth rate = death rate.
    """

    def __init__(self,
                 model,
                 target_layers: List[str],
                 turnover_rate: float = 0.05,  # 5% birth + 5% death
                 interval: int = 500,
                 mutation_strength: float = 0.3,
                 fitness_metric: str = 'grad_activation'):

        # Reuse existing components
        from neuron_apoptosis_fixed import NeuronApoptosisManager

        self.apoptosis_mgr = NeuronApoptosisManager(
            model=model,
            target_layers=target_layers,
            prune_rate=turnover_rate,
            apoptosis_interval=interval,
            fitness_metric=fitness_metric,
            regrowth_strategy='mutation',
            mutation_strength=mutation_strength
        )

        self.step_count = 0
        self.events = []

    def step(self) -> bool:
        """Prune weak and add strong (net zero change)."""
        return self.apoptosis_mgr.step()

    def get_stats(self):
        return self.apoptosis_mgr.get_stats()


print("✓ Growth-only strategies loaded!")
print("\nAvailable:")
print("  1. GrowthOnlyManager - Add neurons, never remove")
print("  2. HybridGrowthAndDeath - Constant turnover (birth = death)")
