"""
Neuron-Level Apoptosis System (Fixed)

Instead of killing entire layers, prune and regrow individual neurons
based on fitness scores.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NeuronMetadata:
    """Metadata for a single neuron."""
    age: int = 0
    birth_step: int = 0
    fitness_history: list = None

    def __post_init__(self):
        if self.fitness_history is None:
            self.fitness_history = []


class NeuronApoptosisManager:
    """
    Manages neuron-level apoptosis and neurogenesis.

    Strategy:
    - Every N steps, compute fitness for all neurons in target layers
    - Prune bottom X% of neurons (set weights to zero)
    - Regrow same number of neurons (reinitialize from high-fitness neurons)
    """

    def __init__(self,
                 model,
                 target_layers: List[str],
                 prune_rate: float = 0.1,
                 apoptosis_interval: int = 500,
                 fitness_metric: str = 'grad_activation',
                 regrowth_strategy: str = 'mutation',
                 mutation_strength: float = 0.3):

        self.model = model
        self.target_layers = target_layers
        self.prune_rate = prune_rate
        self.apoptosis_interval = apoptosis_interval
        self.fitness_metric = fitness_metric
        self.regrowth_strategy = regrowth_strategy
        self.mutation_strength = mutation_strength

        self.step_count = 0
        self.apoptosis_events = []

        # Track neuron metadata
        self.neuron_metadata: Dict[str, List[NeuronMetadata]] = {}

        # Activation hook storage
        self.activations = {}

        # Initialize metadata and hooks
        self._initialize_metadata()
        self._register_hooks()

    def _initialize_metadata(self):
        """Initialize metadata for all neurons in target layers."""
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            if isinstance(layer, nn.Linear):
                num_neurons = layer.out_features
            else:
                raise ValueError(f"Layer {layer_name} is not a Linear layer")

            self.neuron_metadata[layer_name] = [
                NeuronMetadata(age=0, birth_step=0)
                for _ in range(num_neurons)
            ]

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            layer.register_forward_hook(hook_fn(layer_name))

    def _get_layer(self, layer_name: str):
        """Get layer by name (supports nested attributes)."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def compute_fitness(self, layer_name: str) -> torch.Tensor:
        """Compute fitness scores for all neurons in a layer."""
        layer = self._get_layer(layer_name)

        if self.fitness_metric == 'gradient':
            if layer.weight.grad is None:
                return torch.ones(layer.out_features, device=layer.weight.device)
            fitness = torch.norm(layer.weight.grad, dim=1)

        elif self.fitness_metric == 'weight':
            fitness = torch.norm(layer.weight, dim=1)

        elif self.fitness_metric == 'grad_activation':
            if layer.weight.grad is None or layer_name not in self.activations:
                return torch.ones(layer.out_features, device=layer.weight.device)

            grad_norm = torch.norm(layer.weight.grad, dim=1)
            activation = self.activations[layer_name]
            act_norm = torch.abs(activation).mean(dim=(0, 1))
            fitness = grad_norm * act_norm

        elif self.fitness_metric == 'composite':
            weight_fitness = torch.norm(layer.weight, dim=1)

            if layer.weight.grad is not None:
                grad_fitness = torch.norm(layer.weight.grad, dim=1)
            else:
                grad_fitness = torch.zeros_like(weight_fitness)

            if layer_name in self.activations:
                act = self.activations[layer_name]
                act_fitness = torch.abs(act).mean(dim=(0, 1))
            else:
                act_fitness = torch.ones_like(weight_fitness)

            weight_fitness = weight_fitness / (weight_fitness.max() + 1e-8)
            grad_fitness = grad_fitness / (grad_fitness.max() + 1e-8)
            act_fitness = act_fitness / (act_fitness.max() + 1e-8)

            fitness = 0.4 * weight_fitness + 0.4 * grad_fitness + 0.2 * act_fitness

        else:
            raise ValueError(f"Unknown fitness metric: {self.fitness_metric}")

        return fitness

    def step(self) -> bool:
        """Update ages and check for apoptosis."""
        self.step_count += 1

        for layer_name in self.target_layers:
            for meta in self.neuron_metadata[layer_name]:
                meta.age += 1

        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False

    def trigger_apoptosis(self) -> bool:
        """Prune low-fitness neurons and regrow new ones."""
        apoptosis_occurred = False

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            fitness = self.compute_fitness(layer_name)

            num_neurons = len(fitness)
            num_to_prune = max(1, int(num_neurons * self.prune_rate))

            _, sorted_indices = torch.sort(fitness)
            dying_neurons = sorted_indices[:num_to_prune].cpu().numpy()
            healthy_neurons = sorted_indices[num_to_prune:].cpu().numpy()

            if len(dying_neurons) == 0:
                continue

            print(f"\n[Neuron Apoptosis @ step {self.step_count}] {layer_name}")
            print(f"  Pruning {len(dying_neurons)} neurons (bottom {self.prune_rate*100:.0f}%)")
            print(f"  Fitness range: [{fitness.min().item():.4f}, {fitness.max().item():.4f}]")

            self._prune_neurons(layer, dying_neurons)
            self._regrow_neurons(layer, dying_neurons, healthy_neurons, fitness)

            for neuron_idx in dying_neurons:
                self.neuron_metadata[layer_name][neuron_idx] = NeuronMetadata(
                    age=0, birth_step=self.step_count
                )

            self.apoptosis_events.append((self.step_count, layer_name, len(dying_neurons)))
            apoptosis_occurred = True

        return apoptosis_occurred

    def _prune_neurons(self, layer, neuron_indices):
        """Zero out weights for dying neurons."""
        with torch.no_grad():
            layer.weight[neuron_indices, :] = 0
            if layer.bias is not None:
                layer.bias[neuron_indices] = 0

    def _regrow_neurons(self, layer, dying_indices, healthy_indices, fitness):
        """Regrow neurons using specified strategy."""

        if self.regrowth_strategy == 'random':
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    if layer.bias is not None:
                        layer.bias[neuron_idx] = 0

        elif self.regrowth_strategy == 'mutation':
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    if len(healthy_indices) == 0:
                        nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    else:
                        healthy_fitness = fitness[healthy_indices].cpu().numpy()
                        probs = healthy_fitness / (healthy_fitness.sum() + 1e-8)
                        parent_idx = np.random.choice(healthy_indices, p=probs)

                        layer.weight[neuron_idx, :] = layer.weight[parent_idx, :].clone()
                        noise = torch.randn_like(layer.weight[neuron_idx, :]) * self.mutation_strength
                        layer.weight[neuron_idx, :] += noise

                        if layer.bias is not None:
                            layer.bias[neuron_idx] = layer.bias[parent_idx].clone()
                            layer.bias[neuron_idx] += torch.randn(1, device=layer.bias.device).item() * self.mutation_strength

        elif self.regrowth_strategy == 'clone':
            with torch.no_grad():
                if len(healthy_indices) > 0:
                    best_neuron = healthy_indices[fitness[healthy_indices].argmax()]
                    for neuron_idx in dying_indices:
                        layer.weight[neuron_idx, :] = layer.weight[best_neuron, :].clone()
                        if layer.bias is not None:
                            layer.bias[neuron_idx] = layer.bias[best_neuron].clone()

    def get_stats(self) -> Dict:
        """Get statistics about neuron ages."""
        stats = {}
        for layer_name in self.target_layers:
            ages = [meta.age for meta in self.neuron_metadata[layer_name]]
            stats[layer_name] = {
                'mean_age': np.mean(ages),
                'max_age': np.max(ages),
                'min_age': np.min(ages),
                'age_std': np.std(ages),
            }
        return stats


print("✓ Neuron-level apoptosis system loaded!")
