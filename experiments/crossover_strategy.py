"""
Genetic Crossover Strategy - "Selective Breeding"

Instead of mutating a single parent, blend TWO high-fitness parents.
This is sexual reproduction for neurons! 🧬

Key difference from mutation:
- Mutation:  child = parent + noise
- Crossover: child = alpha*parent1 + (1-alpha)*parent2 + small_noise

Expected benefits:
- Combines strengths from multiple high-fitness neurons
- More stable (averaging reduces extremes)
- Greater genetic diversity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class CrossoverApoptosis:
    """
    Neuron apoptosis with genetic crossover (sexual reproduction).

    At each apoptosis event:
    1. Find weakest neurons (bottom X%)
    2. For each weak neuron:
       - Sample TWO high-fitness parents
       - Blend them: child = alpha*p1 + (1-alpha)*p2
       - Add small noise for variation
    """

    def __init__(self,
                 model,
                 target_layers: List[str],
                 prune_rate: float = 0.10,
                 apoptosis_interval: int = 500,
                 fitness_metric: str = 'grad_activation',
                 crossover_mode: str = 'uniform',  # 'uniform', 'fitness_weighted', 'random'
                 mutation_strength: float = 0.1):  # Smaller than single-parent (0.3)

        self.model = model
        self.target_layers = target_layers
        self.prune_rate = prune_rate
        self.apoptosis_interval = apoptosis_interval
        self.fitness_metric = fitness_metric
        self.crossover_mode = crossover_mode
        self.mutation_strength = mutation_strength

        self.step_count = 0
        self.apoptosis_events = []
        self.activations = {}

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
        """Compute fitness scores."""
        layer = self._get_layer(layer_name)

        if self.fitness_metric == 'grad_activation':
            if layer.weight.grad is None or layer_name not in self.activations:
                return torch.ones(layer.out_features, device=layer.weight.device)

            grad_norm = torch.norm(layer.weight.grad, dim=1)
            activation = self.activations[layer_name]
            act_norm = torch.abs(activation).mean(dim=(0, 1))

            # Pad if needed
            if len(act_norm) < len(grad_norm):
                pad_size = len(grad_norm) - len(act_norm)
                act_norm = torch.cat([act_norm, torch.ones(pad_size, device=act_norm.device)])

            fitness = grad_norm * act_norm

        elif self.fitness_metric == 'weight':
            fitness = torch.norm(layer.weight, dim=1)

        else:
            raise ValueError(f"Unknown fitness metric: {self.fitness_metric}")

        return fitness

    def step(self) -> bool:
        """Update and check for apoptosis."""
        self.step_count += 1

        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False

    def trigger_apoptosis(self) -> bool:
        """Prune weak, regrow via crossover."""
        apoptosis_occurred = False

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            fitness = self.compute_fitness(layer_name)

            num_neurons = len(fitness)
            num_to_prune = max(1, int(num_neurons * self.prune_rate))

            _, sorted_indices = torch.sort(fitness)
            dying_neurons = sorted_indices[:num_to_prune].cpu().numpy()
            healthy_neurons = sorted_indices[num_to_prune:].cpu().numpy()

            if len(dying_neurons) == 0 or len(healthy_neurons) < 2:
                continue

            print(f"\n[Crossover Apoptosis @ step {self.step_count}] {layer_name}")
            print(f"  Breeding {len(dying_neurons)} new neurons from top performers")
            print(f"  Fitness range: [{fitness.min().item():.4f}, {fitness.max().item():.4f}]")

            # Prune
            self._prune_neurons(layer, dying_neurons)

            # Regrow via crossover
            self._crossover_regrowth(layer, dying_neurons, healthy_neurons, fitness)

            self.apoptosis_events.append((self.step_count, layer_name, len(dying_neurons)))
            apoptosis_occurred = True

        return apoptosis_occurred

    def _prune_neurons(self, layer, neuron_indices):
        """Zero out dying neurons."""
        with torch.no_grad():
            layer.weight[neuron_indices, :] = 0
            if layer.bias is not None:
                layer.bias[neuron_indices] = 0

    def _crossover_regrowth(self, layer, dying_indices, healthy_indices, fitness):
        """Regrow neurons via crossover of two parents."""

        healthy_fitness = fitness[healthy_indices].cpu().numpy()
        probs = healthy_fitness / (healthy_fitness.sum() + 1e-8)

        with torch.no_grad():
            for neuron_idx in dying_indices:
                # Sample TWO parents (without replacement)
                if len(healthy_indices) >= 2:
                    parent1_idx, parent2_idx = np.random.choice(
                        healthy_indices,
                        size=2,
                        replace=False,
                        p=probs
                    )
                else:
                    # Fallback: single parent
                    parent1_idx = healthy_indices[0]
                    parent2_idx = healthy_indices[0]

                # Determine crossover ratio
                if self.crossover_mode == 'uniform':
                    alpha = 0.5  # Equal blend

                elif self.crossover_mode == 'fitness_weighted':
                    f1 = fitness[parent1_idx].item()
                    f2 = fitness[parent2_idx].item()
                    alpha = f1 / (f1 + f2 + 1e-8)

                elif self.crossover_mode == 'random':
                    alpha = np.random.uniform(0.3, 0.7)

                else:
                    alpha = 0.5

                # Crossover
                child_weight = (
                    alpha * layer.weight[parent1_idx, :] +
                    (1 - alpha) * layer.weight[parent2_idx, :]
                )

                # Small mutation (exploration)
                noise = torch.randn_like(child_weight) * self.mutation_strength
                child_weight = child_weight + noise

                layer.weight[neuron_idx, :] = child_weight

                # Bias
                if layer.bias is not None:
                    child_bias = (
                        alpha * layer.bias[parent1_idx] +
                        (1 - alpha) * layer.bias[parent2_idx]
                    )
                    noise_bias = torch.randn(1, device=layer.bias.device).item() * self.mutation_strength
                    layer.bias[neuron_idx] = child_bias + noise_bias

    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            'apoptosis_events': len(self.apoptosis_events),
            'total_neurons_bred': sum(count for _, _, count in self.apoptosis_events)
        }


class ComparisonRunner:
    """Compare mutation vs crossover strategies."""

    @staticmethod
    def test_strategy(strategy_class, config, model_class, dataset, device, num_steps=2000):
        """Test a single strategy."""

        model = model_class(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=6,
            max_seq_len=128,
            enable_apoptosis=False
        ).to(device)

        target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

        if strategy_class == 'crossover':
            manager = CrossoverApoptosis(
                model=model,
                target_layers=target_layers,
                **config
            )
        else:
            # Use existing NeuronApoptosisManager
            from neuron_apoptosis_fixed import NeuronApoptosisManager
            manager = NeuronApoptosisManager(
                model=model,
                target_layers=target_layers,
                **config
            )

        # Training
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
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
            'min_loss': min(train_losses),
            'max_loss': max(train_losses)
        }


print("✓ Genetic crossover strategy loaded!")
print("\nUsage:")
print("  # Create crossover manager")
print("  crossover_mgr = CrossoverApoptosis(")
print("      model=model,")
print("      target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],")
print("      prune_rate=0.10,")
print("      apoptosis_interval=500,")
print("      crossover_mode='uniform',  # or 'fitness_weighted', 'random'")
print("      mutation_strength=0.1")
print("  )")
print("\n  # Use in training loop")
print("  crossover_mgr.step()")
print("\nCrossover modes:")
print("  - uniform: 50/50 blend of parents")
print("  - fitness_weighted: Higher fitness parent contributes more")
print("  - random: Random ratio between 30-70%")
