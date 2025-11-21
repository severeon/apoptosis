"""
Neuron-Level Apoptosis System

Instead of killing entire layers, prune and regrow individual neurons
based on fitness scores.

Key advantages over layer-level:
1. More granular control
2. Can work with existing architecture
3. Doesn't cripple the model (only remove weakest neurons)
4. Biologically more accurate (individual cell death)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class NeuronMetadata:
    """Metadata for a single neuron."""
    age: int = 0                    # Steps since birth
    birth_step: int = 0             # Global step when born
    fitness_history: List[float] = None  # Rolling fitness scores

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
                 model: nn.Module,
                 target_layers: List[str],  # e.g., ['blocks.0.ffn.0', 'blocks.1.ffn.0']
                 prune_rate: float = 0.1,  # Prune 10% of neurons
                 apoptosis_interval: int = 500,
                 fitness_metric: str = 'grad_activation',  # 'gradient', 'weight', 'grad_activation'
                 regrowth_strategy: str = 'mutation',  # 'mutation', 'random', 'clone'
                 mutation_strength: float = 0.3):

        self.model = model
        self.target_layers = target_layers
        self.prune_rate = prune_rate
        self.apoptosis_interval = apoptosis_interval
        self.fitness_metric = fitness_metric
        self.regrowth_strategy = regrowth_strategy
        self.mutation_strength = mutation_strength

        self.step_count = 0
        self.apoptosis_events = []  # List of (step, layer_name, num_pruned)

        # Track neuron metadata
        self.neuron_metadata: Dict[str, List[NeuronMetadata]] = {}

        # Initialize metadata for all neurons
        self._initialize_metadata()

        # Activation hook storage
        self.activations = {}
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
                # Store activations (detach to save memory)
                self.activations[name] = output.detach()
            return hook

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            layer.register_forward_hook(hook_fn(layer_name))

    def _get_layer(self, layer_name: str) -> nn.Module:
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
        """
        Compute fitness scores for all neurons in a layer.

        Returns:
            fitness: (num_neurons,) tensor of fitness scores
        """
        layer = self._get_layer(layer_name)

        if self.fitness_metric == 'gradient':
            # Pure gradient magnitude
            if layer.weight.grad is None:
                return torch.ones(layer.out_features, device=layer.weight.device)

            fitness = torch.norm(layer.weight.grad, dim=1)  # Per output neuron

        elif self.fitness_metric == 'weight':
            # Weight magnitude (L2 norm)
            fitness = torch.norm(layer.weight, dim=1)

        elif self.fitness_metric == 'grad_activation':
            # Gradient × Activation (Taylor approximation of importance)
            if layer.weight.grad is None or layer_name not in self.activations:
                return torch.ones(layer.out_features, device=layer.weight.device)

            grad_norm = torch.norm(layer.weight.grad, dim=1)
            activation = self.activations[layer_name]

            # Mean absolute activation per neuron
            act_norm = torch.abs(activation).mean(dim=(0, 1))  # (batch, seq, neurons) -> (neurons,)

            fitness = grad_norm * act_norm

        elif self.fitness_metric == 'composite':
            # Weighted combination
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

            # Normalize each component
            weight_fitness = weight_fitness / (weight_fitness.max() + 1e-8)
            grad_fitness = grad_fitness / (grad_fitness.max() + 1e-8)
            act_fitness = act_fitness / (act_fitness.max() + 1e-8)

            fitness = 0.4 * weight_fitness + 0.4 * grad_fitness + 0.2 * act_fitness

        else:
            raise ValueError(f"Unknown fitness metric: {self.fitness_metric}")

        return fitness

    def step(self) -> bool:
        """Update ages and check for apoptosis. Returns True if apoptosis occurred."""
        self.step_count += 1

        # Age all neurons
        for layer_name in self.target_layers:
            for meta in self.neuron_metadata[layer_name]:
                meta.age += 1

        # Check for apoptosis
        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False

    def trigger_apoptosis(self) -> bool:
        """Prune low-fitness neurons and regrow new ones."""
        apoptosis_occurred = False

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)

            # Compute fitness
            fitness = self.compute_fitness(layer_name)

            # Determine pruning threshold (bottom X%)
            num_neurons = len(fitness)
            num_to_prune = max(1, int(num_neurons * self.prune_rate))

            # Find weakest neurons
            _, sorted_indices = torch.sort(fitness)
            dying_neurons = sorted_indices[:num_to_prune].cpu().numpy()
            healthy_neurons = sorted_indices[num_to_prune:].cpu().numpy()

            if len(dying_neurons) == 0:
                continue

            print(f"\n[Neuron Apoptosis @ step {self.step_count}] {layer_name}")
            print(f"  Pruning {len(dying_neurons)} neurons (bottom {self.prune_rate*100:.0f}%)")
            print(f"  Fitness range: [{fitness.min().item():.4f}, {fitness.max().item():.4f}]")
            print(f"  Dying neurons: {dying_neurons.tolist()[:5]}..." if len(dying_neurons) > 5 else f"  Dying neurons: {dying_neurons.tolist()}")

            # Prune and regrow
            self._prune_neurons(layer, dying_neurons)
            self._regrow_neurons(layer, dying_neurons, healthy_neurons, fitness)

            # Update metadata
            for neuron_idx in dying_neurons:
                self.neuron_metadata[layer_name][neuron_idx] = NeuronMetadata(
                    age=0,
                    birth_step=self.step_count
                )

            self.apoptosis_events.append((self.step_count, layer_name, len(dying_neurons)))
            apoptosis_occurred = True

        return apoptosis_occurred

    def _prune_neurons(self, layer: nn.Linear, neuron_indices: np.ndarray):
        """Zero out weights for dying neurons."""
        with torch.no_grad():
            # Zero outgoing weights (rows of weight matrix)
            layer.weight[neuron_indices, :] = 0

            # Zero bias
            if layer.bias is not None:
                layer.bias[neuron_indices] = 0

    def _regrow_neurons(self, layer: nn.Linear, dying_indices: np.ndarray,
                       healthy_indices: np.ndarray, fitness: torch.Tensor):
        """Regrow neurons using specified strategy."""

        if self.regrowth_strategy == 'random':
            # Random reinitialization
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    if layer.bias is not None:
                        layer.bias[neuron_idx] = 0

        elif self.regrowth_strategy == 'mutation':
            # Evolutionary mutation: inherit from high-fitness neurons
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    # Sample a healthy neuron (weighted by fitness)
                    if len(healthy_indices) == 0:
                        # Fallback to random
                        nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    else:
                        # Weighted sampling
                        healthy_fitness = fitness[healthy_indices].cpu().numpy()
                        probs = healthy_fitness / (healthy_fitness.sum() + 1e-8)
                        parent_idx = np.random.choice(healthy_indices, p=probs)

                        # Copy and mutate
                        layer.weight[neuron_idx, :] = layer.weight[parent_idx, :].clone()
                        noise = torch.randn_like(layer.weight[neuron_idx, :]) * self.mutation_strength
                        layer.weight[neuron_idx, :] += noise

                        if layer.bias is not None:
                            layer.bias[neuron_idx] = layer.bias[parent_idx].clone()
                            layer.bias[neuron_idx] += torch.randn(1, device=layer.bias.device).item() * self.mutation_strength

        elif self.regrowth_strategy == 'clone':
            # Clone best neuron (no mutation)
            with torch.no_grad():
                if len(healthy_indices) > 0:
                    best_neuron = healthy_indices[fitness[healthy_indices].argmax()]
                    for neuron_idx in dying_indices:
                        layer.weight[neuron_idx, :] = layer.weight[best_neuron, :].clone()
                        if layer.bias is not None:
                            layer.bias[neuron_idx] = layer.bias[best_neuron].clone()

        else:
            raise ValueError(f"Unknown regrowth strategy: {self.regrowth_strategy}")

    def get_stats(self) -> Dict:
        """Get statistics about neuron ages and fitness."""
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


# ============================================================================
# Integration with Existing Training Loop
# ============================================================================

def integrate_neuron_apoptosis(model, trainer_class):
    """
    Example of how to integrate neuron apoptosis into training.
    """

    # Identify FFN layers in transformer blocks
    target_layers = []
    for i in range(6):  # 6 transformer blocks
        # FFN has two linear layers: block.ffn.0 (expand) and block.ffn.3 (contract)
        target_layers.append(f'blocks.{i}.ffn.0')  # First FFN layer (expands to 4*d_model)

    # Create apoptosis manager
    neuron_apoptosis = NeuronApoptosisManager(
        model=model,
        target_layers=target_layers,
        prune_rate=0.1,          # Prune 10% of neurons
        apoptosis_interval=500,   # Every 500 steps
        fitness_metric='grad_activation',  # Use gradient × activation
        regrowth_strategy='mutation',      # Evolutionary regrowth
        mutation_strength=0.3
    )

    return neuron_apoptosis


# ============================================================================
# Modified Trainer to Support Neuron Apoptosis
# ============================================================================

class NeuronApoptosisTrainer(Trainer):
    """Extended trainer that handles neuron-level apoptosis."""

    def __init__(self, model, neuron_apoptosis_mgr, **kwargs):
        super().__init__(model, apoptosis_mgr=None, **kwargs)  # No layer-level apoptosis
        self.neuron_apoptosis_mgr = neuron_apoptosis_mgr

    def train(self, num_steps: int, eval_interval: int = 100, save_interval: int = 500):
        """Training loop with neuron apoptosis."""
        print(f"\nStarting training with NEURON-LEVEL apoptosis...")
        print(f"Target layers: {self.neuron_apoptosis_mgr.target_layers}")
        print(f"Prune rate: {self.neuron_apoptosis_mgr.prune_rate*100:.0f}%")
        print(f"Interval: every {self.neuron_apoptosis_mgr.apoptosis_interval} steps")

        pbar = tqdm(total=num_steps, desc="Training")
        train_iter = iter(self.train_loader)

        for step in range(num_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Training step
            loss = self.train_step(x, y)

            # Update neuron apoptosis manager
            apoptosis_occurred = self.neuron_apoptosis_mgr.step()
            if apoptosis_occurred:
                self.save_checkpoint('neuron_apoptosis_event')

                # Print stats
                stats = self.neuron_apoptosis_mgr.get_stats()
                print("\nNeuron age statistics:")
                for layer, layer_stats in stats.items():
                    print(f"  {layer}: mean_age={layer_stats['mean_age']:.0f}, "
                          f"range=[{layer_stats['min_age']}, {layer_stats['max_age']}]")

            # Collect and log metrics
            metrics = self.collect_metrics(loss)
            self.metrics_history.append(metrics)
            self.log_metrics(metrics)

            # Evaluation
            if step % eval_interval == 0:
                val_loss, val_ppl = self.eval_step()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.writer.add_scalar('val/perplexity', val_ppl, self.global_step)
                pbar.set_postfix({'loss': f'{loss:.3f}', 'val_ppl': f'{val_ppl:.2f}'})

            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                self.save_checkpoint('checkpoint')

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print("\nTraining complete!")
        self.save_checkpoint('final')


print("✓ Neuron-level apoptosis system loaded!")
print("  Use: NeuronApoptosisManager + NeuronApoptosisTrainer")
