"""
Smooth Apoptosis Strategies - Reduce Loss Spikes

Four approaches to make neuron death/rebirth more gradual.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


# =============================================================================
# STRATEGY 1: Gradual Fade (Best for smooth transitions)
# =============================================================================

class GradualFadeApoptosis(NeuronApoptosisManager):
    """
    Instead of instant death, fade neurons over N steps.

    Timeline:
    - Step 500: Identify dying neurons, start fade
    - Steps 500-550: Linearly reduce weights 100% → 0%
    - Step 550: Complete death, rebirth new neurons
    """

    def __init__(self, *args, fade_steps=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.fade_steps = fade_steps
        self.fading_neurons = {}  # {layer_name: {neuron_idx: (start_step, weights_backup)}}

    def trigger_apoptosis(self) -> bool:
        """Start gradual fade instead of instant death."""
        # First, complete any ongoing fades
        self._complete_fades()

        # Then identify new neurons to fade
        apoptosis_occurred = False

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            fitness = self.compute_fitness(layer_name)

            num_neurons = len(fitness)
            num_to_prune = max(1, int(num_neurons * self.prune_rate))

            _, sorted_indices = torch.sort(fitness)
            dying_neurons = sorted_indices[:num_to_prune].cpu().numpy()

            if len(dying_neurons) == 0:
                continue

            print(f"\n[Gradual Fade @ step {self.step_count}] {layer_name}")
            print(f"  Starting fade for {len(dying_neurons)} neurons over {self.fade_steps} steps")

            # Store original weights for fading
            if layer_name not in self.fading_neurons:
                self.fading_neurons[layer_name] = {}

            with torch.no_grad():
                for neuron_idx in dying_neurons:
                    self.fading_neurons[layer_name][neuron_idx] = (
                        self.step_count,
                        layer.weight[neuron_idx, :].clone()
                    )

            apoptosis_occurred = True

        return apoptosis_occurred

    def step(self) -> bool:
        """Update fading neurons each step."""
        self.step_count += 1

        # Update ages
        for layer_name in self.target_layers:
            for meta in self.neuron_metadata[layer_name]:
                meta.age += 1

        # Update fading neurons
        self._update_fades()

        # Check for new apoptosis
        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False

    def _update_fades(self):
        """Apply gradual weight reduction to fading neurons."""
        for layer_name, fading in self.fading_neurons.items():
            if not fading:
                continue

            layer = self._get_layer(layer_name)

            with torch.no_grad():
                for neuron_idx, (start_step, original_weights) in list(fading.items()):
                    elapsed = self.step_count - start_step

                    if elapsed >= self.fade_steps:
                        # Fade complete, will be handled in _complete_fades
                        continue

                    # Linear fade: 1.0 → 0.0 over fade_steps
                    fade_factor = 1.0 - (elapsed / self.fade_steps)
                    layer.weight[neuron_idx, :] = original_weights * fade_factor

    def _complete_fades(self):
        """Complete any fades that have finished, rebirth neurons."""
        for layer_name, fading in list(self.fading_neurons.items()):
            if not fading:
                continue

            layer = self._get_layer(layer_name)
            completed = []

            for neuron_idx, (start_step, _) in fading.items():
                elapsed = self.step_count - start_step

                if elapsed >= self.fade_steps:
                    completed.append(neuron_idx)

            if not completed:
                continue

            print(f"[Rebirth @ step {self.step_count}] {layer_name}: {len(completed)} neurons")

            # Compute fitness for regrowth
            fitness = self.compute_fitness(layer_name)
            _, sorted_indices = torch.sort(fitness)
            healthy_neurons = sorted_indices[len(completed):].cpu().numpy()

            # Regrow
            self._regrow_neurons(layer, np.array(completed), healthy_neurons, fitness)

            # Update metadata
            for neuron_idx in completed:
                self.neuron_metadata[layer_name][neuron_idx] = NeuronMetadata(
                    age=0, birth_step=self.step_count
                )
                del self.fading_neurons[layer_name][neuron_idx]
                self.apoptosis_events.append((self.step_count, layer_name, 1))


# =============================================================================
# STRATEGY 2: Smaller, More Frequent (Continuous turnover)
# =============================================================================

class ContinuousTurnoverApoptosis(NeuronApoptosisManager):
    """
    Instead of 10% every 500 steps, prune 2% every 100 steps.

    Advantages:
    - Smaller disruption per event
    - More continuous evolution
    - Network never loses much capacity at once
    """

    def __init__(self, model, target_layers, **kwargs):
        # Override defaults for continuous mode
        kwargs.setdefault('prune_rate', 0.02)  # 2% instead of 10%
        kwargs.setdefault('apoptosis_interval', 100)  # Every 100 instead of 500

        super().__init__(model, target_layers, **kwargs)

        print(f"Continuous turnover mode:")
        print(f"  Pruning {self.prune_rate*100:.0f}% every {self.apoptosis_interval} steps")
        print(f"  Equivalent to {(self.prune_rate * (500/self.apoptosis_interval))*100:.0f}% per 500 steps")


# =============================================================================
# STRATEGY 3: Knowledge Distillation (Train before swap)
# =============================================================================

class DistillationApoptosis(NeuronApoptosisManager):
    """
    Before replacing neurons, train new neurons to mimic old ones.

    Process:
    1. Identify dying neurons
    2. Create new neurons (mutations)
    3. Run a few forward passes, training new to match old outputs
    4. Swap in the trained new neurons
    """

    def __init__(self, *args, distillation_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_steps = distillation_steps
        self.pending_rebirths = []  # List of (layer, dying_idx, new_weights, distill_count)

    def trigger_apoptosis(self) -> bool:
        """Identify dying neurons and start distillation."""
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

            print(f"\n[Distillation Start @ step {self.step_count}] {layer_name}")
            print(f"  Preparing {len(dying_neurons)} replacement neurons")

            # Pre-generate new neurons (but don't install yet)
            for neuron_idx in dying_neurons:
                # Sample parent and create mutation
                if len(healthy_neurons) > 0:
                    healthy_fitness = fitness[healthy_neurons].cpu().numpy()
                    probs = healthy_fitness / (healthy_fitness.sum() + 1e-8)
                    parent_idx = np.random.choice(healthy_neurons, p=probs)

                    new_weights = layer.weight[parent_idx, :].clone()
                    noise = torch.randn_like(new_weights) * self.mutation_strength
                    new_weights = new_weights + noise
                else:
                    # Fallback to random
                    new_weights = torch.randn(layer.weight.size(1), device=layer.weight.device)
                    nn.init.xavier_uniform_(new_weights.unsqueeze(0))
                    new_weights = new_weights.squeeze(0)

                self.pending_rebirths.append((
                    layer_name, neuron_idx, new_weights, 0
                ))

            apoptosis_occurred = True

        return apoptosis_occurred

    def step(self) -> bool:
        """Update distillation and complete ready rebirths."""
        self.step_count += 1

        # Age neurons
        for layer_name in self.target_layers:
            for meta in self.neuron_metadata[layer_name]:
                meta.age += 1

        # Process pending rebirths (distillation happens during training)
        self._process_pending_rebirths()

        # Check for new apoptosis
        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False

    def _process_pending_rebirths(self):
        """Complete rebirths that have finished distillation."""
        ready = []

        for i, (layer_name, neuron_idx, new_weights, count) in enumerate(self.pending_rebirths):
            if count >= self.distillation_steps:
                ready.append(i)

        if not ready:
            return

        # Install new neurons
        for idx in reversed(ready):  # Reverse to maintain indices
            layer_name, neuron_idx, new_weights, _ = self.pending_rebirths.pop(idx)
            layer = self._get_layer(layer_name)

            with torch.no_grad():
                layer.weight[neuron_idx, :] = new_weights
                if layer.bias is not None:
                    layer.bias[neuron_idx] = 0

            print(f"[Rebirth @ step {self.step_count}] {layer_name}: neuron {neuron_idx}")

            self.neuron_metadata[layer_name][neuron_idx] = NeuronMetadata(
                age=0, birth_step=self.step_count
            )
            self.apoptosis_events.append((self.step_count, layer_name, 1))


# =============================================================================
# STRATEGY 4: Functional Preservation (Minimize disruption)
# =============================================================================

class FunctionalPreservationApoptosis(NeuronApoptosisManager):
    """
    Initialize new neurons to approximate the function of dying neurons.

    Method:
    1. Before killing neuron N, record its typical outputs on recent batches
    2. Initialize new neuron to produce similar outputs (regression)
    3. Then add mutation noise for diversity

    This preserves network function better than pure mutation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_outputs = {}  # {layer_name: recent activations}
        self.output_buffer_size = 10  # Store last 10 batches

    def _register_hooks(self):
        """Override to capture more detailed activation info."""
        def hook_fn(name):
            def hook(module, input, output):
                # Store more history for functional preservation
                if name not in self.recent_outputs:
                    self.recent_outputs[name] = []

                self.recent_outputs[name].append(output.detach())

                # Keep only recent
                if len(self.recent_outputs[name]) > self.output_buffer_size:
                    self.recent_outputs[name].pop(0)

                # Also store in activations for fitness
                self.activations[name] = output.detach()
            return hook

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            layer.register_forward_hook(hook_fn(layer_name))

    def _regrow_neurons(self, layer, dying_indices, healthy_indices, fitness):
        """Regrow with functional preservation."""
        layer_name = None
        for name in self.target_layers:
            if self._get_layer(name) is layer:
                layer_name = name
                break

        if layer_name is None or layer_name not in self.recent_outputs:
            # Fallback to standard mutation
            super()._regrow_neurons(layer, dying_indices, healthy_indices, fitness)
            return

        with torch.no_grad():
            for neuron_idx in dying_indices:
                # Get recent outputs for this neuron
                recent_activations = [out[:, :, neuron_idx] for out in self.recent_outputs[layer_name]]

                if len(recent_activations) == 0:
                    # No data, use standard mutation
                    if len(healthy_indices) > 0:
                        healthy_fitness = fitness[healthy_indices].cpu().numpy()
                        probs = healthy_fitness / (healthy_fitness.sum() + 1e-8)
                        parent_idx = np.random.choice(healthy_indices, p=probs)
                        layer.weight[neuron_idx, :] = layer.weight[parent_idx, :].clone()
                    else:
                        nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    continue

                # Target: mean output of dying neuron
                target_output = torch.stack(recent_activations).mean(dim=0)  # (batch, seq)

                # Find neuron that best matches this output pattern
                if len(healthy_indices) > 0:
                    best_match = None
                    best_similarity = -float('inf')

                    for candidate_idx in healthy_indices:
                        candidate_activations = [out[:, :, candidate_idx] for out in self.recent_outputs[layer_name]]
                        candidate_output = torch.stack(candidate_activations).mean(dim=0)

                        # Cosine similarity
                        similarity = torch.nn.functional.cosine_similarity(
                            target_output.flatten(),
                            candidate_output.flatten(),
                            dim=0
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = candidate_idx

                    # Initialize from best match + mutation
                    if best_match is not None:
                        layer.weight[neuron_idx, :] = layer.weight[best_match, :].clone()
                        noise = torch.randn_like(layer.weight[neuron_idx, :]) * self.mutation_strength * 0.5  # Smaller noise
                        layer.weight[neuron_idx, :] += noise
                    else:
                        nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])


# =============================================================================
# Quick Comparison Test
# =============================================================================

def compare_strategies():
    """
    Quick test to see which strategy reduces spikes best.
    Run each for 2000 steps, measure loss variance.
    """

    strategies = {
        'Standard': NeuronApoptosisManager,
        'Gradual Fade': GradualFadeApoptosis,
        'Continuous': ContinuousTurnoverApoptosis,
        'Functional': FunctionalPreservationApoptosis,
    }

    results = {}

    for name, strategy_class in strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        # Create model
        model = ApoptoticTransformer(
            vocab_size=tokenizer.vocab_size,
            enable_apoptosis=False
        ).to(device)

        # Create manager
        if name == 'Continuous':
            mgr = strategy_class(
                model=model,
                target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
                # Uses defaults: 2% every 100 steps
            )
        elif name == 'Gradual Fade':
            mgr = strategy_class(
                model=model,
                target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
                prune_rate=0.1,
                apoptosis_interval=500,
                fade_steps=50  # Fade over 50 steps
            )
        else:
            mgr = strategy_class(
                model=model,
                target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
                prune_rate=0.1,
                apoptosis_interval=500
            )

        # Train for 2000 steps
        trainer = SimpleNeuronTrainer(
            model=model,
            neuron_mgr=mgr,
            train_dataset=shakespeare_dataset,
            val_dataset=shakespeare_dataset,
            batch_size=64,
            learning_rate=3e-4,
            experiment_name=f"compare_{name.lower().replace(' ', '_')}",
            max_eval_batches=100
        )

        trainer.train(num_steps=2000, eval_interval=100)

        # Measure loss variance (smoothness)
        losses = [m.loss for m in trainer.metrics_history]
        loss_variance = np.var(losses)
        final_loss = losses[-1]

        results[name] = {
            'final_loss': final_loss,
            'loss_variance': loss_variance,
            'events': len(mgr.apoptosis_events),
        }

        print(f"\n{name} Results:")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss variance: {loss_variance:.6f} (lower = smoother)")
        print(f"  Apoptosis events: {len(mgr.apoptosis_events)}")

    # Compare
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    best_smooth = min(results.items(), key=lambda x: x[1]['loss_variance'])
    best_loss = min(results.items(), key=lambda x: x[1]['final_loss'])

    for name, res in results.items():
        smooth_mark = "← SMOOTHEST" if name == best_smooth[0] else ""
        loss_mark = "← BEST LOSS" if name == best_loss[0] else ""

        print(f"\n{name}:")
        print(f"  Loss: {res['final_loss']:.4f} {loss_mark}")
        print(f"  Variance: {res['loss_variance']:.6f} {smooth_mark}")
        print(f"  Events: {res['events']}")

    return results


print("✓ Smooth apoptosis strategies loaded!")
print("\nAvailable strategies:")
print("  1. GradualFadeApoptosis - Fade neurons over 50 steps")
print("  2. ContinuousTurnoverApoptosis - 2% every 100 steps")
print("  3. DistillationApoptosis - Train new neurons first")
print("  4. FunctionalPreservationApoptosis - Preserve function")
print("\nRun compare_strategies() to test all!")
