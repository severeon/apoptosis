# Neuron & Layer Fitness Metrics

## The Fundamental Question
**"Which neurons/layers are contributing to learning, and which are dead weight?"**

---

## 1. Gradient-Based Fitness (Most Direct)

### Gradient Magnitude
**Idea**: Neurons with large gradients are actively learning.

```python
fitness = torch.mean(torch.abs(param.grad))
```

**Pros**:
- Direct measure of learning activity
- Easy to compute (already calculated during backprop)
- Works for any parameter

**Cons**:
- Noisy (varies batch to batch)
- Can be artificially inflated by bad initialization

**When to use**: Real-time fitness during training

---

## 2. Activation-Based Fitness

### Activation Magnitude
**Idea**: Dead neurons (always ≈0) aren't contributing.

```python
# During forward pass, collect activations
activations = []  # Shape: (batch, neurons)

# Compute fitness
fitness = torch.mean(torch.abs(activations))
```

**Pros**:
- Catches truly dead neurons
- Stable across batches

**Cons**:
- Doesn't distinguish between "important constant" and "useless"

### Activation Variance
**Idea**: Low variance = always same response = redundant.

```python
fitness = torch.var(activations, dim=0)  # Per-neuron variance
```

**Pros**:
- Identifies redundant neurons
- More informative than just magnitude

**Cons**:
- Neurons with high variance might just be noisy

### Activation Entropy
**Idea**: High entropy = diverse responses = more useful.

```python
# Discretize activations into bins
bins = torch.histc(activations, bins=20)
probs = bins / bins.sum()
entropy = -torch.sum(probs * torch.log(probs + 1e-8))
fitness = entropy
```

**Pros**:
- Information-theoretic measure
- Captures "informational value"

**Cons**:
- Expensive to compute
- Requires binning/discretization

---

## 3. Weight-Based Fitness

### Weight Magnitude (L2 Norm)
**Idea**: Small weights = low influence on output.

```python
fitness = torch.norm(weight_matrix[:, neuron_idx])  # Outgoing weights
# OR
fitness = torch.norm(weight_matrix[neuron_idx, :])  # Incoming weights
```

**Pros**:
- No forward/backward pass needed
- Stable metric
- Used in pruning literature (magnitude pruning)

**Cons**:
- Doesn't account for activation patterns
- Small weights might be important for fine-tuning

### Weight Change Rate
**Idea**: Neurons whose weights change rapidly are learning.

```python
# Track weight over time
delta_weight = weight_t - weight_t_prev
fitness = torch.norm(delta_weight)
```

**Pros**:
- Measures learning speed
- Catches neurons still adapting

**Cons**:
- Requires storing historical weights
- Can be high even if learning the wrong thing

---

## 4. Contribution-Based Fitness (Most Expensive)

### Ablation Score
**Idea**: Remove neuron, see how much loss increases.

```python
# For each neuron:
original_loss = compute_loss(model, batch)

# Zero out neuron
with torch.no_grad():
    save_weights = neuron_weights.clone()
    neuron_weights.zero_()

ablated_loss = compute_loss(model, batch)
fitness = ablated_loss - original_loss  # Higher = more important

# Restore
neuron_weights.copy_(save_weights)
```

**Pros**:
- Direct measure of importance
- Gold standard for pruning

**Cons**:
- VERY expensive (N forward passes for N neurons)
- Only practical for small networks

### Gradient × Activation (Taylor Expansion)
**Idea**: First-order approximation of ablation.

```python
# From Taylor expansion: ΔLoss ≈ gradient * activation
fitness = torch.abs(gradient * activation)
```

**Pros**:
- Cheap approximation of ablation
- Used in state-of-art pruning (e.g., SNIP)

**Cons**:
- Only first-order approximation
- Can be noisy

---

## 5. Task-Specific Fitness

### Output Correlation
**Idea**: Neurons whose activations correlate with correct predictions are useful.

```python
# For classification:
correct_mask = (predictions == labels)
activations_when_correct = activations[correct_mask].mean()
activations_when_wrong = activations[~correct_mask].mean()
fitness = activations_when_correct - activations_when_wrong
```

**Pros**:
- Task-aware
- Identifies neurons that "know" the right answer

**Cons**:
- Requires labels
- Doesn't work for generative tasks

---

## Recommended Combo: **Gradient-Magnitude + Weight-Magnitude**

For practical neuron-level apoptosis, use:

```python
def compute_neuron_fitness(layer, activations=None):
    """
    Composite fitness score combining multiple signals.

    Returns:
        fitness: (num_neurons,) tensor of fitness scores
    """
    # 1. Weight magnitude (static)
    weight_fitness = torch.norm(layer.weight, dim=0)  # Per-output-neuron

    # 2. Gradient magnitude (dynamic)
    if layer.weight.grad is not None:
        grad_fitness = torch.norm(layer.weight.grad, dim=0)
    else:
        grad_fitness = torch.zeros_like(weight_fitness)

    # 3. Activation statistics (if available)
    if activations is not None:
        activation_fitness = torch.abs(activations).mean(dim=0)
    else:
        activation_fitness = torch.ones_like(weight_fitness)

    # Combine (weighted average)
    fitness = (
        0.4 * weight_fitness +      # Static importance
        0.4 * grad_fitness +         # Learning activity
        0.2 * activation_fitness     # Current contribution
    )

    return fitness
```

---

## Layer-Level Fitness

For layer-level apoptosis (your original approach):

```python
def compute_layer_fitness(layer):
    """Aggregate fitness of all neurons in a layer."""

    # Option 1: Mean fitness
    neuron_fitness = compute_neuron_fitness(layer)
    return neuron_fitness.mean()

    # Option 2: Max fitness (layer is as good as best neuron)
    return neuron_fitness.max()

    # Option 3: Variance (high variance = diverse = healthy)
    return neuron_fitness.var()
```

---

## Quick Reference Table

| Metric | Compute Cost | Stability | Biological Plausibility | Best For |
|--------|-------------|-----------|------------------------|----------|
| Gradient magnitude | Low | Low | Medium | Real-time pruning |
| Activation variance | Medium | High | High | Finding redundancy |
| Weight magnitude | Very Low | Very High | Low | Static pruning |
| Ablation score | Very High | High | High | One-shot pruning |
| Gradient × Activation | Low | Medium | Medium | Practical compromise |

---

## My Recommendation for Your Experiment

Use **Gradient × Activation** (Taylor approximation):
- Fast enough to compute every step
- More informative than gradient alone
- Approximates the "true" importance
- Biologically plausible (neurons that are active during learning are important)

```python
# During forward pass, save activations
# During backward pass, compute fitness
fitness = torch.abs(grad * activation).mean(dim=0)  # Per-neuron

# Prune bottom 10%
threshold = torch.quantile(fitness, 0.1)
dying_neurons = fitness < threshold
```

This gives you the best bang-for-buck in terms of computation vs. accuracy.
