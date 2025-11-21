# Neural Layer Apoptosis Experiment - Technical Plan

## Objective
Explore architectural plasticity through layer-level apoptosis and neurogenesis, where deep (output-adjacent) layers decay while shallow (input-adjacent) layers are born, modulated by per-layer senescence metadata.

## Core Hypothesis
Networks can maintain or improve performance during continual learning by selectively replacing high-influence (deep) layers with low-influence (shallow) layers, using senescence-weighted parameter dynamics to balance exploration and exploitation.

---

## 1. Architecture

**Model Type**: Character-level autoregressive transformer (decoder-only)

**Why**:
- Autoregressive aligns with your mental model (recurrent processing)
- Layer roles are well-understood (early = syntax, late = semantics/prediction)
- Fast to train, easy to evaluate
- Naturally suited for continual learning experiments

**Specification**:
```python
- Vocabulary: ~100 chars (ASCII subset)
- Embedding dim: 128
- Layers: 6 total (allows 2-layer birth/death while maintaining 4 stable core)
- Heads: 4
- Context window: 128 tokens
- Parameters: ~500K (trains in minutes on CPU, seconds on GPU)
```

---

## 2. Apoptosis & Neurogenesis Mechanics

### Layer Lifecycle

**Senescence Metadata per Layer**:
```python
{
    "age": int,              # steps since birth
    "vitality": float,       # [0.0, 1.0], decays for deep layers
    "influence_weight": float,  # [0.0, 1.0], scales layer output
    "temperature": float,    # exploration parameter
}
```

**Death Zone**: Layers 5-6 (closest to output)
**Birth Zone**: Layers 1-2 (closest to input)
**Stable Core**: Layers 3-4 (never die)

### Decay Schedule

**Deep Layer Vitality Decay**:
```python
vitality = max(0.0, 1.0 - (age / max_lifespan))
influence_weight = vitality  # direct coupling
```

**Shallow Layer Vitality Growth**:
```python
vitality = min(1.0, age / maturation_period)
influence_weight = vitality * plasticity_ceiling  # caps initial influence
```

### Apoptosis Trigger
When layer vitality < 0.1:
1. Remove layer from computation graph
2. Initialize new layer in birth zone
3. Shift remaining layers (layer 5 becomes layer 6, etc.)

**Apoptosis Frequency**: Every `apoptosis_interval` steps (suggest: 500-1000)

---

## 3. Initialization Strategy

### New Layer Initialization

**Option A - Evolutionary Mutation** (RECOMMENDED):
```python
# Sample a living layer, mutate its weights
parent_layer = random.choice([layer for layer in model if layer.vitality > 0.5])
child_weights = parent_layer.weights + gaussian_noise(mean=0, std=mutation_strength)
```

**Option B - Random + Knowledge Distillation**:
```python
# Random init, but train for K steps on parent's outputs before integration
```

**Option C - Pure Random**:
```python
# Standard Xavier/He initialization
```

Start with Option A for warm-starting, fall back to C if too slow.

---

## 4. Senescence-Weighted Parameter Dynamics

### Temperature Modulation

**Attention Temperature** (per layer):
```python
temp = base_temp + (1.0 - vitality) * temp_range
# Young layers: high temp (1.5-2.0), explore widely
# Old layers: low temp (0.5-1.0), exploit learned patterns
```

**Dropout Rate** (per layer):
```python
dropout = base_dropout + vitality * dropout_range
# Young layers: high dropout (0.3-0.5), prevent premature crystallization
# Old layers: low dropout (0.1-0.2), preserve learned features
```

### Learning Rate Scaling

**Per-layer learning rate**:
```python
layer_lr = base_lr * (0.5 + 0.5 * vitality)
# Young layers learn faster to catch up
# Old layers learn slower to preserve knowledge
```

---

## 5. Training Setup

### Dataset

**Primary**: TinyShakespeare (~1MB, 40K lines)
**Secondary**: Python code corpus (for domain shift)

### Training Protocol

**Phase 1 - Baseline** (No Apoptosis):
- Train standard 6-layer model for 5K steps
- Record: loss curve, perplexity

**Phase 2 - Single Domain with Apoptosis**:
- Train apoptotic model on Shakespeare for 5K steps
- Apoptosis every 500 steps (expect 10 deaths/births)
- Record: loss curve, perplexity, layer vitality, layer influence

**Phase 3 - Domain Shift**:
- Continue training on Python code for 2K steps
- Record: adaptation speed (loss change rate), forgetting (test on Shakespeare)

**Phase 4 - Return to Original**:
- Switch back to Shakespeare for 1K steps
- Record: reconsolidation speed

---

## 6. Metrics & Evaluation

### Per-Step Metrics
```python
{
    "loss": float,
    "perplexity": float,
    "layer_vitalities": List[float],  # per layer
    "layer_influences": List[float],   # per layer
    "effective_params": int,           # sum of (params * influence_weight)
    "gradient_norms": List[float],     # per layer
}
```

### Aggregate Metrics

1. **Graceful Degradation Score**:
   ```python
   correlation(layer_death_events, loss_spikes)
   # Lower is better (deaths don't cause catastrophic loss)
   ```

2. **Adaptation Speed**:
   ```python
   steps_to_convergence_after_shift = argmin(|loss[t] - loss[t-100]| < threshold)
   ```

3. **Forgetting Coefficient**:
   ```python
   forgetting = (baseline_perplexity_A - post_shift_perplexity_A) / baseline_perplexity_A
   ```

4. **Parameter Efficiency**:
   ```python
   efficiency = baseline_loss / (apoptotic_loss * (effective_params / total_params))
   ```

### Visualization Requirements

- **Heatmap**: Layer vitality over time
- **Line plot**: Loss + apoptosis events (vertical lines)
- **Scatter**: Gradient norm vs. vitality (per layer, per step)
- **Animation**: Layer "influence sphere" growing/shrinking over time

---

## 7. Hyperparameters to Tune

### Critical
- `max_lifespan`: 2000-5000 steps
- `maturation_period`: 500-1000 steps
- `apoptosis_interval`: 500-1000 steps
- `plasticity_ceiling`: 0.3-0.7 (caps young layer influence)

### Secondary
- `base_temp`: 1.0
- `temp_range`: 0.5-1.0
- `mutation_strength`: 0.1-0.3 (for evolutionary init)
- `base_dropout`: 0.1
- `dropout_range`: 0.2-0.4

### Grid Search Suggestion
Start with conservative values (long lifespan, low plasticity ceiling), tighten once stable.

---

## 8. Success Criteria

**Minimum Viable**:
- ✓ Model trains without NaN
- ✓ Apoptosis events occur without catastrophic loss spikes
- ✓ Young layers develop non-zero gradients

**Interesting**:
- ✓ Apoptotic model matches baseline final loss
- ✓ Adaptation speed > baseline OR forgetting < baseline

**Publication-Worthy**:
- ✓ Apoptotic model outperforms baseline on domain shift
- ✓ Parameter efficiency > 1.0 (better performance per active parameter)
- ✓ Clear senescence-influence correlation in visualizations

---

## 9. Implementation Checklist

```python
[ ] Base transformer implementation (decoder-only)
[ ] Senescence metadata tracking (per layer)
[ ] Layer influence scaling mechanism (multiply layer output by influence_weight)
[ ] Temperature modulation in attention
[ ] Apoptosis trigger logic
[ ] Layer initialization (evolutionary mutation)
[ ] Training loop with metric collection
[ ] Baseline comparison run
[ ] Domain shift experimental protocol
[ ] Visualization suite (vitality heatmap, loss plot, gradient analysis)
```

---

## 10. Open Questions & Experiments

1. **Does position matter?** Try killing shallow layers instead—does the model behave differently?

2. **Gradual vs. sudden death?** Current plan is binary (alive/dead). Could linearly fade influence to zero over N steps.

3. **Senescence randomization?** Initialize new layers with random ages [0, max_age/2] to stagger deaths.

4. **Group apoptosis?** Kill/birth multiple layers simultaneously—does it help or hurt?

5. **Senescence inheritance?** New layers inherit partial senescence from parent (evolutionary continuity).

6. **Attention-based vitality?** Compute vitality from attention patterns (low attention = low vitality).

---

## Notes for Implementation

- Start simple: whole-layer apoptosis before node-level
- Log everything: you'll want to replay this in NACE later
- Save checkpoints before/after each apoptosis event
- Consider `torch.compile()` if using PyTorch 2.0+ for speed
- Notebook structure: Define classes/functions in early cells, run experiments in later cells for easy re-execution

**Estimated implementation time**: 4-6 hours for MVP, 8-12 hours for full experimental suite.
