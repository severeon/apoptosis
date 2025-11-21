# Neuron-Level vs Layer-Level Apoptosis

## Why Layer-Level Failed (Your Results)

**Problem**: Having entire layers at 25-50% influence **crippled the model**
- Baseline: 1.48 loss
- Layer-level: 3.46 loss (2.4x worse!)

**Root cause**: Too aggressive
- 2 out of 6 layers at reduced capacity = 33% of network handicapped
- Transformers are tightly coupled - weak layers break gradient flow
- Like trying to run a race with tied legs

---

## Why Neuron-Level Should Work Better

### 1. **Granular Control**
Instead of killing 1/6 of the network (1 full layer), you kill 1/512 of the network (1 neuron).

**Layer-level**: Remove 170K parameters at once (1 layer)
**Neuron-level**: Remove 128 parameters at once (1 neuron)

**Impact**: 1000x less disruptive!

### 2. **Continuous Operation**
Layers have discrete states (alive or dead).
Neurons can have a gradual population - some die, some are born, most are healthy.

**Layer-level**: Binary (layer is 0% or 100%)
**Neuron-level**: Smooth (network is 90-100% functional)

### 3. **Biologically Accurate**
In real brains:
- Individual neurons die/are born constantly
- Entire cortical columns don't disappear

### 4. **Preserves Architecture**
Layer-level requires modifying the forward pass (influence scaling).
Neuron-level just sets some weights to zero - architecturally identical to baseline.

---

## How Neuron-Level Works

### Step 1: Compute Fitness
Every 500 steps, for each neuron in target layers:

```python
fitness = |gradient| × |activation|
```

This measures: "How much is this neuron contributing to learning?"

### Step 2: Prune Weakest
Find bottom 10% by fitness, zero their weights:

```python
threshold = quantile(fitness, 0.1)
dying_neurons = fitness < threshold
layer.weight[dying_neurons, :] = 0  # Kill them
```

### Step 3: Regrow from Strong
Sample a healthy neuron (weighted by fitness), copy and mutate:

```python
parent = sample(healthy_neurons, weights=fitness)
child_weights = parent_weights + gaussian_noise(σ=0.3)
```

This is **evolutionary algorithm inside the network**!

---

## Expected Performance

### Hypothesis
Neuron-level should be **competitive with baseline** because:

1. **Minimal disruption**: Only 10% of neurons affected per event
2. **Network stays functional**: Still 90%+ capacity at all times
3. **Evolutionary benefit**: Weak neurons replaced by mutated strong ones
4. **No architectural changes**: Same forward pass as baseline

### Prediction
- **Baseline**: 1.48 loss
- **Layer-level**: 3.46 loss (failed - 2.4x worse)
- **Neuron-level**: 1.5-1.7 loss (competitive!)

If neuron-level gets within 0.2 of baseline, the approach works!

---

## Fitness Metrics Deep Dive

### Why Gradient × Activation?

**Gradient alone**: Tells you how much the loss wants to change this neuron
**Activation alone**: Tells you how much this neuron fires
**Gradient × Activation**: Taylor approximation of "how much would loss change if we removed this neuron"

```python
ΔLoss ≈ gradient × activation  # First-order Taylor expansion
```

This is the **cheapest approximation of importance** without actually ablating neurons.

### Alternative: Pure Gradient
Faster but noisier:
```python
fitness = |gradient|
```

Good for: Quick experiments, very frequent apoptosis

### Alternative: Weight Magnitude
Slowest but most stable:
```python
fitness = ||weights||₂
```

Good for: Infrequent apoptosis, pruning before deployment

### Alternative: Composite
Best accuracy but more compute:
```python
fitness = 0.4 × ||weights|| + 0.4 × |gradient| + 0.2 × |activation|
```

Good for: When you want the most accurate fitness

---

## Tuning Knobs (In Order of Importance)

### 1. **Prune Rate** (Most Important)
- 5%: Very conservative (minimal disruption)
- 10%: Moderate (recommended)
- 20%: Aggressive (risky but more genetic diversity)

**Rule**: Start at 10%, increase only if performance is good.

### 2. **Apoptosis Interval**
- 1000 steps: Infrequent (slow evolution)
- 500 steps: Moderate (recommended)
- 250 steps: Frequent (high churn)

**Rule**: Longer interval = more stable, shorter = more exploration

### 3. **Fitness Metric**
- `'gradient'`: Fastest (but noisy)
- `'grad_activation'`: Best balance (recommended)
- `'composite'`: Most accurate (but slower)

**Rule**: Use `grad_activation` unless you have a specific reason not to.

### 4. **Regrowth Strategy**
- `'random'`: No inheritance (like random mutation)
- `'mutation'`: Evolutionary (inherit from strong, add noise) [recommended]
- `'clone'`: Copy best neuron (no diversity)

**Rule**: Mutation gives best of both worlds.

### 5. **Mutation Strength**
- 0.1: Weak mutation (minor variations)
- 0.3: Moderate (recommended)
- 0.5: Strong (high diversity but risky)

**Rule**: Higher = more exploration, lower = more exploitation

---

## Target Layers

### FFN Layers (Recommended)
```python
target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]
```

**Why**: FFN is fully connected, easier to prune neurons independently.

### Attention Layers (Advanced)
```python
target_layers = [f'blocks.{i}.attention.out_proj' for i in range(6)]
```

**Why**: Attention is more coupled, but might be interesting for "attention head apoptosis"

### Both (Aggressive)
```python
target_layers = [f'blocks.{i}.ffn.0' for i in range(6)] + \
                [f'blocks.{i}.attention.out_proj' for i in range(6)]
```

**Why**: Maximum evolutionary pressure across entire network

---

## Success Criteria

### Minimum Viable
- Loss within 0.5 of baseline
- Apoptosis events occur (10+ in 5K steps)
- No NaN/Inf

### Interesting
- Loss within 0.2 of baseline
- Clear fitness diversity (weak neurons being pruned)
- Neuron ages vary (some old, some young)

### Publication-Worthy
- Loss within 0.1 of baseline OR better
- Faster adaptation on domain shift
- Clear evolutionary patterns in neuron lineages

---

## Quick Start

1. **Paste `neuron_apoptosis.py` into notebook** (defines classes)
2. **Paste `test_neuron_apoptosis.py` into next cell** (runs experiment)
3. **Run and watch** (should take ~30 min on M2 Max)
4. **Check results**:
   - If loss ≈ baseline → SUCCESS! 🎉
   - If loss 1.5-2.0 → Promising, tune parameters
   - If loss > 2.0 → Something wrong, check logs

---

## Debugging

### If loss is still way worse:
1. Check fitness values - are they all similar? (bad)
2. Check neuron ages - are any neurons dying? (need to see events)
3. Try higher prune rate (20%) to see more dramatic changes
4. Try `fitness_metric='weight'` (more stable)

### If no apoptosis events:
1. Lower interval (250 instead of 500)
2. Check that hooks are firing (print activations)
3. Verify gradients exist (print grad norms)

### If loss is NaN:
1. Lower mutation strength (0.2 → 0.1)
2. Add gradient clipping (already in code)
3. Check for dead layers (all neurons in one layer pruned)

---

## Next Steps After Success

If neuron-level works:

1. **Test domain shift** (Phase 3-4 from original plan)
2. **Visualize neuron lineages** (who inherited from whom)
3. **Try different fitness metrics** (compare results)
4. **Ablation studies** (which component matters most)
5. **Scale up** (more layers, bigger model)

---

## The Key Insight

**Layer-level**: "Remove the engine and hope the car still drives"
**Neuron-level**: "Replace a few spark plugs while the engine is running"

One is catastrophic, the other is maintenance. 🚗⚡
