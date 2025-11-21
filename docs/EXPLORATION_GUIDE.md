# Neural Apoptosis Exploration Guide

## What We Just Created

You now have **3 powerful exploration tools** to find the optimal apoptosis configuration:

---

## 1. Hyperparameter Sweep (`hyperparameter_sweep.py`)

### What it does:
Tests hundreds of configurations across all strategies to find the sweet spot.

### Strategies tested:
- **Standard Neuron Apoptosis** (48 configs)
  - Prune rates: 5%, 10%, 15%, 20%
  - Intervals: 250, 500, 750 steps
  - Mutation strengths: 0.1, 0.2, 0.3, 0.4

- **Functional Preservation** (27 configs)
  - Prune rates: 5%, 10%, 15%
  - Intervals: 250, 500, 750
  - Mutation strengths: 0.1, 0.2, 0.3

- **Gradual Fade** (4 configs)
  - Prune rates: 10%, 15%
  - Fade durations: 25, 50 steps

- **Growth-Only** (18 configs)
  - Growth rates: 5%, 10%, 15%
  - Intervals: 500, 750, 1000
  - Max capacities: 130%, 150%, 170%

- **Hybrid** (6 configs)
  - Turnover rates: 5%, 10%, 15%
  - Intervals: 250, 500, 750

**Total: ~100 configurations**

### How to use:

```python
# Full sweep (may take 8-10 hours)
exec(open('hyperparameter_sweep.py').read())

sweep = HyperparameterSweep(
    model_class=ApoptoticTransformer,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    device=device,
    num_steps=2000  # Quick tests
)
sweep.run_sweep()
```

**Or use the quick version (~1 hour):**

```python
# Just paste this into a cell!
exec(open('test_quick_sweep.py').read())
```

### Output:
- **JSON file** with all results (`sweep_results_TIMESTAMP.json`)
- **Top 10 by loss** (best performance)
- **Top 10 by stability** (lowest variance)
- **Best config per strategy**

---

## 2. Architecture Variants (`architecture_variants.py`)

### What it does:
Tests different layer selection patterns - which layers should have apoptosis?

### Patterns tested:

1. **standard** - All 6 FFN layers (current approach)
2. **deep_only** - Layers 3-5 only (death zone)
3. **shallow_only** - Layers 0-2 only (birth zone)
4. **alternating** - Layers 0, 2, 4 (every other)
5. **middle_only** - Layers 2-3 (stable core cycling)
6. **attention_only** - Target attention projections
7. **attention_shallow** - Attention in layers 0-2
8. **attention_deep** - Attention in layers 3-5
9. **ffn_and_attention** - BOTH FFN and attention (max coverage)
10. **shallow_ffn_deep_attention** - Mixed approach
11. **graduated** - **Variable turnover rates!**
    - Layers 0-1: 15% turnover (high exploration)
    - Layers 2-3: 10% turnover (moderate)
    - Layers 4-5: 5% turnover (stability)

### How to use:

```python
exec(open('architecture_variants.py').read())

# Quick test of 5 key patterns (~30-40 min)
results = quick_architecture_test()
```

**Or test all 11 patterns:**

```python
exp = ArchitectureExperiment(
    model=None,
    device=device,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset
)
results = exp.run_all_patterns(num_steps=2000)
```

### Why this matters:
- **Shallow layers** learn input features (may need more turnover)
- **Deep layers** learn output representations (may need stability)
- **Graduated approach** mimics biological systems (cortex vs hippocampus)

---

## 3. Growth-Only Strategy (`growth_only_strategy.py`)

### What it does:
Explores neurogenesis WITHOUT death - just keep adding neurons!

### Two strategies:

#### A. GrowthOnlyManager
- Add 10% new neurons every 500 steps
- Never remove old neurons
- Cap at 150% original capacity
- New neurons = mutations of high-fitness parents

**Timeline example:**
```
Step 0:    512 neurons (start)
Step 500:  563 neurons (+51, 10% growth)
Step 1000: 614 neurons (+51)
Step 1500: 665 neurons (+51)
Step 2000: 716 neurons (+51)
Step 2500: 768 neurons (hit cap)
After:     768 neurons (no more growth)
```

#### B. HybridGrowthAndDeath
- Add 5% new neurons every 500 steps
- Remove 5% weak neurons every 500 steps
- **Net effect**: Constant size, 10% turnover
- Like "steady-state" evolution

### How to use:

```python
exec(open('growth_only_strategy.py').read())

# Growth-only (no death)
growth_mgr = GrowthOnlyManager(
    model=neuron_model,
    target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
    growth_rate=0.10,
    growth_interval=500,
    max_capacity=1.5,
    mutation_strength=0.3
)

# OR Hybrid (birth = death)
hybrid_mgr = HybridGrowthAndDeath(
    model=neuron_model,
    target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
    turnover_rate=0.05,
    interval=500,
    mutation_strength=0.3
)
```

### Hypotheses to test:
1. **Growth-only** preserves all old knowledge (no forgetting)
2. **Hybrid** maintains capacity but with evolutionary pressure
3. Growth-only may show **less variance** (no disruption spikes)

---

## Recommended Exploration Path

### Phase 1: Quick Sweep (1 hour)
```python
exec(open('test_quick_sweep.py').read())
```
**Goal**: Find which strategy family works best (Standard vs Functional vs Growth)

### Phase 2: Architecture Test (30 min)
```python
exec(open('architecture_variants.py').read())
quick_architecture_test()
```
**Goal**: Find which layers benefit most from apoptosis

### Phase 3: Growth Comparison (30 min)
Test growth-only vs current best:
```python
# Run growth-only for 2000 steps
# Compare to functional preservation baseline
```

### Phase 4: Full Sweep (optional, 8-10 hours)
If you have time and want the absolute best config:
```python
sweep = HyperparameterSweep(...)
sweep.run_sweep()
```

---

## What to Look For

### Success Metrics:
1. **Final loss < 1.7** (competitive with baseline ~1.48)
2. **Variance < 0.1** (stable training)
3. **10+ apoptosis events** (mechanism is active)
4. **Age diversity** (neurons cycling, not all same age)

### Red Flags:
- Loss > 2.0 (too disruptive)
- Variance > 0.2 (unstable)
- 0 events (mechanism not working)
- NaN/Inf (numerical instability)

---

## Expected Runtimes (M2 Max)

| Experiment | Configs | Time/Config | Total Time |
|------------|---------|-------------|------------|
| Quick Sweep | ~12 | 5 min | **~1 hour** |
| Architecture Test | 5 | 6 min | **~30 min** |
| Full Sweep | ~100 | 5 min | **~8 hours** |
| Single 5K run | 1 | 30 min | **30 min** |

---

## Interpreting Results

### From Hyperparameter Sweep:

**Top 10 by Loss:**
- Shows which configs achieve best performance
- Look for patterns (e.g., all top configs use mutation_strength=0.3)

**Top 10 by Stability:**
- Shows which configs train smoothly
- May not have best loss but easier to work with

**Best per Strategy:**
- Direct comparison: which strategy family wins?
- Example: "Functional is 0.05 better than Standard"

### From Architecture Variants:

**Pattern comparison:**
- **If deep_only wins**: Output layers need more evolution
- **If shallow_only wins**: Input features need more exploration
- **If graduated wins**: Variable turnover rates match biological systems!

### From Growth-Only:

**If growth-only beats standard:**
- Death is harmful (disruption > benefit)
- Consider hybrid or slower turnover

**If hybrid beats both:**
- Sweet spot: some evolution, minimal disruption
- Constant capacity preferred

---

## Quick Decision Tree

```
Run Quick Sweep
    ↓
Which strategy won?
    ↓
┌───────────┬───────────────┬────────────┐
│ Standard  │ Functional    │ Growth     │
│ or Hybrid │ Preservation  │ Only       │
└─────┬─────┴───────┬───────┴─────┬──────┘
      ↓             ↓             ↓
  Tune prune   Tune preservation  Tune capacity
  rate/interval   steps/mutation   and growth rate
      ↓             ↓             ↓
  Run Architecture Variants
      ↓
  Use best pattern + best strategy
      ↓
  Full 5K step validation run
```

---

## Next Steps After Finding Sweet Spot

Once you find a config that works well:

1. **Validate with longer run** (5K-10K steps)
2. **Test domain shift** (Shakespeare → Wikipedia)
3. **Visualize neuron lineages** (who inherited from whom)
4. **Ablation studies** (remove components to see what matters)
5. **Scale up** (bigger model, more layers)

---

## Tips for Overnight Runs

If running full sweep overnight:

```python
# Add error handling
try:
    sweep.run_sweep()
except Exception as e:
    print(f"FAILED: {e}")
    sweep.save_results()  # Save partial results
```

Check progress:
```python
# In a new cell
import json
with open('sweep_results_TIMESTAMP.json') as f:
    results = json.load(f)
print(f"Completed: {len(results)} configs")
```

---

## The Big Question We're Answering

**"What is the optimal way to implement neural apoptosis for continual learning?"**

These tools will tell you:
- ✓ Which strategy (death vs growth vs both)
- ✓ Which layers (shallow vs deep vs all)
- ✓ Which hyperparameters (prune rate, interval, mutation)
- ✓ Trade-offs (performance vs stability vs simplicity)

**Let the experiments run, then we'll have data-driven answers!** 🧬🔬
