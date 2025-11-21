# Quick Start Guide - Neural Apoptosis Exploration

## 🎯 What You Have Now

**3 new exploration systems** to find optimal apoptosis configurations:

1. **Hyperparameter Sweep** - Tests ~100 configs across all strategies
2. **Architecture Variants** - Tests which layers benefit from apoptosis
3. **Growth-Only Strategies** - Tests neurogenesis without death

---

## ⚡ Fastest Way to Start

### Option 1: Run Everything (2 hours)

```python
# Just paste this into a Jupyter cell!
exec(open('run_full_exploration.py').read())
```

**What it does:**
- Tests 12 hyperparameter configs (~1 hour)
- Tests 5 architecture patterns (~30 min)
- Tests 3 growth strategies (~30 min)
- Saves all results to JSON files
- **Prints best overall config at the end**

---

### Option 2: Quick Hyperparameter Test (1 hour)

```python
# Quick sweep of key configs
exec(open('test_quick_sweep.py').read())
```

**What you get:**
- Top 10 configs by loss
- Top 10 configs by stability
- Best config per strategy (Standard, Functional, Growth, etc.)

---

### Option 3: Just Test Growth-Only (30 min)

```python
# Load the strategy
exec(open('growth_only_strategy.py').read())

# Create model
model = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=False
).to(device)

# Create manager
growth_mgr = GrowthOnlyManager(
    model=model,
    target_layers=[f'blocks.{i}.ffn.0' for i in range(6)],
    growth_rate=0.10,
    growth_interval=500,
    max_capacity=1.5,
    mutation_strength=0.3
)

# Train with it (use your existing trainer code)
# Just call growth_mgr.step() in your training loop
```

---

## 📊 Understanding Results

### After hyperparameter sweep:

**Look for:**
- Loss < 1.7 (competitive with baseline 1.48)
- Variance < 0.1 (stable training)
- 10+ events (mechanism working)

**Best means:**
- Lowest final loss = best performance
- Lowest variance = smoothest training
- Good balance of both = production ready

### After architecture variants:

**If "graduated" wins:**
- Variable turnover rates work best
- Shallow layers need more turnover
- Deep layers need stability

**If "ffn_and_attention" wins:**
- Maximum coverage is beneficial
- Both components benefit from evolution

### After growth-only:

**If growth beats standard:**
- Death is too disruptive
- Pure neurogenesis is better
- Consider slower growth rate

**If hybrid wins:**
- Balanced evolution is optimal
- Some death acceptable if matched by birth

---

## 📁 Files Created

| File | Purpose | When to Use |
|------|---------|-------------|
| `growth_only_strategy.py` | Neurogenesis without death | Want to test "no death" hypothesis |
| `hyperparameter_sweep.py` | Systematic config testing | Want to find optimal hyperparams |
| `architecture_variants.py` | Layer pattern testing | Want to know which layers to target |
| `test_quick_sweep.py` | Quick hyperparameter test | Want fast results (~1 hour) |
| `run_full_exploration.py` | All-in-one runner | Want comprehensive results (~2 hours) |
| `EXPLORATION_GUIDE.md` | Detailed documentation | Want to understand everything |

---

## 🎮 Recommended Workflow

### Morning: Kick off exploration
```python
exec(open('run_full_exploration.py').read())
# Go get coffee, check back in 2 hours
```

### Afternoon: Review results
```python
import json

# Check what won
with open('exploration_results/hyperparameter_sweep_TIMESTAMP.json') as f:
    sweep = json.load(f)

# Find best
best = min(sweep, key=lambda x: x.get('final_train_loss', 999))
print(f"Best config: {best['config']}")
print(f"Loss: {best['final_train_loss']:.4f}")
```

### Evening: Validate winner
```python
# Run best config for full 5K steps
# Use exact params from best config
# Compare to baseline (1.48 loss)
```

---

## 🚨 Troubleshooting

### "NameError: tokenizer not defined"
```python
# Load the base experiment first
exec(open('apoptosis_experiment.ipynb'))
# Then run exploration
```

### "Out of memory"
- Reduce batch size to 32
- Reduce num_steps to 1000
- Run fewer configs at once

### "All results are bad (loss > 2.0)"
- Check baseline works first
- Verify gradients flowing (print grad norms)
- Try more conservative settings (prune_rate=0.05)

---

## 💡 What We're Testing

### Core Hypotheses:

1. **Neuron-level > Layer-level** ✓ (Already proven: 1.57 vs 3.46)

2. **Smooth transitions > Instant death** ✓ (Functional: 1.72 vs Standard: 1.76)

3. **Growth-only > Death+Birth** ❓ (Testing now!)

4. **Variable turnover > Uniform** ❓ (Testing with "graduated" pattern)

5. **Which layers matter most** ❓ (Testing with architecture variants)

---

## 🎯 Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Final Loss | < 2.0 | < 1.7 | < 1.5 |
| Variance | < 0.2 | < 0.1 | < 0.05 |
| Events | > 5 | > 10 | > 20 |
| Gap from baseline | < 0.5 | < 0.3 | < 0.1 |

**Baseline reference: 1.48 loss**

---

## 🔬 After Finding Sweet Spot

Once you find a config that works:

1. **Validate** - Run 5K-10K steps (30-60 min)
2. **Domain shift** - Test Shakespeare → Wikipedia
3. **Visualize** - Plot neuron lineages
4. **Ablation** - Remove components to see what matters
5. **Scale** - Try bigger model (256 dim, 12 layers)

---

## 💾 Expected Output Files

After running full exploration:

```
exploration_results/
├── hyperparameter_sweep_TIMESTAMP.json
├── architecture_variants_TIMESTAMP.json
└── growth_strategies_TIMESTAMP.json
```

Each JSON contains:
- All configs tested
- Final loss, variance, events
- Min/max loss during training
- Full config details

---

## ⏱️ Time Estimates (M2 Max)

| Task | Time | Worth It? |
|------|------|-----------|
| Quick sweep | 1 hr | ✓ Yes - fast feedback |
| Architecture test | 30 min | ✓ Yes - key insights |
| Growth test | 30 min | ✓ Yes - tests hypothesis |
| Full exploration | 2 hrs | ✓✓ Yes - comprehensive |
| Full sweep (100 configs) | 8 hrs | Maybe - overnight run |

---

## 🎉 The Goal

**Find a configuration where:**
- Loss ≈ 1.5 (close to baseline 1.48)
- Training is stable (low variance)
- Neurons are actively cycling (10+ events)
- Network maintains performance during turnover

**Then you have a working neural apoptosis system!** 🧬

---

## Quick Commands

```python
# Just want to see what's available?
!ls *.py | grep -E "(sweep|growth|arch)"

# Load and run quick test
exec(open('test_quick_sweep.py').read())

# Check if running
import os
os.system('ps aux | grep python')

# Kill if needed
# Ctrl+C in Jupyter, or find PID and kill
```

---

**Ready? Pick an option above and let's find that sweet spot!** 🚀
