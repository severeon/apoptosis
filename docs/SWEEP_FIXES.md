# Hyperparameter Sweep - Bugs Found & Fixed

## 🐛 Bugs Discovered in First Sweep

### 1. **Functional Strategy: Wrong Class** ❌
**Error:** `NeuronApoptosisManager.__init__() got an unexpected keyword argument 'preservation_steps'`

**Problem:** The create_manager() function was trying to instantiate `NeuronApoptosisManager` for functional strategy, but should use `FunctionalPreservationApoptosis`.

**Fix:**
```python
elif config['strategy'] == 'functional':
    # Import the correct class
    from smooth_apoptosis import FunctionalPreservationApoptosis
    return FunctionalPreservationApoptosis(
        model=model,
        target_layers=target_layers,
        prune_rate=config['prune_rate'],
        apoptosis_interval=config['interval'],
        mutation_strength=config['mutation_strength'],
        preservation_steps=config.get('preservation_steps', 50)
    )
```

---

### 2. **Growth-Only: Dimension Mismatch** ❌
**Error:** `linear(): input and weight.T shapes cannot be multiplied (128x563 and 512x128)`

**Problem:** When GrowthOnlyManager grows `ffn.0` from 512→563 neurons, the next layer (`ffn.2`) still expects 512 inputs but receives 563.

**Root cause:** Growing layers changes architecture dynamically, breaking subsequent layers.

**Fix:** Disabled growth-only strategy for now. Proper fix requires:
```python
# When growing layer i:
# 1. Expand layer i output: 512 → 563
# 2. Update layer i+1 input: 512 → 563 (expand input weights)
# 3. Maintain consistency across network
```

This is non-trivial and requires careful architecture manipulation.

**Status:** Commented out for now, marked as TODO.

---

### 3. **Hybrid: Event Count Shows 0** ⚠️
**Error:** Not actually an error, but `num_events: 0` reported despite hybrid working.

**Problem:** Hybrid manager wraps `NeuronApoptosisManager`, so events are in `manager.apoptosis_mgr.apoptosis_events`, not `manager.apoptosis_events`.

**Fix:**
```python
# Get stats from manager
if hasattr(manager, 'apoptosis_events'):
    num_events = len(manager.apoptosis_events)
elif hasattr(manager, 'growth_events'):
    num_events = len(manager.growth_events)
elif hasattr(manager, 'apoptosis_mgr'):
    # Hybrid manager wraps another manager
    num_events = len(manager.apoptosis_mgr.apoptosis_events)
else:
    num_events = 0
```

---

### 4. **Checkpoint Frequency Too High** ⚠️
**Problem:** Saving checkpoints every 500 steps during 2K step runs is overkill and slows things down.

**Fix:** Removed checkpoint saving from quick sweep experiments (test_quick_sweep_fixed.py).

---

## 📊 Results from First Sweep (Before Fixes)

| Config | Loss | Variance | Status |
|--------|------|----------|--------|
| **Hybrid 5%** | **1.742** | **0.0157** | ✅ **WINNER** |
| Standard 10% | 1.800 | 0.0325 | ✅ Works |
| Hybrid 10% | 1.805 | 0.0316 | ✅ Works |
| Standard 15% | 1.876 | 0.0436 | ✅ Works |
| Functional 10% | - | - | ❌ Wrong class |
| Functional 15% | - | - | ❌ Wrong class |
| Growth 10% | - | - | ❌ Dimension mismatch |
| Growth 15% | - | - | ❌ Dimension mismatch |

---

## ✅ What Was Fixed

1. ✓ **Functional strategy** now uses correct class with proper imports
2. ✓ **Hybrid event counting** now correctly accesses wrapped manager
3. ✓ **Growth-only disabled** with clear TODO comment
4. ✓ **Checkpoint frequency** removed from quick experiments
5. ✓ **QuickSweep updated** to test more hybrid configs (winning strategy!)

---

## 🚀 New Test Scripts

### 1. `test_quick_sweep_fixed.py`
Fixed sweep with 13 configs:
- 2 Standard (10%, 15%)
- 2 Functional (10%, 15%)
- 9 Hybrid (5%, 8%, 10% × 3 intervals)

**Why more hybrid configs?** Because hybrid is winning!

**Estimated time:** ~65 minutes

---

### 2. `test_hybrid_winner.py`
Validation run for best config from sweep:
- Strategy: Hybrid 5% turnover
- Interval: 500 steps
- Full 5K step training
- Compare to baseline (1.4788)

**Estimated time:** ~30 minutes

---

## 🎯 Key Insight: Hybrid is Winning!

**Hybrid 5% achieved:**
- ✅ Best loss: 1.742
- ✅ Best stability: 0.0157 variance
- ✅ Smooth training (no huge spikes)

**Why hybrid works:**
1. **Constant capacity** - No architectural changes
2. **Evolutionary pressure** - Weak neurons replaced by strong
3. **Balanced turnover** - 5% birth + 5% death = 10% churn
4. **Simple** - One parameter to tune (turnover rate)

---

## 📈 Next Steps

### Immediate:
1. Run `test_quick_sweep_fixed.py` to test fixed functional strategy
2. Run `test_hybrid_winner.py` to validate winner with 5K steps
3. Compare to baseline (1.4788 loss)

### If Hybrid 5% validates well:
1. Try domain shift (Shakespeare → Wikipedia)
2. Test on larger model (256 dim, 12 layers)
3. Visualize neuron lineages over time
4. Write up results!

### If you want to fix growth-only:
1. Implement dynamic architecture resizing
2. Update subsequent layer input dimensions when growing
3. Handle residual connection dimension mismatches
4. Test carefully (this is complex!)

---

## 🔧 How to Run Fixed Sweep

```python
# In Jupyter notebook:

# 1. Load base dependencies (as usual)
exec(open('neuron_apoptosis_fixed.py').read())
exec(open('smooth_apoptosis.py').read())
exec(open('growth_only_strategy.py').read())

# 2. Run fixed sweep
exec(open('test_quick_sweep_fixed.py').read())

# Wait ~65 minutes...

# 3. Check results
# Results saved to sweep_results_TIMESTAMP.json
# Top configs printed at end
```

---

## 🎓 Lessons Learned

1. **Wrapper classes need careful introspection** (hybrid event counting)
2. **Dynamic architecture is hard** (growth-only dimension issues)
3. **Import order matters** (functional class not imported)
4. **Checkpoint frequency should scale with run length**
5. **Early results guide later experiments** (focus on hybrid!)

---

## 📊 Expected Results After Fixes

With functional strategy working, we should see:
- Functional 10%: ~1.7-1.8 loss (based on previous 5K run: 1.4776)
- Functional 15%: ~1.8-1.9 loss (more disruption)

If functional beats hybrid:
- Use functional for production
- It's proven to match/beat baseline

If hybrid still wins:
- Hybrid is simpler (one strategy, one param)
- Easier to tune and understand
- Recommend hybrid 5% as default

---

## 🏆 Current Champion

**Hybrid Growth and Death (5% turnover)**
- Loss: 1.742
- Variance: 0.0157
- Interval: 500 steps
- Mechanism: Birth rate = Death rate (steady state)

**Why it's special:**
- No architectural changes (unlike growth-only)
- No preservation overhead (unlike functional)
- Just pure evolutionary selection
- Simple and elegant! 🧬

---

## TODO (Future Work)

- [ ] Fix growth-only dimension issues
- [ ] Test Taguchi search (faster hyperparameter search)
- [ ] Implement orthogonal neuron selection
- [ ] Add crossover strategy to sweep
- [ ] Visualize neuron lineages
- [ ] Domain shift experiments
- [ ] Scale to bigger models

---

**Ready to run the fixed sweep!** 🚀
