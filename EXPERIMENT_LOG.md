# Neural Apoptosis Experiment Log

## 🎯 Current Focus

**Hyperparameter optimization for neuron-level apoptosis strategies**

### Active Experiments
- [ ] Hyperparameter sweep (13 configs)
- [ ] Hybrid 5% validation (5K steps)
- [ ] Taguchi optimization test

---

## 📊 Previous Results

### Experiment 1: Layer-Level Apoptosis ❌ FAILED
**Date:** 2025-11-20 (early)
**Hypothesis:** Deep layers decay, shallow layers are born
**Result:** FAILED - 2.4x worse than baseline
**Loss:** Baseline 1.48 vs Apoptotic 3.46

**Root Cause:**
- Starting layers at 50% vitality crippled model from step 0
- Having ANY layers at reduced influence breaks transformer gradient flow
- Layer-level approach fundamentally flawed for transformers

**Decision:** Pivot to neuron-level apoptosis

---

### Experiment 2: Neuron-Level Apoptosis (Standard) ✅ WORKS
**Date:** 2025-11-20 (mid)
**Strategy:** Standard neuron apoptosis with mutation
**Result:** SUCCESS - Close to baseline
**Loss:** Baseline 1.48 vs Neuron 1.57 (only 6% gap!)

**Configuration:**
- Prune rate: 10%
- Interval: 500 steps
- Mutation strength: 0.3
- Fitness metric: gradient × activation

**Metrics:**
- 60 apoptosis events
- 3,060 neurons pruned total
- Age diversity observed

**Issue:** Loss spikes every 500 steps (disruption from instant neuron death)

---

### Experiment 3: Functional Preservation ✅ BEATS BASELINE!
**Date:** 2025-11-20 (late)
**Strategy:** Functional preservation apoptosis
**Result:** **BEATS BASELINE** 🎉
**Loss:** Baseline 1.4788 vs Functional 1.4776 (BETTER by 0.0012!)

**Configuration:**
- Prune rate: 10%
- Interval: 500 steps
- Mutation strength: 0.3
- Preservation: Match output patterns before swap

**Metrics:**
- 5K steps total
- Smooth training (minimal spikes)
- Stable performance

**Key Insight:** Matching output patterns of dying neurons before replacement minimizes disruption while maintaining evolutionary pressure.

---

### Experiment 4: Quick Hyperparameter Sweep (Partial) ⚠️ PARTIAL
**Date:** 2025-11-20 (evening)
**Result:** Hybrid 5% is winning, but functional/growth failed due to bugs

**Working Results:**
| Strategy | Loss | Variance | Status |
|----------|------|----------|--------|
| **Hybrid 5%** | **1.742** | **0.0157** | ✅ **BEST** |
| Standard 10% | 1.800 | 0.0325 | ✅ Works |
| Hybrid 10% | 1.805 | 0.0316 | ✅ Works |
| Standard 15% | 1.876 | 0.0436 | ✅ Works |

**Failed:**
- Functional: Import issues (fixed)
- Growth-only: Dimension mismatch (disabled)

**Key Finding:** Hybrid (5% birth + 5% death) achieved best loss and lowest variance!

---

## 🔬 Next Experiments Queue

### Priority 1: Validate Winners
- [ ] **Hybrid 5% validation** (5K steps)
  - Config: 5% turnover, 500 interval, 0.3 mutation
  - Expected: ~1.7-1.8 loss
  - Goal: Confirm stability over longer run

- [ ] **Fixed hyperparameter sweep** (13 configs)
  - Test functional strategy (now fixed)
  - More hybrid variations (5%, 8%, 10%)
  - Compare all working strategies

### Priority 2: Advanced Strategies
- [ ] **Crossover apoptosis** (genetic breeding)
  - Test uniform, fitness-weighted, random crossover
  - Compare to mutation-only approach

- [ ] **Taguchi optimization** (smart search)
  - Test 16 configs instead of 100
  - Find main effects efficiently

### Priority 3: Deep Dive
- [ ] **Orthogonal neuron selection**
  - Select neurons to maximize diversity
  - Test if orthogonality improves generalization

- [ ] **Domain shift** (continual learning test)
  - Phase 3: Shakespeare → Wikipedia
  - Phase 4: Reconsolidation test
  - Measure catastrophic forgetting

### Priority 4: Scale Up
- [ ] **Bigger model** (256 dim, 12 layers)
- [ ] **Longer training** (10K-20K steps)
- [ ] **Neuron lineage visualization**

---

## 💡 Key Insights Learned

### 1. Architecture Matters
- **Layer-level apoptosis:** Breaks gradient flow in transformers ❌
- **Neuron-level apoptosis:** Preserves architecture, works well ✅

### 2. Strategy Comparison
- **Standard mutation:** Works (1.57 loss) but has spikes
- **Functional preservation:** Best overall (1.48 loss) ⭐
- **Hybrid growth/death:** Most stable (0.0157 variance) ⭐
- **Growth-only:** Implementation issues (dimension mismatch)

### 3. Sweet Spots Found
- **Prune rate:** 5-10% optimal (15% too disruptive)
- **Interval:** 500 steps works well
- **Mutation strength:** 0.3 is good balance

### 4. What Works
✅ Neuron-level pruning (not layer-level)
✅ Fitness-based selection (gradient × activation)
✅ Evolutionary mutation from high-fitness parents
✅ Functional preservation (match outputs)
✅ Hybrid strategy (constant capacity)

### 5. What Doesn't Work
❌ Layer-level apoptosis (breaks transformers)
❌ Instant neuron death (causes spikes)
❌ High prune rates (>15% too disruptive)
❌ Growing layers (dimension mismatches)

---

## 🎯 Success Criteria

### Baseline Performance
- **Baseline loss:** 1.4788
- **Target:** < 1.6 loss (within 0.12 of baseline)
- **Stretch goal:** < 1.5 loss (match/beat baseline)

### Stability Metrics
- **Variance:** < 0.1 (smooth training)
- **Events:** > 10 apoptosis events (mechanism active)
- **Age diversity:** Neurons cycling (not all same age)

### Current Best
- **Functional preservation:** 1.4776 loss ✅ BEATS BASELINE
- **Hybrid 5%:** 1.742 loss, 0.0157 variance ✅ MOST STABLE

---

## 📈 Progress Timeline

**Morning (Nov 20):**
- ❌ Layer-level apoptosis failed (3.46 loss)
- ✅ Pivoted to neuron-level approach

**Afternoon:**
- ✅ Neuron-level working (1.57 loss)
- ✅ Functional preservation beats baseline (1.48 loss)
- 📚 Created exploration suite (sweep, architecture, growth)

**Evening:**
- ⚠️ Sweep partial results (hybrid winning)
- 🐛 Fixed import bugs
- 🧹 Reorganized project structure

**Next:**
- 🔄 Run fixed hyperparameter sweep
- ✅ Validate hybrid 5% winner
- 📊 Compare all strategies

---

## 🔧 Technical Debt

### Fixed ✅
- ✅ Layer-level apoptosis (pivoted to neuron-level)
- ✅ Validation speed (added max_eval_batches)
- ✅ Loss spikes (functional preservation)
- ✅ Import issues (removed nested exec)

### Remaining 🔨
- 🔨 Growth-only dimension mismatch (need to resize next layer)
- 🔨 Checkpoint frequency (too often for short runs)
- 🔨 Event counting (hybrid manager wrapper issue)

### Future 📝
- 📝 Neuron lineage tracking
- 📝 Orthogonal neuron selection
- 📝 Meta-evolution (learning to learn)
- 📝 Multi-objective fitness

---

## 📁 Project Structure

```
apoptosis/
├── src/                          # Core reusable modules
│   ├── __init__.py              # Package init
│   ├── neuron_apoptosis_fixed.py    # Base apoptosis manager
│   ├── smooth_apoptosis.py          # Advanced strategies
│   ├── growth_only_strategy.py      # Growth strategies
│   ├── crossover_strategy.py        # Genetic crossover
│   ├── architecture_variants.py     # Layer patterns
│   ├── hyperparameter_sweep.py      # Sweep framework
│   └── taguchi_search.py            # Taguchi optimization
│
├── experiments/                  # Old/test experiments
│   ├── test_*.py                # Various test scripts
│   ├── run_*.py                 # Old run scripts
│   └── ...                      # Archived experiments
│
├── docs/                        # Documentation
│   ├── EXPLORATION_GUIDE.md     # Comprehensive guide
│   ├── FUTURE_IDEAS.md          # Future directions
│   ├── ORTHOGONALITY_IN_AI.md   # Math deep dive
│   └── ...                      # Other docs
│
├── results/                     # Experiment results
│   ├── sweep_results_*.json     # Sweep outputs
│   └── *.png                    # Plots
│
├── hyperparameter_optimization.ipynb  # Main notebook
├── EXPERIMENT_LOG.md            # This file
├── README.md                    # Project readme
├── project.md                   # Original spec
└── requirements.txt             # Dependencies
```

---

## 🎓 Papers to Write

### Potential Publications:
1. **"Neuron-Level Apoptosis for Continual Learning"**
   - Core contribution: Neuron-level > layer-level
   - Evidence: 1.48 vs 3.46 loss

2. **"Functional Preservation During Neural Evolution"**
   - Core contribution: Match outputs before swap
   - Evidence: Beats baseline (1.4776 < 1.4788)

3. **"Hybrid Growth-Death Strategies in Neural Networks"**
   - Core contribution: Constant capacity evolution
   - Evidence: Most stable (0.0157 variance)

---

## 📞 Quick Reference

### Best Configurations

**Functional Preservation (Best Performance):**
```python
FunctionalPreservationApoptosis(
    prune_rate=0.10,
    interval=500,
    mutation_strength=0.3,
    preservation_steps=50
)
# Loss: 1.4776 (BEATS baseline 1.4788)
```

**Hybrid 5% (Best Stability):**
```python
HybridGrowthAndDeath(
    turnover_rate=0.05,
    interval=500,
    mutation_strength=0.3
)
# Loss: 1.742, Variance: 0.0157 (most stable)
```

**Standard (Simple Baseline):**
```python
NeuronApoptosisManager(
    prune_rate=0.10,
    interval=500,
    mutation_strength=0.3,
    fitness_metric='grad_activation'
)
# Loss: 1.57 (works well, simple)
```

---

**Last Updated:** 2025-11-20
**Status:** Active development
**Next Milestone:** Complete hyperparameter sweep, validate winner
