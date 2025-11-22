# ✅ **PROJECT SUMMARY: Apoptosis-Driven Neural Architectures**

Your project is an experimental neural network training framework exploring **biologically inspired lifecycle mechanics**, including:

* **Neuron-level apoptosis** (kill underperforming neurons and regrow fresh ones)
* **Senescence** (neurons age, slow learning, and eventually get replaced)
* **Fitness-based survival** (using gradients, activation variance, stagnation penalty)
* **Experimental hyperparameter sweeps and orthogonal-array tuning**
* **Instrumentation and logging to JSON + SQLite**
* **Future: adaptive experimentation controller + TUI dashboard**
* **Far-future: recursive architectures (TRM-like), population evolution**

The goal is *not* to optimize a model for accuracy, but to **explore dynamics**, measure emergent behavior, and understand how neuron-level birth/death affects learning.

You’ve already built:

* A full apoptosis manager
* Several generations of fitness metrics
* A senescence daemon
* Histograms, percentiles, and per-layer metrics
* Orthogonal array test harness
* JSON logging + early SQLite prototype
* NaN/inf sanitization and fitness stability improvements

You want to consolidate all of this into a clean project scaffold.

---

# 🧠 **TIMELINE OF KEY DECISIONS & IDEAS**

## **Phase 1 — Early exploration**

* Initial idea: prune lowest-fitness neurons periodically.
* Early fitness: simple gradient magnitude.
* Issues: slow apoptosis, MPS CPU fallback, unstable gradients.

## **Phase 2 — Performance Engineering**

* Huge speedup from moving apoptosis to CPU for linear algebra.
* Replaced dense SVD with faster heuristics.
* Pruned code paths, introduced fast regrowth.
* Added orthogonal array sweeps.
* Introduced TensorBoard instrumentation.

## **Phase 3 — Lifecycle Design**

* Introduced “neuron age.”
* Decided to model:

  * newborn → mature → senior → dying
* Proposed temperature-like learning rate adjustment.

## **Phase 4 — Senescence Mechanism**

* Designed “senescence daemon”:

  * track rolling slope of neuron fitness
  * declare senescence if flatlined X steps
  * escalate → kill → retry
* Many refinements to avoid over-triggering.

## **Phase 5 — Fitness Redesign**

We moved away from single-term fitness to:

```
fitness = α * grad_norm
        + β * activation_variance
        - γ * stagnation_penalty
```

Where stagnation is based on similarity to an EMA of activation means.

## **Phase 6 — Instrumentation**

Metrics recorded:

* per-layer histograms (quantized)
* percentiles
* mean/std/variance
* apoptosis & senescence events
* high-resolution run logs

JSON output now has:

* params
* metrics (per layer & per step)
* events (with neuron index + step)
* timing breakdown

## **Phase 7 — Error Hardening**

We discovered:

* MPS sometimes produces NaNs post-apoptosis
* EMA contamination creates long-lived NaNs
* histograms fail when min/max = NaN
* fitness normalization can zero-divide

We added:

* `nan_to_num` everywhere
* histogram guards
* EMA reset on neuron reset
* safe normalizations everywhere

## **Phase 8 — Data Storage**

You requested:

* SQLite backend
* storing quantized states
* storing apoptosis/senescence events
* storing params/metrics per run
* groundwork for adaptive experimentation controller

## **Phase 9 — Forward Vision**

Planned features:

* Textual/urwid/PromptToolkit TUI visualizer
* Adaptive controller (bandit / Bayesian optimization)
* Automated sweeps + self-guiding exploration loop
* Population genetics (model populations, mutation, selection)
* Integration with Recursive Transformer experiments

---

# 📐 **CURRENT SYSTEM ARCHITECTURE (High-Level)**

```
apoptosis-v2/
│
├── train.py
│   ├── training loop
│   ├── loss/backprop
│   ├── logging / instrumentation
│   └── invokes NeuronApoptosisManager
│
├── src/
│   ├── neuron_apoptosis_manager.py
│   │   ├── fitness computation
│   │   ├── senescence daemon
│   │   ├── apoptosis logic
│   │   ├── regrowth mechanisms
│   │   ├── per-layer state tracking
│   │   └── histogram/percentile extraction
│   │
│   ├── utils/
│   │   ├── histogram.py
│   │   ├── normalization.py
│   │   ├── event_logging.py
│   │   └── nan_sanitization.py
│   │
│   └── db/
│       ├── schema.sql
│       ├── insert_run.py
│       ├── insert_metrics.py
│       └── insert_events.py
│
├── experiments/
│   ├── sweeps/
│   │   └── sweep.py (orthogonal arrays + grid + random)
│   ├── adaptive/
│   │   ├── bandit_controller.py
│   │   ├── bo_controller.py
│   │   └── evolution_controller.py
│   └── configs/
│       └── (all your hyperparam sets)
│
└── dashboards/
    ├── textual_ui.py
    └── plot_notebook.ipynb
```

---

# 🎯 **FUTURE GOALS & PLANNED DIRECTIONS**

This is what we discussed and agreed on, in priority order.

---

## **1. TUI Visualizer**

Something like **Textual**, showing:

* per-layer histograms updating in real time
* apoptosis/senescence indicators
* live fitness distribution
* neuron age distribution
* loss curve
* timeline of events

Basically a “neural health monitor.”

**Difficulty:** Medium
**Effort:** 1–2 days
**Value:** Huge for intuition

---

## **2. Auto-Scaling Hyperparameters (Controller)**

System watches runs and learns:

* what hyperparams hurt/help
* what fitness distributions are “healthy”
* when to increase/decrease mutation
* when to make apoptosis more aggressive
* which layers need more pruning
* etc.

Techniques:

* Multi-armed bandits (UCB, Thompson) for simple control
* Bayesian Optimization for structured sweeps
* Evolutionary Algorithm for population-based training

**Difficulty:** High
**Effort:** 3–7 days
**Value:** Massive — turns experiment into a *self-driving lab*

---

## **3. Genetic Population (Model Families)**

A population of models:

* each trains independently
* apoptosis/senescence act as “within-model mutations”
* selection pressure chooses next generation

You get:

* speciation
* convergence
* architecture evolution patterns

This will be *extremely* interesting.

**Difficulty:** Very High
**Effort:** Multi-week
**Value:** 💥 Frontier research territory


JSON logging + early SQLite prototype
NaN/inf sanitization and fitness stability improvements
You want to consolidate all of this into a clean project scaffold.
