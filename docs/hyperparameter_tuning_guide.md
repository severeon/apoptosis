# Hyperparameter Tuning Guide for Apoptosis

## The Problem: Too Many Knobs to Turn

You have:
- **Architectural choices**: How many stable vs dynamic layers?
- **Timing parameters**: Lifespan, interval, maturation
- **Scaling parameters**: Plasticity ceiling, mutation strength
- **Dynamics parameters**: Temperature, dropout ranges

## Strategy 1: Binary Search on Key Parameters

### Step 1: Find Minimum Viable Apoptosis Rate

**Goal**: Get at least 10-15 apoptosis events in 5000 steps

| Parameter | Conservative | Aggressive | Extreme |
|-----------|-------------|------------|---------|
| max_lifespan | 3000 | 1500 | 800 |
| apoptosis_interval | 500 | 250 | 100 |

**Quick test** (500 steps each):
```python
for lifespan in [3000, 2000, 1500, 1000, 800]:
    # Train 500 steps, count deaths
    # If deaths < 2, too slow
    # If deaths > 5, too fast
    # Sweet spot: 2-3 deaths per 500 steps
```

### Step 2: Find Optimal Plasticity Ceiling

**Goal**: Young layers contribute meaningfully without crippling the model

Test sequence (run each for 2000 steps):
1. `plasticity_ceiling=1.0` - No handicap (young = old)
2. `plasticity_ceiling=0.9` - Slight handicap
3. `plasticity_ceiling=0.75` - Moderate handicap
4. `plasticity_ceiling=0.5` - Current (might be too low)

**Decision rule**:
- If loss ~baseline at 1.0 → Can reduce ceiling
- If loss way worse at 0.9 → Problem isn't ceiling
- If loss good at 0.75 but bad at 0.5 → Ceiling matters

### Step 3: Tune Maturation Speed

**Goal**: Young layers reach full power quickly enough

| maturation_period | Interpretation |
|-------------------|----------------|
| 200 | Very fast (full power in 200 steps) |
| 500 | Moderate |
| 1000 | Slow (half the lifespan before mature) |

**Decision rule**:
- If `maturation_period > lifespan/2` → Too slow, layers die before maturing
- If `maturation_period < lifespan/5` → Maybe too fast, not much senescence

Optimal: `maturation_period ≈ lifespan / 3`

## Strategy 2: Grid Search (Automated)

```python
import itertools

# Define search space
lifespans = [800, 1200, 1600]
plasticity_ceilings = [0.7, 0.85, 1.0]
mutation_strengths = [0.2, 0.3, 0.4]

# Generate all combinations
configs = list(itertools.product(lifespans, plasticity_ceilings, mutation_strengths))

print(f"Total configs to try: {len(configs)}")  # 27 configs

results = []
for i, (lifespan, plasticity, mutation) in enumerate(configs):
    print(f"\nConfig {i+1}/{len(configs)}: lifespan={lifespan}, plasticity={plasticity}, mutation={mutation}")

    # Create model and train
    model = ApoptoticTransformer(...).to(device)
    mgr = ApoptosisManager(
        model=model,
        max_lifespan=lifespan,
        maturation_period=lifespan // 3,  # Auto-scale
        apoptosis_interval=lifespan // 6,  # Auto-scale
        plasticity_ceiling=plasticity,
        mutation_strength=mutation
    )

    trainer = Trainer(model, mgr, ...)
    trainer.train(num_steps=2000, eval_interval=200)  # Quick 2K step test

    final_loss = trainer.metrics_history[-1].loss
    events = len(mgr.apoptosis_events)

    results.append({
        'lifespan': lifespan,
        'plasticity': plasticity,
        'mutation': mutation,
        'loss': final_loss,
        'events': events,
        'gap': final_loss - baseline_loss
    })

    print(f"  Loss: {final_loss:.3f}, Events: {events}, Gap: {final_loss - baseline_loss:+.3f}")

# Find best config
best = min(results, key=lambda x: x['loss'])
print(f"\n🏆 Best config: {best}")
```

## Strategy 3: Architecture Search

### Question: How Many Stable vs Dynamic Layers?

Test these architectures (6 layers total):

| Config | Stable | Birth | Death | Rationale |
|--------|--------|-------|-------|-----------|
| A | 4 | 1 | 1 | Conservative (current broken: 2-2-2) |
| B | 2 | 2 | 2 | Balanced (your current attempt) |
| C | 0 | 3 | 3 | Aggressive (all layers cycle) |
| D | 5 | 0 | 1 | Minimal (only output layer dies) |

**Quick test**:
```python
architectures = [
    {'stable': [1,2,3,4], 'birth': [0], 'death': [5]},      # Config A
    {'stable': [2,3], 'birth': [0,1], 'death': [4,5]},      # Config B (current)
    {'stable': [], 'birth': [0,1,2], 'death': [3,4,5]},     # Config C
    {'stable': [0,1,2,3,4], 'birth': [], 'death': [5]},     # Config D
]

for i, arch in enumerate(architectures):
    print(f"\nArchitecture {chr(65+i)}:")

    model = ApoptoticTransformer(...).to(device)

    # Assign zones
    for layer_idx in range(6):
        if layer_idx in arch['stable']:
            zone = 'stable'
        elif layer_idx in arch['birth']:
            zone = 'birth'
        else:
            zone = 'death'

        model.senescence[layer_idx].layer_zone = zone

    # Train and compare
    # ...
```

## Strategy 4: Ablation Study (What Matters Most?)

Test each mechanism in isolation:

| Experiment | What's Enabled | What's Disabled |
|------------|----------------|-----------------|
| E1 | Apoptosis only | No temperature, no dropout scaling |
| E2 | Temperature only | No apoptosis, no dropout scaling |
| E3 | Dropout scaling only | No apoptosis, no temperature |
| E4 | Everything | Full system |

**Find out**:
- Is apoptosis helping or hurting?
- Is temperature modulation helping?
- Which mechanism has the biggest impact?

## Quick Diagnostic Table

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Loss 2-3x worse | Too many weak layers | Reduce dynamic layers (Config A) |
| Loss 10%+ worse | Influence scaling wrong | Try dropout method (Exp 4) |
| Only 1-2 deaths | Lifespan too long | Reduce to 800-1000 |
| 50+ deaths | Lifespan too short | Increase to 1500-2000 |
| Loss spikes >1.0 | Mutation too strong | Reduce to 0.1-0.2 |
| No improvement after deaths | Mutation too weak | Increase to 0.4-0.5 |
| Birth layers never contribute | Plasticity too low | Increase to 0.9-1.0 |
| Stable from the start | All parameters wrong | Run diagnostic experiments first |

## Recommended Path Forward

Based on your results (2.4x worse, only 2 events):

### Phase 1: Diagnose Root Cause (1-2 hours)
Run `diagnostic_experiments.py` to find if the problem is:
- Influence scaling itself
- Number of dynamic layers
- Or something else

### Phase 2: Architecture Search (2-3 hours)
Try configs A, C, D to find optimal layer distribution

### Phase 3: Hyperparameter Sweep (3-4 hours)
Once you have a working architecture, tune:
1. Lifespan (binary search to get 10-15 events)
2. Plasticity (try 0.9, 0.95, 1.0)
3. Mutation (try 0.3, 0.4, 0.5)

### Phase 4: Full Run (if Phase 1-3 show promise)
Train best config for full 13K steps across all phases

## Expected Outcomes

**If diagnostics show influence scaling is the problem:**
→ Switch to dropout-based senescence (Experiment 4)
→ Or try weight pruning instead

**If diagnostics show architecture is the problem:**
→ Use Config A (1 death, 1 birth, 4 stable)
→ Or Config D (only output layer dies)

**If everything still fails:**
→ The hypothesis might be wrong for this architecture
→ Consider neuron-level instead of layer-level
→ Or try a different network type (CNN, ResNet with skip connections)

## Time Budget

| Approach | Time | Risk | Potential Payoff |
|----------|------|------|------------------|
| Run all diagnostics | 4 hrs | Low | High (finds root cause) |
| Grid search | 8 hrs | Med | Medium (might miss best config) |
| Manual tuning | 2 hrs | High | Low (might get lucky) |
| Give up and try different approach | 1 hr | Low | Unknown |

**My recommendation**: Run diagnostics first. The 4 experiments will tell you if this approach can ever work.
