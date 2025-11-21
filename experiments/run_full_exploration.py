"""
Full Exploration Suite - Run All Experiments

This runs all three exploration phases:
1. Quick Hyperparameter Sweep (~1 hour)
2. Architecture Variants Test (~30 min)
3. Growth-Only Strategy Test (~30 min)

Total runtime: ~2 hours

Results saved to JSON files with timestamps.
"""

import json
from datetime import datetime
import os

print("\n" + "="*70)
print("FULL NEURAL APOPTOSIS EXPLORATION SUITE")
print("="*70)
print("\nThis will run 3 phases:")
print("  1. Quick Hyperparameter Sweep (~12 configs, ~1 hour)")
print("  2. Architecture Variants (~5 patterns, ~30 min)")
print("  3. Growth-Only Tests (~3 configs, ~30 min)")
print("\nTotal time: ~2 hours")
print("="*70)

input("\nPress Enter to start, or Ctrl+C to cancel...")

# Create results directory
results_dir = "exploration_results"
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# PHASE 1: HYPERPARAMETER SWEEP
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: HYPERPARAMETER SWEEP")
print("="*70)

exec(open('hyperparameter_sweep.py').read())

quick_sweep = QuickSweep(
    model_class=ApoptoticTransformer,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    device=device,
    num_steps=2000,
    eval_interval=100
)

try:
    quick_sweep.run_sweep()

    # Save results
    sweep_results = quick_sweep.results
    with open(f"{results_dir}/hyperparameter_sweep_{timestamp}.json", 'w') as f:
        json.dump(sweep_results, f, indent=2)

    print("\n✓ Phase 1 complete! Results saved.")

except Exception as e:
    print(f"\n✗ Phase 1 failed: {e}")
    sweep_results = []

# ============================================================================
# PHASE 2: ARCHITECTURE VARIANTS
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: ARCHITECTURE VARIANTS")
print("="*70)

exec(open('architecture_variants.py').read())

exp = ArchitectureExperiment(
    model=None,
    device=device,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset
)

patterns_to_test = [
    'standard',
    'deep_only',
    'shallow_only',
    'graduated',
    'ffn_and_attention'
]

try:
    arch_results = []
    for pattern in patterns_to_test:
        result = exp.test_pattern(pattern, num_steps=2000)
        arch_results.append(result)

    exp.analyze_patterns(arch_results)

    # Save results
    with open(f"{results_dir}/architecture_variants_{timestamp}.json", 'w') as f:
        json.dump(arch_results, f, indent=2)

    print("\n✓ Phase 2 complete! Results saved.")

except Exception as e:
    print(f"\n✗ Phase 2 failed: {e}")
    arch_results = []

# ============================================================================
# PHASE 3: GROWTH-ONLY TESTS
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: GROWTH-ONLY STRATEGIES")
print("="*70)

exec(open('growth_only_strategy.py').read())

def test_growth_strategy(strategy_name, manager_class, config, num_steps=2000):
    """Test a growth strategy."""

    print(f"\nTesting: {strategy_name}")

    model = ApoptoticTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=6,
        max_seq_len=128,
        enable_apoptosis=False
    ).to(device)

    target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

    manager = manager_class(
        model=model,
        target_layers=target_layers,
        **config
    )

    # Training
    train_loader = torch.utils.data.DataLoader(
        shakespeare_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        manager.step()

        train_losses.append(loss.item())

    import numpy as np
    final_loss = np.mean(train_losses[-100:])
    variance = np.var(train_losses[-500:])

    if hasattr(manager, 'growth_events'):
        events = len(manager.growth_events)
    elif hasattr(manager, 'apoptosis_mgr'):
        events = len(manager.apoptosis_mgr.apoptosis_events)
    else:
        events = 0

    result = {
        'strategy': strategy_name,
        'config': config,
        'final_loss': final_loss,
        'variance': variance,
        'num_events': events,
        'min_loss': min(train_losses),
        'max_loss': max(train_losses)
    }

    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Variance: {variance:.6f}")
    print(f"  Events: {events}")

    del model
    del manager
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result

try:
    growth_results = []

    # Test 1: Standard growth-only
    result1 = test_growth_strategy(
        'growth_only_10pct',
        GrowthOnlyManager,
        {
            'growth_rate': 0.10,
            'growth_interval': 500,
            'max_capacity': 1.5,
            'mutation_strength': 0.3
        }
    )
    growth_results.append(result1)

    # Test 2: Aggressive growth
    result2 = test_growth_strategy(
        'growth_only_15pct',
        GrowthOnlyManager,
        {
            'growth_rate': 0.15,
            'growth_interval': 500,
            'max_capacity': 1.5,
            'mutation_strength': 0.3
        }
    )
    growth_results.append(result2)

    # Test 3: Hybrid (balanced)
    result3 = test_growth_strategy(
        'hybrid_5pct',
        HybridGrowthAndDeath,
        {
            'turnover_rate': 0.05,
            'interval': 500,
            'mutation_strength': 0.3
        }
    )
    growth_results.append(result3)

    # Analyze
    print("\n" + "="*70)
    print("GROWTH STRATEGY COMPARISON")
    print("="*70)

    for result in sorted(growth_results, key=lambda x: x['final_loss']):
        print(f"\n{result['strategy']}:")
        print(f"  Loss: {result['final_loss']:.4f}")
        print(f"  Variance: {result['variance']:.6f}")
        print(f"  Events: {result['num_events']}")

    # Save results
    with open(f"{results_dir}/growth_strategies_{timestamp}.json", 'w') as f:
        json.dump(growth_results, f, indent=2)

    print("\n✓ Phase 3 complete! Results saved.")

except Exception as e:
    print(f"\n✗ Phase 3 failed: {e}")
    growth_results = []

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EXPLORATION COMPLETE!")
print("="*70)

print(f"\n📊 Results Summary:")
print(f"  Phase 1: {len(sweep_results)} hyperparameter configs tested")
print(f"  Phase 2: {len(arch_results)} architecture patterns tested")
print(f"  Phase 3: {len(growth_results)} growth strategies tested")

print(f"\n📁 Results saved to: {results_dir}/")
print(f"  - hyperparameter_sweep_{timestamp}.json")
print(f"  - architecture_variants_{timestamp}.json")
print(f"  - growth_strategies_{timestamp}.json")

# Find overall best
all_results = []

for r in sweep_results:
    if 'error' not in r:
        all_results.append({
            'source': 'hyperparameter_sweep',
            'loss': r['final_train_loss'],
            'variance': r['train_loss_variance'],
            'config': r['config']
        })

for r in arch_results:
    if 'error' not in r:
        all_results.append({
            'source': 'architecture_variant',
            'loss': r['final_loss'],
            'variance': r['variance'],
            'config': {'pattern': r['pattern']}
        })

for r in growth_results:
    all_results.append({
        'source': 'growth_strategy',
        'loss': r['final_loss'],
        'variance': r['variance'],
        'config': r['config']
    })

if all_results:
    best = min(all_results, key=lambda x: x['loss'])

    print("\n" + "="*70)
    print("🏆 BEST OVERALL CONFIGURATION")
    print("="*70)
    print(f"\nSource: {best['source']}")
    print(f"Loss: {best['loss']:.4f}")
    print(f"Variance: {best['variance']:.6f}")
    print(f"\nConfig:")
    for k, v in best['config'].items():
        print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("\n✓ All exploration phases complete!")
    print("\nNext steps:")
    print("  1. Review JSON files for detailed results")
    print("  2. Run best config for full 5K steps")
    print("  3. Test on domain shift (Phase 3-4)")
    print("  4. Visualize neuron lineages")
    print("\n" + "="*70)

else:
    print("\n⚠️  No valid results collected. Check errors above.")
