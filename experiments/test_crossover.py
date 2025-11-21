"""
Test Crossover vs Mutation Strategy

Quick comparison (2000 steps each, ~10 min per test):
1. Mutation (current approach)
2. Uniform Crossover (50/50 blend)
3. Fitness-Weighted Crossover (stronger parent dominates)
4. Random Crossover (random ratio)
"""

# Load strategies
exec(open('neuron_apoptosis_fixed.py').read())
exec(open('crossover_strategy.py').read())

print("\n" + "="*70)
print("MUTATION vs CROSSOVER COMPARISON")
print("="*70)

results = []

# Test 1: Mutation (baseline)
print("\n[1/4] Testing MUTATION (current approach)...")
mutation_result = ComparisonRunner.test_strategy(
    strategy_class='mutation',
    config={
        'prune_rate': 0.10,
        'apoptosis_interval': 500,
        'fitness_metric': 'grad_activation',
        'regrowth_strategy': 'mutation',
        'mutation_strength': 0.3
    },
    model_class=ApoptoticTransformer,
    dataset=shakespeare_dataset,
    device=device,
    num_steps=2000
)
mutation_result['strategy'] = 'Mutation (baseline)'
results.append(mutation_result)

# Test 2: Uniform Crossover
print("\n[2/4] Testing UNIFORM CROSSOVER (50/50 blend)...")
uniform_result = ComparisonRunner.test_strategy(
    strategy_class='crossover',
    config={
        'prune_rate': 0.10,
        'apoptosis_interval': 500,
        'crossover_mode': 'uniform',
        'mutation_strength': 0.1  # Smaller noise
    },
    model_class=ApoptoticTransformer,
    dataset=shakespeare_dataset,
    device=device,
    num_steps=2000
)
uniform_result['strategy'] = 'Uniform Crossover'
results.append(uniform_result)

# Test 3: Fitness-Weighted Crossover
print("\n[3/4] Testing FITNESS-WEIGHTED CROSSOVER...")
fitness_weighted_result = ComparisonRunner.test_strategy(
    strategy_class='crossover',
    config={
        'prune_rate': 0.10,
        'apoptosis_interval': 500,
        'crossover_mode': 'fitness_weighted',
        'mutation_strength': 0.1
    },
    model_class=ApoptoticTransformer,
    dataset=shakespeare_dataset,
    device=device,
    num_steps=2000
)
fitness_weighted_result['strategy'] = 'Fitness-Weighted Crossover'
results.append(fitness_weighted_result)

# Test 4: Random Crossover
print("\n[4/4] Testing RANDOM CROSSOVER...")
random_result = ComparisonRunner.test_strategy(
    strategy_class='crossover',
    config={
        'prune_rate': 0.10,
        'apoptosis_interval': 500,
        'crossover_mode': 'random',
        'mutation_strength': 0.1
    },
    model_class=ApoptoticTransformer,
    dataset=shakespeare_dataset,
    device=device,
    num_steps=2000
)
random_result['strategy'] = 'Random Crossover'
results.append(random_result)

# Analysis
print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Sort by loss
sorted_results = sorted(results, key=lambda x: x['final_loss'])

print("\n🏆 RANKING BY FINAL LOSS:")
for i, result in enumerate(sorted_results, 1):
    print(f"\n{i}. {result['strategy']}")
    print(f"   Loss: {result['final_loss']:.4f}")
    print(f"   Variance: {result['variance']:.6f}")
    print(f"   Range: [{result['min_loss']:.4f}, {result['max_loss']:.4f}]")

# Baseline comparison
baseline_loss = mutation_result['final_loss']
print(f"\n📊 COMPARISON TO MUTATION BASELINE ({baseline_loss:.4f}):")

for result in sorted_results[1:]:  # Skip mutation itself
    diff = result['final_loss'] - baseline_loss
    pct = (diff / baseline_loss) * 100

    if diff < 0:
        print(f"\n✓ {result['strategy']}: {diff:+.4f} ({pct:+.1f}%) - BETTER")
    elif abs(diff) < 0.05:
        print(f"\n≈ {result['strategy']}: {diff:+.4f} ({pct:+.1f}%) - COMPARABLE")
    else:
        print(f"\n✗ {result['strategy']}: {diff:+.4f} ({pct:+.1f}%) - WORSE")

print("\n" + "="*70)
print("\n💡 INTERPRETATION:")

best = sorted_results[0]
if best['strategy'] != 'Mutation (baseline)':
    print(f"\n🎉 {best['strategy']} WINS!")
    print("Genetic crossover outperforms single-parent mutation.")
    print("Sexual reproduction for neurons is beneficial! 🧬")
else:
    print("\n🤔 Mutation still wins.")
    print("Crossover didn't provide benefit in this test.")
    print("Possible reasons:")
    print("  - Noise too small (try 0.15 instead of 0.1)")
    print("  - Need more steps to see benefit")
    print("  - Single-parent mutation already optimal for this task")

print("\n" + "="*70)
