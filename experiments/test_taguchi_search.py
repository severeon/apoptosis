"""
Test Taguchi Search vs Full Grid Search

Comparison:
- Full grid:  4 strategies × 3 rates × 3 intervals × 4 mutations = 144 experiments
- Taguchi L16: Only 16 experiments (9x faster!)

This demonstrates the power of orthogonal arrays.
"""

exec(open('taguchi_search.py').read())

print("\n" + "="*70)
print("TAGUCHI vs GRID SEARCH COMPARISON")
print("="*70)

# Define parameter space
parameters = {
    'strategy': ['standard', 'functional', 'crossover'],
    'prune_rate': [0.05, 0.10, 0.15],
    'interval': [250, 500, 750],
    'mutation_strength': [0.1, 0.2, 0.3],
}

print("\n📊 Parameter Space:")
for param, levels in parameters.items():
    print(f"  {param}: {levels}")

# Calculate full grid size
full_grid_size = np.prod([len(levels) for levels in parameters.values()])
print(f"\n🔢 Full grid search: {full_grid_size} experiments")

# Create Taguchi design
taguchi = TaguchiSearch(parameters, array_name='L16')
print(f"🎯 Taguchi {taguchi.array_name}: {len(taguchi.experiments)} experiments")
print(f"⚡ Speedup: {full_grid_size / len(taguchi.experiments):.1f}x faster!")

print("\n" + "="*70)
print("TAGUCHI EXPERIMENT DESIGN")
print("="*70)

print("\n📋 Experiments to run:")
for i, config in enumerate(taguchi.experiments, 1):
    print(f"\n{i}. {config['strategy']}")
    print(f"   prune_rate={config['prune_rate']}, "
          f"interval={config['interval']}, "
          f"mutation_strength={config['mutation_strength']}")

print("\n" + "="*70)
print("RUNNING TAGUCHI SEARCH")
print("="*70)
print("\nThis will take ~80 min (16 configs × 5 min each)")
print("Compare to full grid: ~12 hours (144 configs)!\n")

response = input("Run search? (y/n): ")

if response.lower() == 'y':
    analysis, results = ApoptosisTaguchiSearch.run_search(
        model_class=ApoptoticTransformer,
        dataset=shakespeare_dataset,
        device=device,
        num_steps=2000
    )

    print("\n" + "="*70)
    print("TAGUCHI ANALYSIS COMPLETE")
    print("="*70)

    print("\n💡 Key Insights:")
    print("\nMain effects tell you:")
    print("  - Which parameters matter MOST")
    print("  - Optimal level for each parameter")
    print("  - Expected performance of optimal config")

    print("\n🎯 Recommended next steps:")
    print("  1. Test the optimal config for 5K steps")
    print("  2. If good, you found the sweet spot!")
    print("  3. If not, run another Taguchi with refined ranges")

    # Save results
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'taguchi_results_{timestamp}.json', 'w') as f:
        json.dump({
            'experiments': taguchi.experiments,
            'results': results,
            'analysis': {
                'optimal_config': analysis['optimal_config'],
                'main_effects': analysis['main_effects'],
                'expected_sn_ratio': analysis['expected_sn_ratio']
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to: taguchi_results_{timestamp}.json")

else:
    print("\n⏭️  Skipped. Run anytime by pasting this script!")
    print("\nTo run specific config from the design:")
    print("  config = taguchi.experiments[0]  # First experiment")
    print("  result = run_single_experiment(config, ...)")

print("\n" + "="*70)
