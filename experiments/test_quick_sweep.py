"""
Quick Hyperparameter Sweep Test
Paste this into Jupyter notebook after loading all the apoptosis code!

Tests ~10-12 configurations in about 1 hour.
"""

# Load the sweep system
exec(open('hyperparameter_sweep.py').read())

print("\n" + "="*70)
print("QUICK HYPERPARAMETER SWEEP")
print("="*70)

# Create sweep
quick_sweep = QuickSweep(
    model_class=ApoptoticTransformer,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    device=device,
    num_steps=2000,  # Quick test
    eval_interval=100
)

# Run it
quick_sweep.run_sweep()

print("\n" + "="*70)
print("SWEEP COMPLETE!")
print("="*70)

# Results are automatically saved to JSON and printed
print("\nCheck the JSON file for full results.")
print("Top configs are printed above.")
