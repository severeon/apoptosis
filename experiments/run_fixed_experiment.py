"""
QUICK RUN: Fixed apoptosis experiment (all layers start healthy)

Paste this into a Jupyter cell to test the fix!
"""

print("\n" + "="*70)
print("APOPTOSIS EXPERIMENT - FIXED (All Layers Start Healthy)")
print("="*70)

# Create model with FIXED initialization
apoptotic_model_fixed = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=True
).to(device)

# OVERRIDE the broken initialization
print("\n🔧 Applying fix: All layers start at 100% vitality...")

for i in range(apoptotic_model_fixed.n_layers):
    if i < 2:  # Birth zone
        zone = "birth"
    elif i >= 4:  # Death zone
        zone = "death"
    else:  # Stable
        zone = "stable"

    # Everyone starts healthy!
    apoptotic_model_fixed.senescence[i] = SenescenceMetadata(
        age=0,
        vitality=1.0,        # 🔥 100% (was 0.5 for birth layers)
        influence_weight=1.0, # 🔥 FULL POWER (was 0.5)
        layer_zone=zone,
        temperature=1.0,
        dropout_rate=0.1,
        learning_rate_scale=1.0
    )

print("✓ Fixed! All layers now start at full power")
print("\nInitial vitality by layer:")
for i, meta in enumerate(apoptotic_model_fixed.senescence):
    print(f"  Layer {i} ({meta.layer_zone:6s}): vitality={meta.vitality:.2f}, influence={meta.influence_weight:.2f}")

# Create apoptosis manager with AGGRESSIVE settings
apoptosis_manager_fixed = ApoptosisManager(
    model=apoptotic_model_fixed,
    max_lifespan=1500,        # Aggressive
    maturation_period=500,
    apoptosis_interval=250,
    vitality_threshold=0.15,
    plasticity_ceiling=0.75,  # Higher ceiling
    mutation_strength=0.3
)

print(f"\nHyperparameters:")
print(f"  Lifespan: {apoptosis_manager_fixed.max_lifespan} steps")
print(f"  Interval: {apoptosis_manager_fixed.apoptosis_interval} steps")
print(f"  Plasticity: {apoptosis_manager_fixed.plasticity_ceiling}")

# Create trainer
apoptotic_trainer_fixed = Trainer(
    model=apoptotic_model_fixed,
    apoptosis_mgr=apoptosis_manager_fixed,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="fixed_all_healthy",
    max_eval_batches=100
)

print("\n" + "="*70)
print("STARTING TRAINING (Fixed Model)")
print("="*70)

# Train!
apoptotic_trainer_fixed.train(num_steps=5000, eval_interval=100, save_interval=1000)

# Results
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

baseline_loss = baseline_trainer.metrics_history[-1].loss
original_loss = apoptotic_trainer.metrics_history[-1].loss
fixed_loss = apoptotic_trainer_fixed.metrics_history[-1].loss

print(f"\n📊 Final Loss:")
print(f"  Baseline:         {baseline_loss:.4f}")
print(f"  Original (broken): {original_loss:.4f}  ({original_loss/baseline_loss:.2f}x worse)")
print(f"  Fixed (healthy):   {fixed_loss:.4f}  ({fixed_loss/baseline_loss:.2f}x)")

diff = fixed_loss - baseline_loss
print(f"\n  Fixed vs Baseline: {diff:+.4f}")

if abs(diff) < 0.1:
    print("  🎉 COMPETITIVE! Fixed model matches baseline!")
elif diff < 0:
    print("  🚀 BETTER than baseline!")
elif diff < 0.3:
    print("  ✓ Close to baseline (within 0.3)")
else:
    print("  ⚠️  Still worse, but should be MUCH better than original")

print(f"\n💀 Apoptosis Events:")
print(f"  Original: {len(apoptosis_manager.apoptosis_events)} events")
print(f"  Fixed:    {len(apoptosis_manager_fixed.apoptosis_events)} events")

print("\n" + "="*70)

# Save quick summary
with open('fixed_experiment_summary.txt', 'w') as f:
    f.write(f"QUICK SUMMARY\n")
    f.write(f"=============\n\n")
    f.write(f"Baseline Loss:  {baseline_loss:.4f}\n")
    f.write(f"Original Loss:  {original_loss:.4f} ({original_loss/baseline_loss:.2f}x worse)\n")
    f.write(f"Fixed Loss:     {fixed_loss:.4f} ({fixed_loss/baseline_loss:.2f}x)\n")
    f.write(f"\nImprovement: {original_loss - fixed_loss:.4f}\n")
    f.write(f"Gap to baseline: {diff:+.4f}\n")
    f.write(f"\nApoptosis events: {len(apoptosis_manager_fixed.apoptosis_events)}\n")

print("✓ Summary saved to: fixed_experiment_summary.txt")
