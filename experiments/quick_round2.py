"""
Quick Round 2: Run this in a Jupyter cell to try aggressive settings.
Paste this whole block into a new cell and run!
"""

# Round 2: Aggressive Apoptosis
print("\n" + "="*60)
print("ROUND 2: AGGRESSIVE APOPTOSIS")
print("="*60)

# Create apoptotic model (fresh start)
apoptotic_model_v2 = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=True
).to(device)

print(f"Model parameters: {apoptotic_model_v2.get_num_params():,}")

# AGGRESSIVE apoptosis manager
apoptosis_manager_v2 = ApoptosisManager(
    model=apoptotic_model_v2,
    max_lifespan=1500,           # 🔥 Shorter (was 3000)
    maturation_period=500,        # 🔥 Faster (was 750)
    apoptosis_interval=250,       # 🔥 More frequent (was 500)
    vitality_threshold=0.15,      # Die slightly earlier
    plasticity_ceiling=0.75,      # 🔥 Young layers stronger (was 0.5)
    mutation_strength=0.3         # 🔥 More exploration (was 0.2)
)

print("\nChanges from Round 1:")
print("  ✓ Lifespan: 3000 → 1500 (2x faster death)")
print("  ✓ Checks: every 500 → 250 steps")
print("  ✓ Plasticity: 0.5 → 0.75 (young layers +50% stronger)")
print("  ✓ Mutation: 0.2 → 0.3 (more diversity)")
print(f"\nExpected: ~20-25 apoptosis events (was ~10)")

# Create trainer
apoptotic_trainer_v2 = Trainer(
    model=apoptotic_model_v2,
    apoptosis_mgr=apoptosis_manager_v2,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="round2_aggressive",
    max_eval_batches=100  # Fast validation
)

# Train for 5000 steps
print("\nStarting aggressive training...")
apoptotic_trainer_v2.train(num_steps=5000, eval_interval=100, save_interval=1000)

print(f"\n🎯 Apoptosis events: {len(apoptosis_manager_v2.apoptosis_events)}")
for step, layer in apoptosis_manager_v2.apoptosis_events:
    print(f"  Step {step}: Layer {layer} died")

# Compare to baseline
baseline_final = baseline_trainer.metrics_history[-1].loss
v2_final = apoptotic_trainer_v2.metrics_history[-1].loss
diff = v2_final - baseline_final

print(f"\n📊 Results:")
print(f"  Baseline:     {baseline_final:.4f}")
print(f"  Round 2:      {v2_final:.4f}")
print(f"  Difference:   {diff:+.4f}")

if abs(diff) < 0.1:
    print("  → Competitive! ✓")
elif diff < 0:
    print("  → BETTER than baseline! 🎉")
else:
    print("  → Worse, but let's check domain shift performance...")
