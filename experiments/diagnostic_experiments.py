"""
Systematic Diagnostic Experiments to Find Root Cause

Run these in order to isolate the problem.
"""

# ============================================================================
# EXPERIMENT 0: Sanity Check - Does Baseline Work?
# ============================================================================
# Already done: baseline_loss = 1.48 ✓
# This proves: Training works, data is good, device is fine


# ============================================================================
# EXPERIMENT 1: Apoptosis Disabled, Influence Scaling Only
# ============================================================================
# Test if influence scaling itself breaks the model

print("\n" + "="*70)
print("EXPERIMENT 1: Influence Scaling Test (No Apoptosis)")
print("="*70)

apoptotic_model_test1 = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=True  # Scaling enabled, but no deaths
).to(device)

# Set ALL layers to full vitality/influence (no senescence)
for i in range(6):
    apoptotic_model_test1.senescence[i] = SenescenceMetadata(
        age=0,
        vitality=1.0,
        influence_weight=1.0,  # Full power
        layer_zone="stable"
    )

# Train WITHOUT apoptosis manager (no senescence updates)
trainer_test1 = Trainer(
    model=apoptotic_model_test1,
    apoptosis_mgr=None,  # No apoptosis!
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="test1_scaling_only",
    max_eval_batches=100
)

trainer_test1.train(num_steps=2000, eval_interval=100)

loss_test1 = trainer_test1.metrics_history[-1].loss
print(f"\n📊 Experiment 1 Result:")
print(f"  Baseline:         {baseline_loss:.4f}")
print(f"  Scaling Only:     {loss_test1:.4f}")
print(f"  Difference:       {loss_test1 - baseline_loss:+.4f}")

if abs(loss_test1 - baseline_loss) < 0.2:
    print("  ✓ Influence scaling is fine")
else:
    print("  ✗ PROBLEM: Influence scaling itself breaks the model!")
    print("     → The x * influence_weight operation is the issue")


# ============================================================================
# EXPERIMENT 2: Static Reduced Influence (No Dynamics)
# ============================================================================
# Test if reduced influence is the problem (not the dynamics)

print("\n" + "="*70)
print("EXPERIMENT 2: Static Reduced Influence (50% on layers 0-1)")
print("="*70)

apoptotic_model_test2 = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    enable_apoptosis=True
).to(device)

# Set layers 0-1 to permanent 50% influence
for i in range(6):
    influence = 0.5 if i < 2 else 1.0
    apoptotic_model_test2.senescence[i] = SenescenceMetadata(
        age=0,
        vitality=1.0,
        influence_weight=influence,  # Static 50% for birth layers
        layer_zone="stable"
    )

trainer_test2 = Trainer(
    model=apoptotic_model_test2,
    apoptosis_mgr=None,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="test2_static_reduced",
    max_eval_batches=100
)

trainer_test2.train(num_steps=2000, eval_interval=100)

loss_test2 = trainer_test2.metrics_history[-1].loss
print(f"\n📊 Experiment 2 Result:")
print(f"  Baseline:         {baseline_loss:.4f}")
print(f"  Static 50%:       {loss_test2:.4f}")
print(f"  Difference:       {loss_test2 - baseline_loss:+.4f}")

if abs(loss_test2 - baseline_loss) > 0.5:
    print("  ✗ PROBLEM: Having 2 layers at 50% cripples the model!")
    print("     → Even static reduced influence is too harsh")
    print("     → Need higher plasticity ceiling OR fewer dynamic layers")


# ============================================================================
# EXPERIMENT 3: Fewer Dynamic Layers (1 death, 1 birth)
# ============================================================================
# Test if having 4 dynamic layers (2 death + 2 birth) is too many

print("\n" + "="*70)
print("EXPERIMENT 3: Only 1 Death Layer + 1 Birth Layer")
print("="*70)

apoptotic_model_test3 = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    enable_apoptosis=True
).to(device)

# NEW ZONES:
# Layer 0: birth (1 layer instead of 2)
# Layers 1-4: stable (4 stable instead of 2)
# Layer 5: death (1 layer instead of 2)

for i in range(6):
    if i == 0:
        zone, vitality, influence = "birth", 1.0, 1.0
    elif i == 5:
        zone, vitality, influence = "death", 1.0, 1.0
    else:
        zone, vitality, influence = "stable", 1.0, 1.0

    apoptotic_model_test3.senescence[i] = SenescenceMetadata(
        age=0,
        vitality=vitality,
        influence_weight=influence,
        layer_zone=zone
    )

apoptosis_mgr_test3 = ApoptosisManager(
    model=apoptotic_model_test3,
    max_lifespan=1000,  # Shorter for faster deaths
    maturation_period=300,
    apoptosis_interval=200,
    plasticity_ceiling=0.9,  # Higher ceiling
    mutation_strength=0.3
)

trainer_test3 = Trainer(
    model=apoptotic_model_test3,
    apoptosis_mgr=apoptosis_mgr_test3,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="test3_fewer_dynamic",
    max_eval_batches=100
)

trainer_test3.train(num_steps=3000, eval_interval=100)

loss_test3 = trainer_test3.metrics_history[-1].loss
events_test3 = len(apoptosis_mgr_test3.apoptosis_events)

print(f"\n📊 Experiment 3 Result:")
print(f"  Baseline:         {baseline_loss:.4f}")
print(f"  1 Birth + 1 Death: {loss_test3:.4f}  ({events_test3} events)")
print(f"  Difference:       {loss_test3 - baseline_loss:+.4f}")


# ============================================================================
# EXPERIMENT 4: NO Influence Scaling (Alternative Approach)
# ============================================================================
# Instead of scaling layer outputs, use dropout to reduce influence

print("\n" + "="*70)
print("EXPERIMENT 4: Dropout-Based Senescence (No Output Scaling)")
print("="*70)

# Modify the transformer to use dropout instead of influence scaling
class ApoptoticTransformerV3(ApoptoticTransformer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)

        mask = self.causal_mask[:seq_len, :seq_len]

        # Pass through blocks with DROPOUT instead of scaling
        for i, block in enumerate(self.blocks):
            meta = self.senescence[i]

            x = block(x, mask=mask, temperature=meta.temperature)

            # NEW: Apply senescence via dropout, not output scaling
            if self.enable_apoptosis:
                dropout_rate = 1.0 - meta.influence_weight  # Low influence = high dropout
                if dropout_rate > 0 and self.training:
                    mask_prob = torch.full((batch_size, seq_len, self.d_model),
                                          1.0 - dropout_rate, device=x.device)
                    dropout_mask = torch.bernoulli(mask_prob)
                    x = x * dropout_mask / (1.0 - dropout_rate + 1e-8)
                # DON'T scale output: x = x * meta.influence_weight

        logits = self.output_proj(x)
        return logits

apoptotic_model_test4 = ApoptoticTransformerV3(
    vocab_size=tokenizer.vocab_size,
    enable_apoptosis=True
).to(device)

# Initialize all at full power
for i in range(6):
    zone = "birth" if i < 2 else "death" if i >= 4 else "stable"
    apoptotic_model_test4.senescence[i] = SenescenceMetadata(
        age=0, vitality=1.0, influence_weight=1.0, layer_zone=zone
    )

apoptosis_mgr_test4 = ApoptosisManager(
    model=apoptotic_model_test4,
    max_lifespan=1000,
    maturation_period=300,
    apoptosis_interval=200,
    plasticity_ceiling=0.9,
    mutation_strength=0.3
)

trainer_test4 = Trainer(
    model=apoptotic_model_test4,
    apoptosis_mgr=apoptosis_mgr_test4,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="test4_dropout_senescence",
    max_eval_batches=100
)

trainer_test4.train(num_steps=3000, eval_interval=100)

loss_test4 = trainer_test4.metrics_history[-1].loss
print(f"\n📊 Experiment 4 Result:")
print(f"  Baseline:              {baseline_loss:.4f}")
print(f"  Dropout Senescence:    {loss_test4:.4f}")
print(f"  Difference:            {loss_test4 - baseline_loss:+.4f}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

results = [
    ("Baseline", baseline_loss, None),
    ("Test 1: Scaling Only", loss_test1, None),
    ("Test 2: Static 50%", loss_test2, None),
    ("Test 3: Fewer Dynamic", loss_test3, events_test3),
    ("Test 4: Dropout Method", loss_test4, len(apoptosis_mgr_test4.apoptosis_events))
]

print("\nFinal Losses:")
for name, loss, events in results:
    event_str = f" ({events} events)" if events else ""
    diff = loss - baseline_loss if name != "Baseline" else 0
    print(f"  {name:25s}: {loss:.4f}  {diff:+.4f}{event_str}")

print("\n🔍 DIAGNOSIS:")

# Find best approach
best = min(results[1:], key=lambda x: x[1])
if best[1] - baseline_loss < 0.2:
    print(f"  ✓ SUCCESS: {best[0]} works!")
    print(f"    → This approach gets within 0.2 of baseline")
else:
    print(f"  ⚠️  Best result: {best[0]} at {best[1]:.4f}")
    print(f"    → Still {best[1] - baseline_loss:.2f} worse than baseline")
    print(f"    → Consider:")
    print(f"       • The influence scaling approach might be fundamentally flawed")
    print(f"       • Try weight pruning instead of output scaling")
    print(f"       • Try neuron-level apoptosis instead of layer-level")
