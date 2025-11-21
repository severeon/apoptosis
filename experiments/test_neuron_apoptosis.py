"""
Quick Test: Neuron-Level Apoptosis

Paste this into a Jupyter cell to test neuron-level apoptosis!
Should perform MUCH better than layer-level.
"""

# First, load the neuron apoptosis code
%run neuron_apoptosis.py

print("\n" + "="*70)
print("NEURON-LEVEL APOPTOSIS TEST")
print("="*70)

# Create fresh model (standard transformer, no special modifications needed!)
neuron_model = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=False  # Don't use layer-level influence scaling!
).to(device)

print(f"Model parameters: {neuron_model.get_num_params():,}")

# Identify target layers (FFN layers in each transformer block)
target_layers = []
for i in range(6):
    # Target the first FFN layer (the expansion: d_model -> 4*d_model)
    target_layers.append(f'blocks.{i}.ffn.0')

print(f"\nTarget layers for neuron apoptosis:")
for layer in target_layers:
    l = eval(f'neuron_model.{layer}')
    print(f"  {layer}: {l.in_features} → {l.out_features} neurons")

# Create neuron apoptosis manager
neuron_apoptosis = NeuronApoptosisManager(
    model=neuron_model,
    target_layers=target_layers,
    prune_rate=0.10,              # Prune 10% of neurons per layer
    apoptosis_interval=500,        # Every 500 steps
    fitness_metric='grad_activation',  # Use gradient × activation
    regrowth_strategy='mutation',      # Evolutionary regrowth
    mutation_strength=0.3
)

print(f"\nConfiguration:")
print(f"  Prune rate: {neuron_apoptosis.prune_rate*100:.0f}% of neurons per layer")
print(f"  Interval: every {neuron_apoptosis.apoptosis_interval} steps")
print(f"  Fitness metric: {neuron_apoptosis.fitness_metric}")
print(f"  Regrowth: {neuron_apoptosis.regrowth_strategy}")

# Calculate expected pruning
neurons_per_layer = 512  # 128 * 4 (d_model * 4 in FFN)
neurons_per_event = int(neurons_per_layer * neuron_apoptosis.prune_rate)
total_layers = len(target_layers)
total_neurons_per_event = neurons_per_event * total_layers

print(f"\nExpected per event: {neurons_per_event} neurons/layer × {total_layers} layers = {total_neurons_per_event} total neurons")
print(f"Expected events in 5000 steps: {5000 // neuron_apoptosis.apoptosis_interval}")

# Create custom trainer
class SimpleNeuronTrainer(Trainer):
    """Simplified trainer with neuron apoptosis."""

    def __init__(self, model, neuron_mgr, **kwargs):
        super().__init__(model, apoptosis_mgr=None, **kwargs)
        self.neuron_mgr = neuron_mgr

    def train_step(self, x, y):
        """Training step - standard."""
        self.model.train()

        logits = self.model(x)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_steps, eval_interval=100, save_interval=500):
        """Training loop with neuron apoptosis."""
        print(f"\nStarting training for {num_steps} steps...")

        pbar = tqdm(total=num_steps, desc="Training")
        train_iter = iter(self.train_loader)

        for step in range(num_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Train
            loss = self.train_step(x, y)

            # Neuron apoptosis
            apoptosis_occurred = self.neuron_mgr.step()
            if apoptosis_occurred:
                stats = self.neuron_mgr.get_stats()
                print(f"\n  Neuron ages: mean={np.mean([s['mean_age'] for s in stats.values()]):.0f}")

            # Metrics
            metrics = self.collect_metrics(loss)
            self.metrics_history.append(metrics)
            self.log_metrics(metrics)

            # Eval
            if step % eval_interval == 0:
                val_loss, val_ppl = self.eval_step()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.writer.add_scalar('val/perplexity', val_ppl, self.global_step)
                pbar.set_postfix({'loss': f'{loss:.3f}', 'val_ppl': f'{val_ppl:.2f}'})

            if step % save_interval == 0 and step > 0:
                self.save_checkpoint('checkpoint')

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print("Training complete!")
        self.save_checkpoint('final')

# Create trainer
neuron_trainer = SimpleNeuronTrainer(
    model=neuron_model,
    neuron_mgr=neuron_apoptosis,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="neuron_apoptosis_test",
    max_eval_batches=100
)

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

# Train!
neuron_trainer.train(num_steps=5000, eval_interval=100, save_interval=1000)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

baseline_loss = baseline_trainer.metrics_history[-1].loss
neuron_loss = neuron_trainer.metrics_history[-1].loss

print(f"\n📊 Final Loss:")
print(f"  Baseline (no apoptosis):    {baseline_loss:.4f}")
print(f"  Layer-level (broken):       {apoptotic_trainer.metrics_history[-1].loss:.4f}  (2.4x worse)")
print(f"  Neuron-level (NEW):         {neuron_loss:.4f}")
print(f"  Difference from baseline:   {neuron_loss - baseline_loss:+.4f}")

if abs(neuron_loss - baseline_loss) < 0.1:
    print("\n  🎉 SUCCESS! Neuron-level matches baseline!")
elif abs(neuron_loss - baseline_loss) < 0.3:
    print("\n  ✓ PROMISING! Close to baseline (within 0.3)")
elif neuron_loss < baseline_loss:
    print("\n  🚀 AMAZING! Better than baseline!")
else:
    print(f"\n  ⚠️  Still worse by {neuron_loss - baseline_loss:.2f}, but should be WAY better than layer-level")

print(f"\n💀 Apoptosis Events:")
print(f"  Layer-level: {len(apoptosis_manager.apoptosis_events)} events")
print(f"  Neuron-level: {len(neuron_apoptosis.apoptosis_events)} events")

print(f"\n📊 Neurons Pruned & Regrown:")
total_neurons = sum(count for _, _, count in neuron_apoptosis.apoptosis_events)
print(f"  Total: {total_neurons} neurons across all events")

# Age statistics
stats = neuron_apoptosis.get_stats()
print(f"\n🧬 Neuron Age Statistics (final):")
for layer, layer_stats in stats.items():
    print(f"  {layer}:")
    print(f"    Mean age: {layer_stats['mean_age']:.0f} steps")
    print(f"    Range: [{layer_stats['min_age']}, {layer_stats['max_age']}] steps")

# Save summary
with open('neuron_apoptosis_summary.txt', 'w') as f:
    f.write("NEURON-LEVEL APOPTOSIS RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Baseline Loss:      {baseline_loss:.4f}\n")
    f.write(f"Layer-level Loss:   {apoptotic_trainer.metrics_history[-1].loss:.4f} (FAILED)\n")
    f.write(f"Neuron-level Loss:  {neuron_loss:.4f}\n")
    f.write(f"\nDifference: {neuron_loss - baseline_loss:+.4f}\n")
    f.write(f"\nApoptosis events: {len(neuron_apoptosis.apoptosis_events)}\n")
    f.write(f"Total neurons pruned/regrown: {total_neurons}\n")

print("\n✓ Summary saved to: neuron_apoptosis_summary.txt")

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"Baseline:          {baseline_loss:.4f}  (reference)")
print(f"Layer-level:       {apoptotic_trainer.metrics_history[-1].loss:.4f}  ({apoptotic_trainer.metrics_history[-1].loss/baseline_loss:.2f}x worse)")
print(f"Neuron-level:      {neuron_loss:.4f}  ({neuron_loss/baseline_loss:.2f}x)")
print("="*70)
