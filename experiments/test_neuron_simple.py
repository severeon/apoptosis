"""
Simple Neuron-Level Apoptosis Test (Fixed)
Paste this into Jupyter notebook cell!
"""

# First load the fixed neuron apoptosis code
exec(open('neuron_apoptosis_fixed.py').read())

print("\n" + "="*70)
print("NEURON-LEVEL APOPTOSIS TEST")
print("="*70)

# Create fresh model
neuron_model = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=False  # Don't use layer-level!
).to(device)

print(f"Model parameters: {neuron_model.get_num_params():,}")

# Target FFN layers (first linear in each FFN)
target_layers = []
for i in range(6):
    target_layers.append(f'blocks.{i}.ffn.0')

print(f"\nTarget layers:")
for layer_name in target_layers:
    layer = neuron_model.blocks[i].ffn[0]  # Access directly
    print(f"  {layer_name}: {layer.in_features} → {layer.out_features} neurons")

# Create neuron apoptosis manager
neuron_apoptosis = NeuronApoptosisManager(
    model=neuron_model,
    target_layers=target_layers,
    prune_rate=0.10,
    apoptosis_interval=500,
    fitness_metric='grad_activation',
    regrowth_strategy='mutation',
    mutation_strength=0.3
)

print(f"\nConfiguration:")
print(f"  Prune rate: {neuron_apoptosis.prune_rate*100:.0f}%")
print(f"  Interval: every {neuron_apoptosis.apoptosis_interval} steps")
print(f"  Fitness: {neuron_apoptosis.fitness_metric}")

# Extend the existing Trainer class
from tqdm import tqdm

class SimpleNeuronTrainer(Trainer):
    def __init__(self, model, neuron_mgr, **kwargs):
        super().__init__(model, apoptosis_mgr=None, **kwargs)
        self.neuron_mgr = neuron_mgr

    def train(self, num_steps, eval_interval=100, save_interval=500):
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

            # Standard training step
            self.model.train()
            logits = self.model(x)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Neuron apoptosis
            apoptosis_occurred = self.neuron_mgr.step()

            # Metrics
            metrics = self.collect_metrics(loss.item())
            self.metrics_history.append(metrics)
            self.log_metrics(metrics)

            # Eval
            if step % eval_interval == 0:
                val_loss, val_ppl = self.eval_step()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.writer.add_scalar('val/perplexity', val_ppl, self.global_step)
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'val_ppl': f'{val_ppl:.2f}'})

            if step % save_interval == 0 and step > 0:
                self.save_checkpoint('checkpoint')

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print("Training complete!")

# Create trainer
neuron_trainer = SimpleNeuronTrainer(
    model=neuron_model,
    neuron_mgr=neuron_apoptosis,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    batch_size=64,
    learning_rate=3e-4,
    experiment_name="neuron_apoptosis",
    max_eval_batches=100
)

print("\n" + "="*70)
print("TRAINING")
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
print(f"  Baseline:       {baseline_loss:.4f}")
print(f"  Neuron-level:   {neuron_loss:.4f}")
print(f"  Difference:     {neuron_loss - baseline_loss:+.4f}")

if abs(neuron_loss - baseline_loss) < 0.1:
    print("\n  🎉 SUCCESS! Neuron-level matches baseline!")
elif abs(neuron_loss - baseline_loss) < 0.3:
    print("\n  ✓ PROMISING! Close to baseline")
else:
    print(f"\n  ⚠️  Still {abs(neuron_loss - baseline_loss):.2f} away from baseline")

print(f"\n💀 Apoptosis Events: {len(neuron_apoptosis.apoptosis_events)}")
print(f"   Total neurons pruned: {sum(c for _, _, c in neuron_apoptosis.apoptosis_events)}")

# Stats
stats = neuron_apoptosis.get_stats()
print(f"\n🧬 Neuron Ages (final):")
for layer, s in stats.items():
    print(f"  {layer}: mean={s['mean_age']:.0f}, range=[{s['min_age']}, {s['max_age']}]")

print("\n" + "="*70)
