"""
Fixed Quick Hyperparameter Sweep
- Functional strategy fixed (correct class)
- Hybrid event counting fixed
- Growth-only disabled (needs architecture fixes)
- No checkpoints during short runs
- Focus on hybrid configs (they're winning!)

Total: 13 configs (~65 min on M2 Max)
"""

print("Loading dependencies in correct order...")

# Load dependencies first (MUST be in this order!)
print("  1/4 Loading neuron_apoptosis_fixed.py...")
exec(open('neuron_apoptosis_fixed.py').read())

print("  2/4 Loading smooth_apoptosis.py...")
exec(open('smooth_apoptosis.py').read())

print("  3/4 Loading growth_only_strategy.py...")
exec(open('growth_only_strategy.py').read())

# Now load sweep system (depends on above)
print("  4/4 Loading hyperparameter_sweep.py...")
exec(open('hyperparameter_sweep.py').read())

print("✓ All dependencies loaded successfully!\n")

print("\n" + "="*70)
print("FIXED QUICK HYPERPARAMETER SWEEP")
print("="*70)
print("\nChanges from previous run:")
print("  ✓ Functional strategy fixed (uses correct class)")
print("  ✓ Hybrid event counting fixed")
print("  ✓ Growth-only disabled (dimension mismatch issues)")
print("  ✓ More hybrid configs (5%, 8%, 10% × 3 intervals)")
print("  ✓ No checkpoints (too frequent for short runs)")
print("="*70)

# Override run_single_experiment to disable checkpoints
original_run = HyperparameterSweep.run_single_experiment

def run_single_experiment_no_checkpoint(self, config):
    """Run experiment without saving checkpoints."""

    # Create fresh model
    model = ApoptoticTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=6,
        max_seq_len=128,
        enable_apoptosis=False
    ).to(self.device)

    # Create manager
    manager = self.create_manager(config, model)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        self.val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Metrics
    train_losses = []
    val_losses = []

    # Training loop (NO CHECKPOINTS)
    train_iter = iter(train_loader)

    for step in range(self.num_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(self.device), y.to(self.device)

        # Train step
        model.train()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Apoptosis step
        manager.step()

        train_losses.append(loss.item())

        # Eval (every 100 steps, quick)
        if step % self.eval_interval == 0 and step > 0:
            model.eval()
            val_loss_sum = 0
            val_count = 0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    logits_val = model(x_val)
                    loss_val = criterion(
                        logits_val.reshape(-1, logits_val.size(-1)),
                        y_val.reshape(-1)
                    )
                    val_loss_sum += loss_val.item()
                    val_count += 1

                    if val_count >= 50:  # Quick eval
                        break

            val_losses.append(val_loss_sum / val_count)

    # Compute metrics
    final_train_loss = np.mean(train_losses[-100:])
    final_val_loss = val_losses[-1] if val_losses else float('inf')
    train_loss_variance = np.var(train_losses[-500:])

    # Get stats from manager (FIXED event counting)
    if hasattr(manager, 'apoptosis_events'):
        num_events = len(manager.apoptosis_events)
    elif hasattr(manager, 'growth_events'):
        num_events = len(manager.growth_events)
    elif hasattr(manager, 'apoptosis_mgr'):
        # Hybrid manager wraps another manager
        num_events = len(manager.apoptosis_mgr.apoptosis_events)
    else:
        num_events = 0

    result = {
        'config': config,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'train_loss_variance': train_loss_variance,
        'num_events': num_events,
        'min_train_loss': min(train_losses),
        'max_train_loss': max(train_losses),
    }

    # Clean up
    del model
    del manager
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result

# Monkey patch
HyperparameterSweep.run_single_experiment = run_single_experiment_no_checkpoint

# Create sweep
quick_sweep = QuickSweep(
    model_class=ApoptoticTransformer,
    train_dataset=shakespeare_dataset,
    val_dataset=shakespeare_dataset,
    device=device,
    num_steps=2000,
    eval_interval=100
)

print(f"\nTesting {len(quick_sweep.define_search_space())} configurations")
print("Estimated time: ~65 minutes (~5 min per config)")
print("\n" + "="*70)

# Run it
quick_sweep.run_sweep()

print("\n" + "="*70)
print("SWEEP COMPLETE!")
print("="*70)

# Print summary
print("\n🏆 TOP 3 CONFIGURATIONS:")
valid_results = [r for r in quick_sweep.results if 'error' not in r]
sorted_results = sorted(valid_results, key=lambda x: x['final_train_loss'])

for i, result in enumerate(sorted_results[:3], 1):
    config = result['config']
    print(f"\n{i}. {config['strategy'].upper()}")
    print(f"   Loss: {result['final_train_loss']:.4f}")
    print(f"   Variance: {result['train_loss_variance']:.6f}")
    print(f"   Events: {result['num_events']}")
    print(f"   Config: {config}")

print("\n" + "="*70)
print("\n✓ Results saved to sweep_results_TIMESTAMP.json")
print("\nNext step: Run best config for full 5K steps!")
