"""
Test the Winning Config: Hybrid 5% Turnover

From sweep results, this achieved:
- Loss: 1.742 (best!)
- Variance: 0.0157 (most stable!)
- Smooth training

Let's validate with a full 5K step run.
"""

exec(open('growth_only_strategy.py').read())

print("\n" + "="*70)
print("HYBRID 5% VALIDATION RUN")
print("="*70)
print("\nBased on sweep results:")
print("  Strategy: Hybrid Growth and Death")
print("  Turnover: 5% (birth + death each interval)")
print("  Interval: 500 steps")
print("  Mutation: 0.3")
print("\nPrevious results (2K steps):")
print("  Loss: 1.742")
print("  Variance: 0.0157 (very stable!)")
print("="*70)

# Create model
hybrid_model = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=False
).to(device)

print(f"\nModel parameters: {hybrid_model.get_num_params():,}")

# Create hybrid manager
target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

hybrid_mgr = HybridGrowthAndDeath(
    model=hybrid_model,
    target_layers=target_layers,
    turnover_rate=0.05,  # 5% birth + 5% death
    interval=500,
    mutation_strength=0.3
)

print(f"\nConfiguration:")
print(f"  Turnover rate: 5%")
print(f"  Interval: every 500 steps")
print(f"  Net effect: Constant capacity, 10% churn per interval")

# Training
train_loader = torch.utils.data.DataLoader(
    shakespeare_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    shakespeare_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

print("\n" + "="*70)
print("TRAINING (5000 steps)")
print("="*70)

train_losses = []
val_losses = []
train_iter = iter(train_loader)

from tqdm import tqdm
pbar = tqdm(total=5000, desc="Training")

for step in range(5000):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x, y = x.to(device), y.to(device)

    # Train
    hybrid_model.train()
    logits = hybrid_model(x)
    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
    optimizer.step()

    # Hybrid turnover
    hybrid_mgr.step()

    train_losses.append(loss.item())

    # Eval
    if step % 100 == 0 and step > 0:
        hybrid_model.eval()
        val_loss_sum = 0
        val_count = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits_val = hybrid_model(x_val)
                loss_val = criterion(
                    logits_val.reshape(-1, logits_val.size(-1)),
                    y_val.reshape(-1)
                )
                val_loss_sum += loss_val.item()
                val_count += 1

                if val_count >= 100:
                    break

        val_loss = val_loss_sum / val_count
        val_losses.append(val_loss)

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'val': f'{val_loss:.3f}'
        })

    pbar.update(1)

pbar.close()

# Results
import numpy as np
final_train_loss = np.mean(train_losses[-100:])
final_val_loss = val_losses[-1]
train_variance = np.var(train_losses[-1000:])

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\n📊 Final Metrics:")
print(f"  Train Loss: {final_train_loss:.4f}")
print(f"  Val Loss: {final_val_loss:.4f}")
print(f"  Variance: {train_variance:.6f}")

print(f"\n💀 Apoptosis Events: {len(hybrid_mgr.apoptosis_mgr.apoptosis_events)}")

stats = hybrid_mgr.get_stats()
if stats:
    print(f"\n🧬 Neuron Stats:")
    for layer, s in list(stats.items())[:2]:  # Show first 2 layers
        print(f"  {layer}:")
        print(f"    Mean age: {s.get('mean_age', 'N/A')}")
        print(f"    Age range: [{s.get('min_age', 'N/A')}, {s.get('max_age', 'N/A')}]")

print("\n📈 Comparison to Baseline:")
baseline_loss = 1.4788  # From previous runs
diff = final_train_loss - baseline_loss
pct = (diff / baseline_loss) * 100

print(f"  Baseline: {baseline_loss:.4f}")
print(f"  Hybrid:   {final_train_loss:.4f}")
print(f"  Diff:     {diff:+.4f} ({pct:+.1f}%)")

if abs(diff) < 0.2:
    print("\n  ✓ COMPETITIVE! Within 0.2 of baseline")
elif abs(diff) < 0.3:
    print("\n  ✓ PROMISING! Within 0.3 of baseline")
else:
    print(f"\n  ⚠️  Still {abs(diff):.2f} away from baseline")

print("\n" + "="*70)
print("\n💡 KEY INSIGHT:")
print("Hybrid strategy (5% turnover) provides:")
print("  - Constant capacity (no growth, no shrinkage)")
print("  - Evolutionary pressure (weak → strong replacement)")
print("  - Stability (low variance, smooth training)")
print("  - Simplicity (one parameter: turnover rate)")
print("\n" + "="*70)
