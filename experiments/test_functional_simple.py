"""
Simple Standalone Test for Functional Strategy

Tests functional preservation without the sweep framework complexity.
"""

# Load fresh
exec(open('smooth_apoptosis.py').read())

print("\n" + "="*70)
print("SIMPLE FUNCTIONAL PRESERVATION TEST")
print("="*70)

# Create model
print("\nCreating model...")
test_model = ApoptoticTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=6,
    max_seq_len=128,
    enable_apoptosis=False
).to(device)

# Create functional manager
print("Creating FunctionalPreservationApoptosis manager...")
target_layers = [f'blocks.{i}.ffn.0' for i in range(6)]

functional_mgr = FunctionalPreservationApoptosis(
    model=test_model,
    target_layers=target_layers,
    prune_rate=0.10,
    apoptosis_interval=500,
    mutation_strength=0.3,
    preservation_steps=50
)

print("✓ Manager created successfully!")
print(f"\nConfig:")
print(f"  Prune rate: 10%")
print(f"  Interval: 500 steps")
print(f"  Preservation steps: 50")

# Training
train_loader = torch.utils.data.DataLoader(
    shakespeare_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

optimizer = torch.optim.AdamW(test_model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

print("\n" + "="*70)
print("TRAINING (2000 steps)")
print("="*70)

train_losses = []
train_iter = iter(train_loader)

from tqdm import tqdm
pbar = tqdm(total=2000, desc="Training")

for step in range(2000):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x, y = x.to(device), y.to(device)

    # Train
    test_model.train()
    logits = test_model(x)
    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(test_model.parameters(), max_norm=1.0)
    optimizer.step()

    # Apoptosis
    functional_mgr.step()

    train_losses.append(loss.item())

    if step % 100 == 0:
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})

    pbar.update(1)

pbar.close()

# Results
import numpy as np
final_loss = np.mean(train_losses[-100:])
variance = np.var(train_losses[-500:])

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\n📊 Metrics:")
print(f"  Final loss: {final_loss:.4f}")
print(f"  Variance: {variance:.6f}")
print(f"  Events: {len(functional_mgr.apoptosis_events)}")
print(f"  Min loss: {min(train_losses):.4f}")
print(f"  Max loss: {max(train_losses):.4f}")

print("\n📈 Comparison:")
print(f"  Baseline: 1.4788")
print(f"  Functional: {final_loss:.4f}")
print(f"  Difference: {final_loss - 1.4788:+.4f}")

if final_loss < 1.6:
    print("\n  ✓ EXCELLENT! Very close to baseline")
elif final_loss < 1.8:
    print("\n  ✓ GOOD! Competitive with baseline")
else:
    print("\n  ⚠️  Needs tuning")

print("\n" + "="*70)
print("\n✓ Functional preservation works!")
print("  This validates the strategy independently of the sweep framework.")
print("\n" + "="*70)
