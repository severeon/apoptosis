#!/usr/bin/env python3
"""
Production Training Script for Neural Apoptosis (stable Option B)

Features:
- TensorBoard instrumentation
- Vectorized functional regrowth
- Dual-activation pipeline (GPU for fitness, CPU for historical buffers)
- Senescence daemon lifecycle (phases 0..4) with stabilization heuristics (Option B)
- CLI for tuning
"""

import time
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.apoptotic_transformer import ApoptoticTransformer
from src.char_dataset import CharDataset
from src.char_tokenizer import CharTokenizer
from src.neuron_apoptosis_manager import NeuronApoptosisManager
from src.utils.event_logging import ExperimentLogger
from src.config import ExperimentConfig, parse_args_to_config
from src.step_scheduler import StepScheduler

# Add src to path (your src expected in repo)
# sys.path.insert(0, str(Path(__file__).parent / 'src'))

def load_shakespeare():
    """Download and load Shakespeare dataset."""
    import urllib.request

    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # print("Downloading Shakespeare dataset...")

    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')

    # print(f"Dataset size: {len(text):,} characters")
    return text

# ============================================================================
# Text Generation (unchanged)
# ============================================================================
@torch.no_grad()
def generate_text(model, tokenizer, prompt="ROMEO:", max_tokens=200, temperature=0.8, device='cpu'):
    """Generate text from the model."""
    model.eval()

    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        # Get predictions
        logits = model(tokens[:, -128:])  # Use last 128 tokens
        logits = logits[:, -1, :] / temperature

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return tokenizer.decode(tokens[0].cpu().tolist())

# ============================================================================
# Training Loop
# ============================================================================
def train(config: ExperimentConfig):
    """Main training function."""

    logger = ExperimentLogger()

    # Save run hyperparams for the database
    logger.set_params(config.to_dict())

    # Setup
    device = torch.device(config.training.device)
    run_name = config.logging.run_name or f"{config.apoptosis.strategy}_{config.training.seed}_{config.apoptosis.fitness_alpha}_{config.apoptosis.fitness_beta}_{config.apoptosis.fitness_gamma}_{config.apoptosis.activation_ema_decay}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create directories
    checkpoint_dir = Path('checkpoints') / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config.logging.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # Load data
    text = load_shakespeare()
    tokenizer = CharTokenizer(text)

    # Create train/val split (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = CharDataset(train_text, tokenizer, seq_len=config.model.seq_len)
    val_dataset = CharDataset(val_text, tokenizer, seq_len=config.model.seq_len)

    use_pin_memory = (device.type == 'cuda')
    num_workers = 4 if device.type == 'cuda' else 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        persistent_workers=False
    )

    # Create model
    torch.manual_seed(config.training.seed)
    model = ApoptoticTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_seq_len=config.model.seq_len
    ).to(device)

    # print(f"Model: {model.get_num_params():,} parameters")
    # print(f"Device: {device}")
    # print(f"Strategy: {config.apoptosis.strategy}")

    # Create apoptosis manager
    target_layers = [f'blocks.{i}.ffn.0' for i in range(config.model.n_layers)]
    manager = NeuronApoptosisManager(
        model=model,
        target_layers=target_layers,
        prune_rate=config.apoptosis.prune_rate,
        apoptosis_interval=config.apoptosis.apoptosis_interval,
        mutation_strength=config.apoptosis.mutation_strength,
        fitness_metric='grad_activation',
        regrowth_strategy='mutation',
        writer=writer,
        fitness_alpha=config.apoptosis.fitness_alpha,
        fitness_beta=config.apoptosis.fitness_beta,
        fitness_gamma=config.apoptosis.fitness_gamma,
        activation_ema_decay=config.apoptosis.activation_ema_decay,
        logger=logger
    )

    # attach config as args for backwards compatibility with manager
    manager.args = config.to_args_namespace()
    manager.writer = writer
    manager.output_buffer_size = config.logging.output_buffer_size

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize step scheduler for periodic actions
    scheduler = StepScheduler()
    data_iter = iter(dataloader)
    val_iter = iter(val_dataloader)
    losses = []

    # Define validation callback
    def run_validation():
        model.eval()
        with torch.no_grad():
            nonlocal val_iter
            try:
                val_x, val_y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_x, val_y = next(val_iter)

            val_x, val_y = val_x.to(device), val_y.to(device)
            logits = model(val_x)
            val_loss = criterion(logits.reshape(-1, logits.size(-1)), val_y.reshape(-1))
            writer.add_scalar('val/loss', val_loss.item(), scheduler.current_step)

    # Define text generation callback
    def generate_sample():
        sample = generate_text(model, tokenizer, device=str(device), max_tokens=150)
        writer.add_text('samples/generated', sample, scheduler.current_step)

    # Define checkpoint callback
    def save_checkpoint_callback():
        if scheduler.current_step > 0:
            checkpoint = {
                'step': scheduler.current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses[-1] if losses else 0.0,
                'config': config.to_dict()
            }

            if manager is not None and hasattr(manager, 'apoptosis_events'):
                checkpoint['num_apoptosis_events'] = len(manager.apoptosis_events)

            torch.save(checkpoint, checkpoint_dir / f'checkpoint_step_{scheduler.current_step}.pt')

    # Register periodic actions with scheduler
    scheduler.subscribe(run_validation, interval=config.training.val_interval, name='validation')
    scheduler.subscribe(generate_sample, interval=config.training.log_interval * 5, name='text_generation')
    scheduler.subscribe(save_checkpoint_callback, interval=config.training.checkpoint_interval, name='checkpoint')

    # print(f"\nTraining for {config.training.num_steps} steps...\n")
    # pbar = tqdm(total=config.training.num_steps, desc="Training")

    while scheduler.current_step < config.training.num_steps:
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass
        model.train()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        manager.step(loss)
        manager.log_tensorboard(writer, scheduler.current_step)

        # Logging
        losses.append(loss.item())
        writer.add_scalar('train/loss', loss.item(), scheduler.current_step)
        writer.add_scalar('apoptosis/num_events', len(manager.apoptosis_events), scheduler.current_step)

        avg_loss = sum(losses[-100:]) / min(100, len(losses))
        # pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Execute scheduled periodic actions (validation, text generation, checkpointing)
        scheduler.step()
        # pbar.update(1)

    # pbar.close()

    # Final evaluation
    final_loss = sum(losses[-100:]) / 100

    # Save final checkpoint
    checkpoint = {
        'step': scheduler.current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': final_loss,
        'config': config.to_dict()
    }

    if manager is not None and hasattr(manager, 'apoptosis_events'):
        checkpoint['num_apoptosis_events'] = len(manager.apoptosis_events)
        checkpoint['apoptosis_events'] = manager.apoptosis_events

    torch.save(checkpoint, checkpoint_dir / f'final_step_{scheduler.current_step}.pt')

    # Save results summary
    results = {
        'run_name': run_name,
        'strategy': config.apoptosis.strategy,
        'seed': config.training.seed,
        'final_loss': final_loss,
        'num_steps': scheduler.current_step,
        'num_apoptosis_events': len(manager.apoptosis_events) if manager and hasattr(manager, 'apoptosis_events') else 0,
        'config': config.to_dict()
    }

    with open(checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()

    return logger.finalize(final_loss)


if __name__ == '__main__':
    config = parse_args_to_config()
    train(config)
