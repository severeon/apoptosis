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
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.apoptotic_transformer import ApoptoticTransformer
from src.char_dataset import CharDataset
from src.char_tokenizer import CharTokenizer
from src.neuron_apoptosis_manager import NeuronApoptosisManager
from src.logger import ExperimentLogger

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
def train(args):
    """Main training function."""

    logger = ExperimentLogger()

    # Save run hyperparams for the database
    logger.set_params(vars(args))

    # Setup
    device = torch.device(args.device)
    run_name = args.run_name or f"{args.strategy}_{args.seed}_{args.fitness_alpha}_{args.fitness_beta}_{args.fitness_gamma}_{args.activation_ema_decay}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create directories
    checkpoint_dir = Path('checkpoints') / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # Load data
    text = load_shakespeare()
    tokenizer = CharTokenizer(text)

    # Create train/val split (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = CharDataset(train_text, tokenizer, seq_len=args.seq_len)
    val_dataset = CharDataset(val_text, tokenizer, seq_len=args.seq_len)

    use_pin_memory = (device.type == 'cuda')
    num_workers = 4 if device.type == 'cuda' else 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        persistent_workers=False
    )

    # Create model
    torch.manual_seed(args.seed)
    model = ApoptoticTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len
    ).to(device)

    # print(f"Model: {model.get_num_params():,} parameters")
    # print(f"Device: {device}")
    # print(f"Strategy: {args.strategy}")

    # Create apoptosis manager
    target_layers = [f'blocks.{i}.ffn.0' for i in range(args.n_layers)]
    manager = NeuronApoptosisManager(
        model=model,
        target_layers=target_layers,
        prune_rate=args.prune_rate,
        apoptosis_interval=args.apoptosis_interval,
        mutation_strength=args.mutation_strength,
        fitness_metric='grad_activation',
        regrowth_strategy='mutation',
        writer=writer,
        fitness_alpha=args.fitness_alpha,
        fitness_beta=args.fitness_beta,
        fitness_gamma=args.fitness_gamma,
        activation_ema_decay=args.activation_ema_decay,
        logger=logger
    )

    # attach args and writer to manager for daemon behavior
    manager.args = args
    manager.writer = writer
    manager.output_buffer_size = getattr(args, 'output_buffer_size', 10) # type: ignore

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    global_step = 0
    data_iter = iter(dataloader)
    val_iter = iter(val_dataloader)
    losses = []

    # print(f"\nTraining for {args.num_steps} steps...\n")
    # pbar = tqdm(total=args.num_steps, desc="Training")

    while global_step < args.num_steps:
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

        if global_step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_x, val_y = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_x, val_y = next(val_iter)

                val_x, val_y = val_x.to(device), val_y.to(device)
                logits = model(val_x)
                val_loss = criterion(logits.reshape(-1, logits.size(-1)), val_y.reshape(-1))
                writer.add_scalar('val/loss', val_loss.item(), global_step)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        manager.step(loss)
        manager.log_tensorboard(writer, global_step)

        # Logging
        losses.append(loss.item())
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('apoptosis/num_events', len(manager.apoptosis_events), global_step)

        avg_loss = sum(losses[-100:]) / min(100, len(losses))
        # pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Periodic logging
        if global_step % args.log_interval == 0:
            # Generate sample text
            if global_step % (args.log_interval * 5) == 0:
                sample = generate_text(model, tokenizer, device=str(device), max_tokens=150)
                writer.add_text('samples/generated', sample, global_step)

                # if args.verbose:
                #     print(f"\n{'='*70}")
                #     print(f"Step {global_step} Sample:")
                #     print(f"{'='*70}")
                #     print(sample[:300])
                #     print(f"{'='*70}\n")

        # Save checkpoint
        if global_step % args.checkpoint_interval == 0 and global_step > 0:
            checkpoint = {
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'args': vars(args)
            }

            if manager is not None and hasattr(manager, 'apoptosis_events'):
                checkpoint['num_apoptosis_events'] = len(manager.apoptosis_events)

            torch.save(checkpoint, checkpoint_dir / f'checkpoint_step_{global_step}.pt')

        global_step += 1
        # pbar.update(1)

    # pbar.close()

    # Final evaluation
    final_loss = sum(losses[-100:]) / 100

    # Save final checkpoint
    checkpoint = {
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': final_loss,
        'args': vars(args)
    }

    if manager is not None and hasattr(manager, 'apoptosis_events'):
        checkpoint['num_apoptosis_events'] = len(manager.apoptosis_events)
        checkpoint['apoptosis_events'] = manager.apoptosis_events

    torch.save(checkpoint, checkpoint_dir / f'final_step_{global_step}.pt')

    # Save results summary
    results = {
        'run_name': run_name,
        'strategy': args.strategy,
        'seed': args.seed,
        'final_loss': final_loss,
        'num_steps': global_step,
        'num_apoptosis_events': len(manager.apoptosis_events) if manager and hasattr(manager, 'apoptosis_events') else 0,
        'config': vars(args)
    }

    with open(checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()

    return logger.finalize(final_loss)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer with neural apoptosis')

    parser.add_argument('--run_name', type=str, default=f"run_{str(time.time())}", help='Name of Run')
    parser.add_argument('--svd_rank', type=int, default=0, help='<UNUSED>')

    # Model
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')

    # Training
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device')

    # Apoptosis
    parser.add_argument('--strategy', type=str, default='functional',
                        choices=['baseline', 'functional', 'hybrid', 'standard'],
                        help='Apoptosis strategy')
    parser.add_argument('--prune_rate', type=float, default=0.15, help='Prune/turnover rate')
    parser.add_argument('--apoptosis_interval', type=int, default=125, help='Steps between apoptosis')
    parser.add_argument('--mutation_strength', type=float, default=0.3, help='Mutation strength')

    # Fitness function parameters (3-term model)
    parser.add_argument('--fitness_alpha', type=float, default=1.0, help='Plasticity weight (grad_norm) in fitness')
    parser.add_argument('--fitness_beta', type=float, default=1.0, help='Usefulness weight (activation_variance) in fitness')
    parser.add_argument('--fitness_gamma', type=float, default=2.0, help='Stagnation penalty weight (cosine similarity) in fitness')
    parser.add_argument('--activation_ema_decay', type=float, default=0.9, help='EMA decay for activation history tracking')

    # Senescence / lifecycle (Option B tuning)
    parser.add_argument('--enable_senescence', action='store_true', help='Enable lifecycle senescence daemon')
    parser.add_argument('--senescence_low_pct', type=float, default=0.1, help='Bottom percentile considered low-fitness')
    parser.add_argument('--senescence_patience', type=int, default=20, help='Steps of low-fitness before at-risk')
    parser.add_argument('--senescence_age_factor', type=float, default=0.01, help='Age multiplier contributing to risk')
    parser.add_argument('--phase_durations', type=str, default='10,5,1,5', help='Comma list of durations for phases 1-4 (in steps)')
    parser.add_argument('--lifecycle_warmup_steps', type=int, default=200, help='Warmup steps before senescence daemon starts')
    parser.add_argument('--max_escalations_per_step', type=int, default=5, help='Max neurons escalated to at-risk per layer per step')
    parser.add_argument('--max_kills_per_layer', type=int, default=3, help='Max neurons killed per layer per step')
    parser.add_argument('--slope_window', type=int, default=5, help='Window (steps) used to compute fitness slope')
    parser.add_argument('--slope_threshold', type=float, default=2e-4, help='Threshold for negative slope to consider escalation')
    parser.add_argument('--min_to_escalate', type=int, default=2, help='Minimum neurons to escalate when candidates exist')

    # Logging
    parser.add_argument('--log_interval', type=int, default=250, help='Steps between logging')
    parser.add_argument('--log_dir', type=str, default="logs", help='logging directory')
    parser.add_argument('--checkpoint_interval', type=int, default=2500, help='Steps between checkpoints')
    parser.add_argument('--verbose', action='store_true', help='print generated samples')

    parser.add_argument('--val_interval', type=int, default=500, help='Steps between validation checks')
    parser.add_argument('--output_buffer_size', type=int, default=10, help='Activation history buffer size')
    parser.add_argument("--output_json", action="store_true", help="print structured JSON summary instead of normal prints")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
