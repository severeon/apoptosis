import torch
import torch.nn as nn

from src.transformer_block import TransformerBlock

class ApoptoticTransformer(nn.Module):
    """Simple Transformer for character-level language modeling."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=6, max_seq_len=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
