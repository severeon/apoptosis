import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = x + attn_out
        x = self.ln1(x)

        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ln2(x)

        return x
