import torch

class CharDataset(torch.utils.data.Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, text, tokenizer, seq_len=128):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
