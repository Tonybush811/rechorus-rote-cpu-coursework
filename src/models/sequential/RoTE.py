"""RoTE-SASRec inside ReChorus, implemented from Zhang et al. (SIGIR 2026)."""

import math

import numpy as np
import torch
from torch import nn

from models.BaseModel import SequentialModel


def calendar_levels(unix_seconds):
    """UTC calendar ordinals since 1970-01-01, including padded zeroes."""
    dates = np.asarray(unix_seconds, dtype=np.int64).astype('datetime64[s]')
    return tuple(dates.astype(unit).astype(np.int64)
                 for unit in ('datetime64[Y]', 'datetime64[M]', 'datetime64[D]'))


def rotate_pairs(x, theta):
    """Apply a 2-D rotation to each adjacent pair of head coordinates."""
    even, odd = x[..., ::2], x[..., 1::2]
    cosine, sine = theta.cos(), theta.sin()
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


class RoTEAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        if emb_size % num_heads or emb_size // num_heads % 2:
            raise ValueError('emb_size divided by num_heads must be even')
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        exponents = torch.arange(0, self.head_dim, 2).float() / self.head_dim
        for level, base in (('year', 1e6), ('month', 1e4), ('day', 1e2)):
            self.register_buffer(f'frequency_{level}', base ** (-exponents),
                                 persistent=False)

    def _heads(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _temporal_rotation(self, x, levels):
        fused = 0
        for values, level, weight in zip(levels, ('year', 'month', 'day'),
                                         (1.5, 1.0, 0.5)):
            frequency = getattr(self, f'frequency_{level}')
            angles = values[:, None, :, None].float() * frequency[None, None, None, :]
            fused = fused + weight * rotate_pairs(x, angles)
        return fused

    def forward(self, sequence, levels, valid):
        batch, length, _ = sequence.shape
        query = self._temporal_rotation(self._heads(self.query(sequence)), levels)
        key = self._temporal_rotation(self._heads(self.key(sequence)), levels)
        value = self._heads(self.value(sequence))
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones((length, length), dtype=torch.bool,
                            device=sequence.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        result = logits.softmax(-1) @ value
        return result.transpose(1, 2).reshape(batch, length, -1)


class RoTELayer(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.attention = RoTEAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff1 = nn.Linear(emb_size, emb_size)
        self.ff2 = nn.Linear(emb_size, emb_size)

    def forward(self, sequence, levels, valid):
        context = self.attention(sequence, levels, valid)
        sequence = self.norm1(sequence + self.dropout1(context))
        transformed = self.ff2(self.ff1(sequence).relu())
        return self.norm2(sequence + self.dropout2(transformed))


class RoTE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=4)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        if args.history_max < 1:
            raise ValueError('history_max must be positive')
        self.emb_size = args.emb_size
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.items = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.blocks = nn.ModuleList(RoTELayer(self.emb_size, self.num_heads,
                                              self.dropout)
                                    for _ in range(self.num_layers))
        self.apply(self.init_weights)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def forward(self, feed_dict):
        self.check_list = []
        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        ordinals = calendar_levels(feed_dict['history_times'].detach().cpu().numpy())
        levels = tuple(torch.as_tensor(x, device=history.device) for x in ordinals)
        valid = history > 0
        state = self.items(history)
        for block in self.blocks:
            state = block(state, levels, valid)
        last = state[torch.arange(history.shape[0], device=history.device), lengths - 1]
        candidates = self.items(feed_dict['item_id'])
        return {'prediction': (last.unsqueeze(1) * candidates).sum(dim=-1)}
