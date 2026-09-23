from types import SimpleNamespace

import numpy as np
import torch

from models.sequential.RoTE import RoTE, RoTEAttention, calendar_levels, rotate_pairs


def test_calendar_levels_use_utc_ordinals_since_epoch():
    y, m, d = calendar_levels(np.array([0, 31536000, 1383264000]))
    assert y.tolist() == [0, 1, 43]
    assert m.tolist() == [0, 12, 526]
    assert d.tolist() == [0, 365, 16010]


def test_rotary_transform_preserves_each_pair_norm():
    x = torch.tensor([[[[1., 2., 3., 4.]]]])
    theta = torch.tensor([[[[0.3, 1.1]]]])
    result = rotate_pairs(x, theta)
    torch.testing.assert_close(result.norm(dim=-1), x.norm(dim=-1))


def test_attention_responds_to_changed_time_span():
    block = RoTEAttention(4, 1)
    with torch.no_grad():
        for linear in (block.query, block.key, block.value):
            linear.weight.copy_(torch.eye(4))
            linear.bias.zero_()
    sequence = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
    same = tuple(torch.tensor([[n, n]]) for n in (1, 12, 365))
    apart = tuple(torch.tensor([[a, b]]) for a, b in ((1, 10), (12, 120), (365, 3650)))
    valid = torch.tensor([[True, True]])
    assert not torch.allclose(block(sequence, same, valid), block(sequence, apart, valid))


def test_rechorus_model_forward_and_gradients_on_cpu():
    args = SimpleNamespace(device=torch.device('cpu'), model_path='unused.pt',
                           buffer=0, num_neg=1, dropout=0., test_all=0,
                           history_max=4, emb_size=8, num_layers=1, num_heads=2)
    corpus = SimpleNamespace(n_users=2, n_items=12)
    model = RoTE(args, corpus)
    feed = {'item_id': torch.tensor([[4, 5, 6]]),
            'history_items': torch.tensor([[1, 2, 3, 0]]),
            'history_times': torch.tensor([[31536000, 63158400, 94694400, 0]]),
            'lengths': torch.tensor([3])}
    scores = model(feed)['prediction']
    assert scores.shape == (1, 3) and torch.isfinite(scores).all()
    scores.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
