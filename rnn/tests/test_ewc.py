import unittest

import numpy as np
import torch
from torch import nn

from src.methods.ewc import compute_fisher_information


class _Trial:
    def __init__(self, config, x, y, mask):
        self.config = config
        self._tensors = (x, y, mask)

    def to_tensor(self, device='cpu'):
        return tuple(tensor.to(device) for tensor in self._tensors)


class _OpposingGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        score = x[..., 0] * self.weight
        return torch.stack([score, torch.zeros_like(score)], dim=-1)


def _opposing_gradient_task(config, batch_size, mode):
    del mode
    if batch_size != 1:
        raise AssertionError("This test task expects batch_size=1")

    x = torch.tensor([[[1.0]], [[-1.0]]])
    y = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    mask = torch.ones(2, 1)
    return _Trial(config, x, y, mask)


def _random_gradient_task(config, batch_size, mode):
    del mode
    if batch_size != 1:
        raise AssertionError("This test task expects batch_size=1")

    value = float(config['rng'].uniform(0.5, 2.0))
    x = torch.tensor([[[value]]])
    y = torch.tensor([[[1.0, 0.0]]])
    mask = torch.ones(1, 1)
    return _Trial(config, x, y, mask)


class FisherInformationTest(unittest.TestCase):
    def test_timestep_gradients_are_squared_before_averaging(self):
        model = _OpposingGradientModel()
        model.train()
        config = {
            'loss_type': 'cross_entropy',
            'rng': np.random.RandomState(999),
        }
        original_rng = config['rng']

        trial = _opposing_gradient_task(config, batch_size=1, mode='random')
        x, y, _ = trial.to_tensor()
        sequence_loss = -(y * torch.log_softmax(model(x), dim=-1)).sum(dim=-1).mean()
        sequence_loss.backward()
        self.assertAlmostEqual(model.weight.grad.item(), 0.0, places=6)
        model.zero_grad(set_to_none=True)

        fisher, optimum = compute_fisher_information(
            model,
            _opposing_gradient_task,
            config,
            num_samples=16,
            fisher_seed=7,
        )

        self.assertAlmostEqual(fisher['weight'].item(), 0.25, places=6)
        self.assertEqual(optimum['weight'].item(), 0.0)
        self.assertTrue(model.training)
        self.assertIs(config['rng'], original_rng)

    def test_fisher_sampling_is_reproducible(self):
        model = _OpposingGradientModel()
        config = {'loss_type': 'cross_entropy'}

        first, _ = compute_fisher_information(
            model,
            _random_gradient_task,
            config,
            num_samples=5,
            fisher_seed=23,
        )
        second, _ = compute_fisher_information(
            model,
            _random_gradient_task,
            config,
            num_samples=5,
            fisher_seed=23,
        )

        torch.testing.assert_close(first['weight'], second['weight'])


if __name__ == '__main__':
    unittest.main()
