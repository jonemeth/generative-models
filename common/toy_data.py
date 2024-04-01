import numpy as np
import torch
import torch.distributions as D


def spiral_samples(n: int) -> torch.Tensor:
    a, b = 0.0, 0.25 / np.pi
    theta = 2.0 * np.pi * 2.0 * torch.rand((n, 1))
    r = a + b * theta

    x = r * torch.cos(theta) + 0.025 * torch.randn((n, 1))
    y = r * torch.sin(theta) + 0.025 * torch.randn((n, 1))
    return torch.cat([x, y], dim=1)


class GaussianMixtureModel(D.mixture_same_family.MixtureSameFamily):
    def __init__(self, mean, scale, mix):
        super().__init__(
            D.Categorical(mix),
            D.Independent(D.Normal(mean, scale), 1))


def get_gmm_samples(n: int) -> torch.Tensor:
    mean = torch.tensor([[-0.7, 0.7], [0.6, 0.8], [-0.6, -0.7], [0.7, -0.7]])
    scale = torch.tensor([[0.1, 0.3], [0.4, 0.2], [0.3, 0.2], [0.1, 0.4]])
    mix = torch.ones(mean.shape[0],) / mean.shape[0]

    model = GaussianMixtureModel(mean, scale, mix)

    return model.sample(torch.Size([n]))