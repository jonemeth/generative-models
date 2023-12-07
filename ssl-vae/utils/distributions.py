from enum import Enum
import torch


LOG_2PI = 1.83736998048


class DistributionType(Enum):
    NORMAL = "normal"
    BERNOULLI = "bernoulli"


def log_normal_pdf(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - mu) ** 2 / torch.exp(logvar) + logvar + LOG_2PI)


def log_standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x ** 2 + LOG_2PI)


def log_pz_marginal(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + (z_mu**2 + torch.exp(z_logvar)))


def log_qz_marginal(z_logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + 1 + z_logvar)


def to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = torch.zeros(labels.shape[0], num_classes)
    one_hot[torch.arange(labels.shape[0]), labels] = 1
    return one_hot.to(labels.device)
