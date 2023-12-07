from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from models.mlp import MLP
from utils.distributions import DistributionType, log_normal_pdf, log_standard_normal_pdf, log_pz_marginal, log_qz_marginal


class VAE(pl.LightningModule):
    def __init__(self,
                 data_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int],
                 px_type: DistributionType,
                 num_total_samples: int,
                 marginal_kl: bool) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.px_type = px_type
        self.num_total_samples = num_total_samples
        self.marginal_kl = marginal_kl

        self.encoder = MLP(
            [data_dim],
            hidden_dims,
            [latent_dim, latent_dim],
            hidden_activation=torch.nn.Softplus,
            # init_std = 1e-2,
            # zero_init_output_weights=[False, True],
            # zero_init_output_biases=[False, True]
        )

        self.decoder = MLP(
            [latent_dim],
            hidden_dims,
            [data_dim] if self.px_type == DistributionType.BERNOULLI else [data_dim, data_dim],
            hidden_activation=torch.nn.Softplus,
            out_activations=[torch.nn.Sigmoid] if px_type == DistributionType.BERNOULLI else None,
            # init_std = 1e-2,
            # zero_init_output_weights=[False] if px_type == DistributionType.BERNOULLI else [False, False],
            # zero_init_output_biases=[True] if px_type == DistributionType.BERNOULLI else [True, True]
        )

    @staticmethod
    def reparameterization(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(size=mu.shape, device=torch.device(mu.get_device()))
        # eps = torch.clamp(eps, -4, 4)
        return mu + eps * (0.5*logvar).exp()

    def compute_likelihoods(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.encoder(x)
        # z_logvar = torch.clamp(z_logvar, -6, 1)
        z = self.reparameterization(z_mu, z_logvar)

        if self.px_type == DistributionType.BERNOULLI:
            x_probs = self.decoder(z)
            log_px_z = -F.binary_cross_entropy(x_probs, x, reduction='none').sum(1)
        elif self.px_type == DistributionType.NORMAL:
            x_mu, x_logvar = self.decoder(z)
            # x_logvar = torch.clamp(x_logvar, -6, 1)
            log_px_z = log_normal_pdf(x, x_mu, x_logvar).sum(1)
        else:
            raise ValueError(f"Unknown px_type: {self.px_type}")

        if self.marginal_kl:
            log_qz_x = log_qz_marginal(z_logvar).sum(1)
            log_pz = log_pz_marginal(z_mu, z_logvar).sum(1)
        else:
            log_qz_x = log_normal_pdf(z, z_mu, z_logvar).sum(1)
            log_pz = log_standard_normal_pdf(z).sum(1)

        return log_px_z, log_pz, log_qz_x

    def training_step(self, *args: Any, **_: Any) -> STEP_OUTPUT:
        batch, _ = args
        x, _ = batch[0]

        log_px, log_pz, log_qz = self.compute_likelihoods(x)

        kl = log_qz - log_pz
        elbo = log_px - kl

        loss = -elbo.mean()

        return {
            "loss": loss,
            "elbo": elbo.mean(),
            "lpx": log_px.mean(),
            "kl": kl.mean(),
            "lpz": log_pz.mean(),
            "lqz": log_qz.mean(),
        }

    def configure_optimizers(self):
        weights = [mod.weight for mod in self.modules() if hasattr(mod, "weight") and mod.weight is not None]
        biases = [mod.bias for mod in self.modules() if hasattr(mod, "bias") and mod.bias is not None]

        param_groups = [{'params': weights, 'weight_decay': 1.0/self.num_total_samples},
                        {'params': biases, 'weight_decay': 1.0/self.num_total_samples}]

        return torch.optim.Adam(param_groups, lr=3e-4, betas=(0.9, 0.999))
