from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from models.mlp import MLP
from utils.ema import EMA
from utils.distributions import DistributionType, log_normal_pdf, log_pz_marginal, log_qz_marginal, log_standard_normal_pdf, to_onehot


class SSLVAE(pl.LightningModule):
    def __init__(self,
                 data_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int],
                 px_type: DistributionType,
                 num_classes: int,
                 classification_weight: float,
                 num_total_samples: int,
                 marginal_kl: bool) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.px_type = px_type
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.num_total_samples = num_total_samples
        self.marginal_kl = marginal_kl

        self.encoder = MLP(
            [data_dim, num_classes],
            hidden_dims,
            [latent_dim, latent_dim],
            hidden_activation=torch.nn.Softplus,
            # init_std = 1e-3,
            # zero_init_output_weights=[False, True],
            # zero_init_output_biases=[False, True]
        )

        self.decoder = MLP(
            [latent_dim, num_classes],
            hidden_dims,
            [data_dim] if self.px_type == DistributionType.BERNOULLI else [data_dim, data_dim],
            hidden_activation=torch.nn.Softplus,
            out_activations=[torch.nn.Sigmoid] if px_type == DistributionType.BERNOULLI else None,
            # init_std = 1e-3,
            # zero_init_output_weights=[False, False],
            # zero_init_output_biases=[True, True]
        )

        self.classifier = MLP(
            [data_dim],
            hidden_dims,
            [num_classes],
            hidden_activation=torch.nn.Softplus,
            # init_std = 1e-3,
            # zero_init_biases=True,
            # zero_init_output_biases=[True]
        )

    @staticmethod
    def reparameterization(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(size=mu.shape, device=torch.device(mu.get_device()))
        # eps = torch.clamp(eps, -4, 4)
        return mu + eps * (0.5*logvar).exp()

    def compute_likelihoods(self, x: torch.Tensor,
                            y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = to_onehot(y, self.num_classes)
        z_mu, z_logvar = self.encoder(x, y)
        # z_logvar = torch.clamp(z_logvar, -6, 1)
        z = self.reparameterization(z_mu, z_logvar)

        if self.px_type == DistributionType.BERNOULLI:
            x_probs = self.decoder(z, y)
            log_px_z = -F.binary_cross_entropy(x_probs, x, reduction='none').sum(1)
        elif self.px_type == DistributionType.NORMAL:
            x_mu, x_logvar = self.decoder(z, y)
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


        log_py = (y * np.log(1.0 / self.num_classes)).sum(1)

        return log_px_z, log_pz, log_qz_x, log_py

    def training_step(self, *args: Any, **_: Any) -> STEP_OUTPUT:
        batch, _ = args
        x_sup, y_sup = batch["supervised"]
        x_unsup, y_unsup = batch["unsupervised"]
        n_sup = x_sup.shape[0]

        y_unsup_probs = F.softmax(self.classifier(x_unsup), dim=1)

        x_all = torch.cat([x_sup, torch.repeat_interleave(x_unsup, self.num_classes, dim=0)], dim=0)
        y_all = torch.cat([y_sup, torch.arange(self.num_classes, device=y_sup.device).repeat(x_unsup.shape[0])])
        log_px_z_all, log_pz_all, log_qz_x_all, log_py_all = self.compute_likelihoods(x_all, y_all)

        sup_px_z = log_px_z_all[:n_sup]
        sup_kl = log_qz_x_all[:n_sup] - log_pz_all[:n_sup]
        sup_py = log_py_all[:n_sup]
        sup_elbo = sup_px_z - sup_kl + sup_py

        y_sup_logits = self.classifier(x_sup)
        y_sup_log_probs = torch.log_softmax(y_sup_logits, 1)
        sup_log_qy = (to_onehot(y_sup, self.num_classes) * y_sup_log_probs).sum(1)

        unsup_px_z = log_px_z_all[n_sup:]
        unsup_kl = log_qz_x_all[n_sup:] - log_pz_all[n_sup:]
        unsup_py = log_py_all[n_sup:]
        unsup_entropy_qy = (-y_unsup_probs * (1e-8+y_unsup_probs).log()).sum(dim=1)
        unsup_elbo = ((unsup_px_z - unsup_kl + unsup_py) * y_unsup_probs.flatten()).view(-1, self.num_classes).sum(-1) + unsup_entropy_qy


        # Mean
        num_batches = self.num_total_samples // (x_sup.shape[0]+x_unsup.shape[0])
        loss = -(num_batches*(sup_elbo.sum() + self.classification_weight*sup_log_qy.sum() +
                              unsup_elbo.sum())) / self.num_total_samples

        sup_acc = (y_sup == torch.argmax(y_sup_logits, dim=1)).sum().item() / y_sup_logits.shape[0]
        unsup_acc = (y_unsup == torch.argmax(y_unsup_probs, dim=1)).sum().item() / y_unsup_probs.shape[0]
        sup_entropy_qy = (-y_sup_log_probs.exp() * y_sup_log_probs).sum(dim=1)

        return {
            "loss": loss,
            "s_lpx": sup_px_z.mean(),
            "u_lpx": unsup_px_z.mean(),
            "s_kl": sup_kl.mean(),
            "u_kl": unsup_kl.mean(),
            "s_py": sup_py.mean(),
            "u_py": unsup_py.mean(),
            "s_qy": sup_log_qy.mean(),
            "s_H": sup_entropy_qy.mean(),
            "u_H": unsup_entropy_qy.mean(),
            "s_acc": sup_acc,
            "u_acc": unsup_acc,
        }

    def configure_optimizers(self):
        # weights = [mod.weight for mod in self.modules() if hasattr(mod, "weight") and mod.weight is not None]
        # biases = [mod.bias for mod in self.modules() if hasattr(mod, "bias") and mod.bias is not None]

        # param_groups = [{'params': weights, 'weight_decay': 1.0/self.num_total_samples},
        #                 {'params': biases, 'weight_decay': 1.0/self.num_total_samples}]

        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1.0/self.num_total_samples)
