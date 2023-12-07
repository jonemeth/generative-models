import argparse
import os
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from data.common import DataSource, DatasetType
from data.mnist import MNISTDataSource
from data.svhn import SVHNDataSource
from models.vae import VAE
from utils.ema import EMA
from utils.distributions import DistributionType
from utils.utils import MyLogger, create_grid_image, save_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("dataset", type=str, choices=[t.value for t in DatasetType])
    parser.add_argument("random_seed", type=int)
    parser.add_argument("--pca_filename", type=str, default=None)
    return parser.parse_args()


class VAECallback(Callback):
    def __init__(self, logger: MyLogger, data_source: DataSource):
        super().__init__()

        self.logger = logger
        self.data_source = data_source
        self.log_images_folder: str = ""

        self.test_loader = DataLoader(data_source.create_test_dataset(),
                                    batch_size=100,
                                    num_workers=2,
                                    shuffle=True,
                                    pin_memory=True)

        self.data_sample, _ = next(iter(self.test_loader))
        self.latents_to_sample_from: Optional[torch.Tensor] = None

        self.ema = EMA(0.99)

    def on_train_start(self, trainer: pl.Trainer, pl_module: VAE) -> None:
        if trainer.logger is not None and trainer.logger.log_dir is not None:
            self.log_images_folder = trainer.logger.log_dir + "/images"
        os.makedirs(self.log_images_folder, exist_ok=True)

        self.latents_to_sample_from = torch.randn(size=(100, pl_module.latent_dim), device=pl_module.device)

        save_image(create_grid_image(self.data_source.decode_data(self.data_sample)),
                   f"{self.log_images_folder}/input.png")
        self.data_sample = self.data_sample.to(pl_module.device, non_blocking=True)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: VAE, outputs: Mapping[str, Any],
                           batch: Any, batch_idx: int) -> None:
        metrics = self.ema.update(outputs)
        pl_module.log_dict(metrics)

        format_strings = {"s_acc": "{:.4f}", "u_acc": "{:.4f}"}
        if 0 == (1+batch_idx) % 100:
            metrics_str = ", ".join([ ("{}: "+format_strings.get(k, "{:.2f}")).format(k, v)
                                     for k, v in metrics.items()])
            self.logger.log_text(f"epoch: {1+trainer.current_epoch:6d}, it: {1+batch_idx:8d}, {metrics_str}")


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: VAE) -> None:
        if 0 != (1+trainer.current_epoch) % 10 or 0 != trainer.global_rank:
            return

        pl_module.eval()

        with torch.inference_mode():
            z_mu, _ = pl_module.encoder(self.data_sample)
            reconstructions = pl_module.decoder(z_mu)
            if isinstance(reconstructions, tuple):
                reconstructions = reconstructions[0]

            reconstructions = self.data_source.decode_data(reconstructions)
            save_image(create_grid_image(reconstructions),
                       f"{self.log_images_folder}/reconstructions_{1+trainer.current_epoch:06d}.png")

            if pl_module.px_type == DistributionType.BERNOULLI:
                samples = pl_module.decoder(self.latents_to_sample_from)
            else:
                samples, _ = pl_module.decoder(self.latents_to_sample_from)

            samples = self.data_source.decode_data(samples)
            save_image(create_grid_image(samples),
                       f"{self.log_images_folder}/samples_{1+trainer.current_epoch:06d}.png")

        pl_module.train()

def get_experiment_name(base_name: str, dataset_type: DatasetType, pca: bool) -> str:
    return f"{base_name}_{dataset_type.value}{'_pca' if pca else ''}"


def main():
    args = get_args()
    pl.seed_everything(args.random_seed, workers=True)

    dataset_type = DatasetType(args.dataset)
    experiment_name = get_experiment_name("vae", dataset_type, args.pca_filename is not None)

    hidden_dims = [600, 600]
    marginal_kl = True
    use_svhn_extra = None
    gradient_clip_val = 5.0
    gradient_clip_algorithm = 'value'
    latent_dim = 50


    if dataset_type == DatasetType.MNIST:
        batch_size = 500
        max_epochs = 1000
        data_source = MNISTDataSource("datasets/mnist_28.pkl.gz")
        px_type = DistributionType.BERNOULLI
    elif dataset_type == DatasetType.SVHN:
        use_svhn_extra = True
        max_epochs = 5000 if use_svhn_extra else 3000
        batch_size = 5000 if use_svhn_extra else 500
        data_source = SVHNDataSource("datasets/svhn", use_svhn_extra, args.pca_filename, 0)
        px_type = DistributionType.NORMAL
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    train_dataset = data_source.create_train_dataset()

    train_loaders = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=True,
                               drop_last=True),

    model = VAE(data_source.data_dim(),
                latent_dim,
                hidden_dims,
                px_type,
                data_source.num_train_data(),
                marginal_kl)

    model.to("cuda")

    logger = MyLogger("logs", name=experiment_name, flush_logs_every_n_steps=100)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=max_epochs,
                         enable_progress_bar=False,
                         logger=logger,
                         callbacks=[VAECallback(logger, data_source), checkpoint_callback],
                         log_every_n_steps=100,
                         deterministic=True,
                         gradient_clip_val=gradient_clip_val,
                         gradient_clip_algorithm=gradient_clip_algorithm
                         )

    logger.log_hyperparams({"batch_size": batch_size,
                            "hidden_dims": hidden_dims,
                            "latent_dim": latent_dim,
                            "random_seed": args.random_seed,
                            "max_epochs": max_epochs,
                            "pca_filename": args.pca_filename,
                            "marginal_kl": marginal_kl,
                            "use_svhn_extra": use_svhn_extra,
                            "gradient_clip_val": gradient_clip_val,
                            "gradient_clip_algorithm": gradient_clip_algorithm
                             })

    trainer.fit(model=model, train_dataloaders=train_loaders)


if __name__ == "__main__":
    main()
