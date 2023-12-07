import argparse
import os
from typing import Any, Mapping, Optional, Tuple
import yaml

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback as pl_Callback

from data.common import DataSource, DatasetType
from data.latent import LatentData, LatentDataSource
from data.mnist import MNISTDataSource
from data.svhn import SVHNDataSource
from models.sslvae import SSLVAE
from models.vae import VAE
from utils.distributions import DistributionType, to_onehot
from utils.ema import EMA
from utils.utils import MyLogger, create_grid_image, save_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("dataset", type=str, choices=[t.value for t in DatasetType])
    parser.add_argument("num_sup", type=int)
    parser.add_argument("random_seed", type=int)
    parser.add_argument("--pca_filename", type=str, default=None)
    parser.add_argument("--first_layer_log_dir", type=str, default=None)
    return parser.parse_args()


def compute_accuracy(dataloader: DataLoader, model: SSLVAE) -> float:
    hits, count = 0, 0

    for x, y in dataloader:
        x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        count += len(y)
        predictions = model.classifier(x)
        predictions = torch.argmax(predictions, dim=1)
        hits += (y == predictions).sum().cpu()
    return hits / count


def create_cross_samples(model: SSLVAE, classes: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    classes = to_onehot(classes, model.num_classes)

    n_classes = classes.shape[0]
    n_latents = latents.shape[0]

    classes = classes.repeat_interleave(n_latents, dim=0)
    latents = latents.repeat(n_classes, 1)

    samples = model.decoder(latents, classes)
    if isinstance(samples, tuple):
        samples = samples[0]

    return samples


def create_analogies(model: SSLVAE, samples: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    samples = samples.to(model.device, non_blocking=True)
    latents = model.encoder(samples, to_onehot(classes, model.num_classes))
    if isinstance(latents, tuple):
        latents = latents[0]
    cross_samples = create_cross_samples(model, classes, latents)
    return torch.cat([samples, cross_samples])



class Callback(pl_Callback):
    def __init__(self,
                 logger: MyLogger,
                 data_source: DataSource,
                 first_layer: Optional[VAE] = None,
                 first_layer_data_source: Optional[DataSource] = None):
        super().__init__()

        self.logger = logger
        self.data_source = data_source
        self.first_layer = first_layer
        self.first_layer_data_source = first_layer_data_source

        self.log_images_folder: str = ""
        self.sample_latents: Optional[torch.Tensor] = None
        self.sample_classes: Optional[torch.Tensor] = None

        self.validation_loader = DataLoader(data_source.create_validation_dataset(),
                                            batch_size=100,
                                            num_workers=2,
                                            shuffle=True,
                                            pin_memory=True)

        self.test_loader = DataLoader(data_source.create_test_dataset(),
                                      batch_size=100,
                                      num_workers=2,
                                      pin_memory=True)

        # Collect one sample per class for analogies
        analogy_samples = {}
        for samples, labels in self.validation_loader:
            for label in [ix for ix in range(data_source.num_classes()) if ix not in analogy_samples]:
                indices = torch.where(labels == label)[0]
                if len(indices) > 0:
                    analogy_samples[label] = samples[indices[0]]

            if None not in analogy_samples:
                break
        self.analogy_samples = torch.stack([analogy_samples[ix] for ix in range(data_source.num_classes())])

        self.ema = EMA(0.99)

    def on_train_start(self, trainer: pl.Trainer, pl_module: SSLVAE) -> None:
        if trainer.logger is not None and trainer.logger.log_dir is not None:
            self.log_images_folder = trainer.logger.log_dir + "/images"
        os.makedirs(self.log_images_folder, exist_ok=True)

        self.sample_latents = torch.randn(size=(10, pl_module.latent_dim), device=pl_module.device)
        self.sample_classes = torch.arange(0, pl_module.num_classes, device=pl_module.device)


    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: SSLVAE, outputs: Mapping[str, Any],
                           batch: Any, batch_idx: int) -> None:
        metrics = self.ema.update(outputs)
        format_strings = {"s_acc": "{:.4f}", "u_acc": "{:.4f}"}
        pl_module.log_dict(metrics)

        if 0 == (1+batch_idx) % 100:
            metrics_str = ", ".join([ ("{}: "+format_strings.get(k, "{:.2f}")).format(k, v)
                                     for k, v in metrics.items()])
            self.logger.log_text(f"epoch: {1+trainer.current_epoch:6d}, it: {1+batch_idx:8d}, {metrics_str}")


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSLVAE) -> None:
        if 0 != (1+trainer.current_epoch) % 10 or 0 != trainer.global_rank:
            return

        pl_module.eval()
        with torch.inference_mode():
            val_accuracy = compute_accuracy(self.validation_loader, pl_module)
            test_accuracy = compute_accuracy(self.test_loader, pl_module)

            # Samples
            if self.sample_classes is not None and self.sample_latents is not None:
                cross_samples = create_cross_samples(pl_module, self.sample_classes, self.sample_latents)
                cross_samples = self.data_source.decode_data(cross_samples)

                if self.first_layer is not None and self.first_layer_data_source is not None:
                    cross_samples = self.first_layer.decoder(cross_samples)
                    if isinstance(cross_samples, tuple):
                        cross_samples = cross_samples[0]
                    cross_samples = self.first_layer_data_source.decode_data(cross_samples)


                save_image(create_grid_image(cross_samples),
                           f"{self.log_images_folder}/samples_{1+trainer.current_epoch:06d}.png")

            # Analogies
            if self.sample_classes is not None and self.analogy_samples is not None:
                analogies = create_analogies(pl_module, self.analogy_samples, self.sample_classes)
                analogies = self.data_source.decode_data(analogies)

                if self.first_layer is not None and self.first_layer_data_source is not None:
                    analogies = self.first_layer.decoder(analogies)
                    if isinstance(analogies, tuple):
                        analogies = analogies[0]
                    analogies = self.first_layer_data_source.decode_data(analogies)

                save_image(create_grid_image(analogies, n_rows=1+pl_module.num_classes, transpose=True),
                           f"{self.log_images_folder}/analogies_{1+trainer.current_epoch:06d}.png")


        pl_module.train()

        self.logger.log_text(f"epoch: {1+trainer.current_epoch:04d}/{trainer.max_epochs:04d}, val_accuracy:" +
                             f"{val_accuracy:.4f}, test_accuracy: {test_accuracy:.4f}")


def encode_dataset(loader: DataLoader, vae: VAE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means, logvars, ys = [], [], []

    for x, y in loader:
        x = x.to(vae.device, non_blocking=True)
        y = y.to(vae.device, non_blocking=True)
        mean, logvar = vae.encoder(x)
        means.append(mean)
        logvars.append(logvar)
        ys.append(y)

    return torch.cat(means), torch.cat(logvars), torch.cat(ys)


def create_latent_data_source(data_source: DataSource, vae: VAE, *, test_on_mean_latents: bool, minimal_std: Optional[float]) -> LatentDataSource:
    train_loader = DataLoader(data_source.create_train_dataset(),
                              batch_size=100,
                              num_workers=2,
                              pin_memory=True)
    valid_loader = DataLoader(data_source.create_validation_dataset(),
                              batch_size=100,
                              num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(data_source.create_test_dataset(),
                             batch_size=100,
                             num_workers=2,
                             pin_memory=True)

    train_mean, train_logvar, train_y = [t.cpu().numpy() for t in encode_dataset(train_loader, vae)]
    valid_mean, valid_logvar, valid_y = [t.cpu().numpy() for t in encode_dataset(valid_loader, vae)]
    test_mean, test_logvar, test_y = [t.cpu().numpy() for t in encode_dataset(test_loader, vae)]

    if test_on_mean_latents:
        valid_logvar = None
        test_logvar = None

    return LatentDataSource(LatentData(train_mean, train_logvar, train_y,
                                       valid_mean, valid_logvar, valid_y,
                                       test_mean, test_logvar, test_y),
                            data_source.num_classes(), minimal_std)



def get_experiment_name(base_name: str, dataset: DatasetType, pca: bool, num_sup: int) -> str:
    return f"{base_name}_{dataset.value}{'_pca' if pca else ''}_{num_sup:04d}"


def main():
    args = get_args()

    dataset = DatasetType(args.dataset)

    pl.seed_everything(args.random_seed, workers=True)

    first_layer_hparams = None
    if args.first_layer_log_dir is not None:
        with open(args.first_layer_log_dir+'/hparams.yaml', 'rt', encoding='utf-8') as file:
            first_layer_hparams = yaml.safe_load(file)

    pca = args.pca_filename is not None or \
        (first_layer_hparams is not None and "pca_filename" in first_layer_hparams and first_layer_hparams['pca_filename'] is not None)

    experiment_name = get_experiment_name("sslvae_2layers" if args.first_layer_log_dir is not None else "sslvae_1layer",
                                          dataset,
                                          pca,
                                          args.num_sup)
    num_batches = 100 if args.num_sup == 100 else 200
    hidden_dims = [500]
    latent_dim = 50
    max_epochs = 3000
    marginal_kl = True
    gradient_clip_val = None
    gradient_clip_algorithm = None
    pca_filename = None

    if dataset == DatasetType.MNIST:
        alpha = 0.1
        data_source = MNISTDataSource("datasets/mnist_28.pkl.gz")
        px_type = DistributionType.BERNOULLI
    elif dataset == DatasetType.SVHN:
        alpha = 0.5
        if first_layer_hparams is not None:
            assert args.pca_filename is None
            pca_filename = first_layer_hparams['pca_filename']
        else:
            pca_filename = args.pca_filename

        print("pca_filename:", pca_filename)

        data_source = SVHNDataSource("datasets/svhn", False, pca_filename, n_valid=5000)
        px_type = DistributionType.NORMAL
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if args.first_layer_log_dir is not None:
        first_layer_data_source = data_source

        first_layer = VAE.load_from_checkpoint(args.first_layer_log_dir+'/checkpoints/last.ckpt')
        first_layer.to("cuda")
        first_layer.eval()
        first_layer.freeze()

        with torch.inference_mode():
            data_source = create_latent_data_source(data_source, first_layer, test_on_mean_latents=True, minimal_std=0.1)
        px_type = DistributionType.NORMAL
    else:
        first_layer = None
        first_layer_data_source = None

    num_total = data_source.num_train_data()
    num_unsup = num_total - args.num_sup

    assert 0 == args.num_sup % num_batches
    sup_batch_size = args.num_sup // num_batches
    unsup_batch_size = num_unsup // num_batches

    assert num_total == (sup_batch_size+unsup_batch_size) * num_batches

    ssl_train_datasets = data_source.create_ssl_train_datasets(args.num_sup)

    assert args.num_sup == len(ssl_train_datasets.supervised_dataset)
    assert num_unsup == len(ssl_train_datasets.unsupervised_dataset)

    train_loaders = {
        "supervised": DataLoader(ssl_train_datasets.supervised_dataset,
                                 batch_size=sup_batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=4,
                                 pin_memory=True),
        "unsupervised": DataLoader(ssl_train_datasets.unsupervised_dataset,
                                   batch_size=unsup_batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=4,
                                   pin_memory=True)
    }

    model = SSLVAE(data_source.data_dim(),
                   latent_dim,
                   hidden_dims,
                   px_type,
                   data_source.num_classes(),
                   alpha*(num_total/args.num_sup),
                   num_total,
                   marginal_kl)

    model.to("cuda")

    logger = MyLogger("logs", name=experiment_name, flush_logs_every_n_steps=100)

    trainer = pl.Trainer(max_epochs=max_epochs,
                         enable_progress_bar=False,
                         logger=logger,
                         callbacks=[Callback(logger, data_source, first_layer, first_layer_data_source)],
                         deterministic=True,
                         gradient_clip_val=gradient_clip_val,
                         gradient_clip_algorithm=gradient_clip_algorithm)

    logger.log_hyperparams({"num_batches": num_batches,
                            "num_sup": args.num_sup,
                            "hidden_dims": hidden_dims,
                            "latent_dim": latent_dim,
                            "random_seed": args.random_seed,
                            "pca_filename": pca_filename,
                            "max_epochs": max_epochs,
                            "first_layer_log_dir": args.first_layer_log_dir,
                            "alpha": alpha,
                            "marginal_kl": marginal_kl,
                            "gradient_clip_val": gradient_clip_val,
                            "gradient_clip_algorithm": gradient_clip_algorithm})

    trainer.fit(model=model, train_dataloaders=train_loaders)


if __name__ == "__main__":
    main()
