from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from data.common import DataSource, SSLDataset, create_ssl_index_splits



@dataclass
class LatentData:
    train_mean: np.ndarray
    train_logvar: Optional[np.ndarray]
    train_y: np.ndarray
    valid_mean: np.ndarray
    valid_logvar: Optional[np.ndarray]
    valid_y: np.ndarray
    test_mean: np.ndarray
    test_logvar: Optional[np.ndarray]
    test_y: np.ndarray


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 means: np.ndarray,
                 logvars: Optional[np.ndarray],
                 *,
                 labels: Optional[np.ndarray]=None,
                 indices: Optional[np.ndarray]=None):
        self.means = means
        self.scales = np.exp(0.5*logvars) if logvars is not None else None
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        if self.indices is not None:
            return self.indices.size
        return self.means.shape[0]

    def _get_sample(self, ix):
        if self.scales is not None:
            return np.random.normal(loc=self.means[ix], scale=self.scales[ix]).astype(np.float32)
        return self.means[ix].astype(np.float32)


    def __getitem__(self, ix) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.indices is not None:
            ix = self.indices[ix % len(self.indices)]

        sample = self._get_sample(ix)

        if self.labels is not None:
            return sample, self.labels[ix % len(self.means)]

        return sample

class LatentDataSource(DataSource):
    def __init__(self, latent_data: LatentData, num_classes: int, minimal_std: Optional[float]):
        self._num_classes = num_classes
        if minimal_std is not None:
            self.keep_dimensions = np.std(latent_data.train_mean, axis=0) >= minimal_std
            print(f"Keeping {np.sum(self.keep_dimensions)} dimensions")
        else:
            self.keep_dimensions = np.ones(latent_data.train_mean.shape[1], dtype=bool)
            print(f"Keeping all {np.sum(self.keep_dimensions)} latent dimensions")
        self.original_dimensions = latent_data.train_mean.shape[1]

        self.latent_data = LatentData(
            latent_data.train_mean[:, self.keep_dimensions],
            latent_data.train_logvar[:, self.keep_dimensions] if latent_data.train_logvar is not None else None,
            latent_data.train_y,
            latent_data.valid_mean[:, self.keep_dimensions],
            latent_data.valid_logvar[:, self.keep_dimensions] if latent_data.valid_logvar is not None else None,
            latent_data.valid_y,
            latent_data.test_mean[:, self.keep_dimensions],
            latent_data.test_logvar[:, self.keep_dimensions] if latent_data.test_logvar is not None else None,
            latent_data.test_y
        )


    def num_train_data(self) -> int:
        return self.latent_data.train_mean.shape[0]

    def create_ssl_train_datasets(self, num_supervised: int) -> SSLDataset:
        ssl_index_splits = create_ssl_index_splits(self.latent_data.train_y, num_supervised)
        return SSLDataset(
            LatentDataset(self.latent_data.train_mean,
                          self.latent_data.train_logvar,
                          labels=self.latent_data.train_y,
                          indices=ssl_index_splits.supervised_indices),
            LatentDataset(self.latent_data.train_mean,
                          self.latent_data.train_logvar,
                          labels=self.latent_data.train_y,
                          indices=ssl_index_splits.unsupervised_indices)
        )

    def create_train_dataset(self) -> torch.utils.data.Dataset:
        return LatentDataset(self.latent_data.train_mean,
                            self.latent_data.train_logvar,
                             labels=self.latent_data.train_y)

    def create_validation_dataset(self) -> torch.utils.data.Dataset:
        return LatentDataset(self.latent_data.valid_mean,
                            self.latent_data.valid_logvar,
                             labels=self.latent_data.valid_y)

    def create_test_dataset(self) -> torch.utils.data.Dataset:
        return LatentDataset(self.latent_data.test_mean,
                             self.latent_data.test_logvar,
                             labels=self.latent_data.test_y)

    def decode_data(self, data: torch.Tensor) -> torch.Tensor:
        result = torch.zeros((data.shape[0], self.original_dimensions), device=data.device, dtype=data.dtype)
        result[:, self.keep_dimensions] = data
        return result

    def data_dim(self) -> int:
        return self.latent_data.train_mean.shape[1]

    def num_classes(self) -> int:
        return self._num_classes