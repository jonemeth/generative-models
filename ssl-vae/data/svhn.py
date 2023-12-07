import math
from typing import Optional, Tuple
from dataclasses import dataclass

from scipy.io import loadmat
import numpy as np
import torch

from data.common import DataSource, SSLDataset, SimpleDataset, create_ssl_index_splits
from data.pca import load_pca_data, pca_encode, pca_decode


@dataclass
class SVHNData:
    train_x: np.ndarray
    train_y: np.ndarray
    valid_x: Optional[np.ndarray]
    valid_y: Optional[np.ndarray]
    test_x: np.ndarray
    test_y: np.ndarray

def load_svhn_mat(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    train_data = loadmat(filename)
    train_x, train_y = train_data['X'], train_data['y']
    train_x = train_x.swapaxes(0,1).T.reshape((train_x.shape[3], -1)).astype(np.float32) / 256.0
    train_y = train_y.squeeze().astype(np.int64) - 1

    return train_x, train_y


def load_svhn_data(path: str, use_extra: bool, n_valid: int) -> SVHNData:
    train_x, train_y = load_svhn_mat(path+'/train_32x32.mat')
    test_x, test_y = load_svhn_mat(path+'/test_32x32.mat')
    extra_x, extra_y = load_svhn_mat(path+'/extra_32x32.mat')

    if n_valid > 0:
        n_valid = 5000
        valid_x = train_x[-n_valid:, :]
        valid_y = train_y[-n_valid:]
        train_x = train_x[:-n_valid, :]
        train_y = train_y[:-n_valid]
    else:
        valid_x, valid_y = None, None

    if use_extra:
        train_x = np.concatenate([train_x, extra_x], axis=0)
        train_y = np.concatenate([train_y, extra_y], axis=0)

    keep = int(math.floor(train_x.shape[0]/1000.)*1000)
    train_x = train_x[:keep, :]
    train_y = train_y[:keep]

    return SVHNData(train_x, train_y, valid_x, valid_y, test_x, test_y)


class SVHNDataSource(DataSource):
    def __init__(self, path: str, use_extra: bool, pca_filename: Optional[str], n_valid: int):

        self.svhn_data = load_svhn_data(path, use_extra, n_valid)
        self.pca_data = None

        if pca_filename is not None:
            self.pca_data = load_pca_data(pca_filename)
            self.svhn_data.train_x = pca_encode(self.svhn_data.train_x, self.pca_data)
            if self.svhn_data.valid_x is not None:
                self.svhn_data.valid_x = pca_encode(self.svhn_data.valid_x, self.pca_data)
            self.svhn_data.test_x = pca_encode(self.svhn_data.test_x, self.pca_data)

    def num_train_data(self) -> int:
        return self.svhn_data.train_x.shape[0]

    def create_ssl_train_datasets(self, num_supervised: int) -> SSLDataset:
        ssl_index_splits = create_ssl_index_splits(self.svhn_data.train_y, num_supervised)
        return SSLDataset(
            SimpleDataset(self.svhn_data.train_x,
                         labels=self.svhn_data.train_y,
                         indices=ssl_index_splits.supervised_indices),
            SimpleDataset(self.svhn_data.train_x,
                         labels=self.svhn_data.train_y,
                         indices=ssl_index_splits.unsupervised_indices)
        )

    def create_train_dataset(self) -> torch.utils.data.Dataset:
        return SimpleDataset(self.svhn_data.train_x,
                            labels=self.svhn_data.train_y)

    def create_validation_dataset(self) -> torch.utils.data.Dataset:
        assert self.svhn_data.valid_x is not None, "No validation set"
        return SimpleDataset(self.svhn_data.valid_x,
                            labels=self.svhn_data.valid_y)

    def create_test_dataset(self) -> torch.utils.data.Dataset:
        return SimpleDataset(self.svhn_data.test_x,
                            labels=self.svhn_data.test_y)

    def decode_data(self, data: torch.Tensor) -> torch.Tensor:
        if self.pca_data:
            images = torch.tensor(pca_decode(data.cpu(), self.pca_data), device=data.device)
        else:
            images = data
        images = torch.clamp(images, 0.0, 1.0)
        return images.reshape(-1, 3, 32, 32)

    def data_dim(self) -> int:
        return self.svhn_data.train_x.shape[1]

    def num_classes(self) -> int:
        return 10
