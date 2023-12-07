from dataclasses import dataclass
import gzip
import pickle

import numpy as np
import torch

from data.common import DataSource, SSLDataset, SimpleDataset, create_ssl_index_splits


@dataclass
class MNISTData:
    train_x: np.ndarray
    train_y: np.ndarray
    valid_x: np.ndarray
    valid_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def load_mnist_data(filename: str) -> MNISTData:
    with gzip.open(filename, "rb") as file:
        train, valid, test = pickle.load(file, encoding="latin1")
    return MNISTData(*train, *valid, *test)


class MNISTDataSource(DataSource):
    def __init__(self, filename: str):
        self.mnist_data = load_mnist_data(filename)

    def num_train_data(self) -> int:
        return self.mnist_data.train_x.shape[0]

    def create_ssl_train_datasets(self, num_supervised: int) -> SSLDataset:
        ssl_index_splits = create_ssl_index_splits(self.mnist_data.train_y, num_supervised)
        return SSLDataset(
            SimpleDataset(self.mnist_data.train_x,
                         labels=self.mnist_data.train_y,
                         indices=ssl_index_splits.supervised_indices),
            SimpleDataset(self.mnist_data.train_x,
                         labels=self.mnist_data.train_y,
                         indices=ssl_index_splits.unsupervised_indices)
        )

    def create_train_dataset(self) -> torch.utils.data.Dataset:
        return SimpleDataset(self.mnist_data.train_x,
                            labels=self.mnist_data.train_y)

    def create_validation_dataset(self) -> torch.utils.data.Dataset:
        return SimpleDataset(self.mnist_data.valid_x,
                            labels=self.mnist_data.valid_y)

    def create_test_dataset(self) -> torch.utils.data.Dataset:
        return SimpleDataset(self.mnist_data.test_x,
                            labels=self.mnist_data.test_y)

    def decode_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.reshape(-1, 1, 28, 28)

    def data_dim(self) -> int:
        return 28*28

    def num_classes(self) -> int:
        return 10
