from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch


class DatasetType(Enum):
    MNIST = "mnist"
    SVHN = "svhn"


@dataclass
class SSLIndexSplits:
    supervised_indices: np.ndarray
    unsupervised_indices: np.ndarray


def create_ssl_index_splits(labels: np.ndarray, num_supervised: int) -> SSLIndexSplits:
    num_classes = 1 + np.max(labels)

    assert 0 == num_supervised % num_classes
    sup_samples_per_classes = num_supervised // num_classes

    sup_indices = []
    unsup_indices = []

    for i in range(num_classes):
        indices = (labels == i).nonzero()[0]
        np.random.shuffle(indices)

        sup_indices.append(indices[:sup_samples_per_classes])
        unsup_indices.append(indices[sup_samples_per_classes:])

    sup_indices = np.concatenate(sup_indices, axis=0)
    unsup_indices = np.concatenate(unsup_indices, axis=0)

    return SSLIndexSplits(sup_indices, unsup_indices)


@dataclass
class SSLDataset:
    supervised_dataset: torch.utils.data.Dataset
    unsupervised_dataset: torch.utils.data.Dataset


class DataSource:
    @abstractmethod
    def num_train_data(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def create_ssl_train_datasets(self, num_supervised: int) -> SSLDataset:
        raise NotImplementedError

    @abstractmethod
    def create_train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def create_validation_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def create_test_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def decode_data(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def data_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vectors: np.ndarray,
                 *,
                 labels: Optional[np.ndarray]=None,
                 indices: Optional[np.ndarray]=None):
        self.vectors = vectors
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        if self.indices is not None:
            return self.indices.size
        return self.vectors.shape[0]

    def __getitem__(self, ix) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.indices is not None:
            ix = self.indices[ix % len(self.indices)]

        if self.labels is not None:
            return self.vectors[ix % len(self.vectors)], self.labels[ix % len(self.vectors)]

        return self.vectors[ix % len(self.vectors)]