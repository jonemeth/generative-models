from typing import Union
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PCAData:
    eigvec: np.ndarray
    eigval: np.ndarray
    x_center: np.ndarray
    x_sd: np.ndarray


def pca(x_in, cutoff=0.99, global_sd=True):
    x_center = x_in.mean(axis=1, keepdims=True)
    x = x_in - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    x_cov = x.dot(x.T) / x.shape[1]

    eigval, eigvec = np.linalg.eig(x_cov)

    if cutoff <= 1:
        n_used = ((eigval.cumsum() / eigval.sum()) < cutoff).sum()
    else:
        n_used = cutoff

    sort_indices = eigval.argsort()[::-1]

    eigval = eigval[sort_indices[:n_used]].reshape((-1,1))
    eigvec = eigvec[:,sort_indices[:n_used]]

    return PCAData(eigvec.astype(np.float32),
                   eigval.astype(np.float32),
                   x_center.astype(np.float32),
                   x_sd.astype(np.float32))


def load_pca_data(filename: str) -> PCAData:
    pca_data = np.load(filename)

    return PCAData(pca_data['eigvec'],
                   pca_data['eigval'],
                   pca_data['x_center'],
                   pca_data['x_sd'])

# def pca_encode(data: np.ndarray, pca_data: PCAData) -> np.ndarray:
#     out = (pca_data.eigvec.T.dot(data.T - pca_data.x_center) / pca_data.x_sd) / np.sqrt(pca_data.eigval)
#     return out.T

# def pca_decode(data: Union[np.ndarray, torch.Tensor], pca_data: PCAData) -> Union[np.ndarray, torch.Tensor]:
#     out = pca_data.eigvec.dot(data.T * np.sqrt(pca_data.eigval)) * pca_data.x_sd + pca_data.x_center
#     return out.T

def pca_encode(data: np.ndarray, pca_data: PCAData) -> np.ndarray:
    batch_size = 1000

    def pca_encode_batch(batch: np.ndarray):
        return ((pca_data.eigvec.T.dot(batch.T - pca_data.x_center) / pca_data.x_sd) / np.sqrt(pca_data.eigval)).T

    out = [pca_encode_batch(data[i:i+batch_size, :]) for i in range(0, data.shape[0], batch_size)]
    out = np.concatenate(out, axis=0)
    assert out.shape[0] == data.shape[0]

    return out

def pca_decode(data: Union[np.ndarray, torch.Tensor], pca_data: PCAData) -> Union[np.ndarray, torch.Tensor]:
    batch_size = 1000

    def pca_decode_batch(batch: Union[np.ndarray, torch.Tensor]):
        return (pca_data.eigvec.dot(batch.T * np.sqrt(pca_data.eigval)) * pca_data.x_sd + pca_data.x_center).T

    out = [pca_decode_batch(data[i:i+batch_size, :]) for i in range(0, data.shape[0], batch_size)]
    out = np.concatenate(out, axis=0)
    assert out.shape[0] == data.shape[0]

    return out
