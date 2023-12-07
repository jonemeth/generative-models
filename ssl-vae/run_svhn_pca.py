import argparse
import numpy as np
from data.pca import pca

from data.svhn import load_svhn_data


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform PCA on SVHN")
    parser.add_argument("pca_dims", type=int)
    parser.add_argument("random_seed", type=int)
    parser.add_argument("num_samples", type=int)
    return parser.parse_args()


def get_pca_filename(path: str, pca_dims: int, use_extra: bool, random_seed: int, num_samples: int) -> str:
    return f"{path}/pca_{pca_dims}{'_extra' if use_extra else ''}_seed{random_seed}_n{num_samples}.npz"


def main():
    args = get_args()
    np.random.seed(args.random_seed)

    svhn_path = "datasets/svhn"

    use_extra = True
    svhn_data = load_svhn_data(svhn_path, use_extra, n_valid=0)

    subset = svhn_data.train_x.copy()
    np.random.shuffle(subset)
    subset = subset[:args.num_samples, :]
    pca_data = pca(subset.T, cutoff=args.pca_dims)

    np.savez(get_pca_filename(svhn_path, args.pca_dims, use_extra, args.random_seed, args.num_samples),
             eigvec=pca_data.eigvec,
             eigval=pca_data.eigval,
             x_center=pca_data.x_center,
             x_sd=pca_data.x_sd)

if __name__ == "__main__":
    main()
