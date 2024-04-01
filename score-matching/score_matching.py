import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

from common.toy_data import get_gmm_samples, spiral_samples


class ScoreModel(torch.nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self._dims = dims

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(dims, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, dims)
        )

    @property
    def dims(self):
        return self._dims

    def forward(self, x):
        return self.nn(x)


def score_matching_loss(score_model, x):
    x = x.requires_grad_(True)
    score = score_model(x)

    score_grad = [
        # We sum over the batch dimension to get element-wise gradients.
        torch.autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
        for i in range(score.shape[1])
    ]
    score_grad = torch.stack(score_grad, dim=1)

    loss = score_grad.sum(1) + 0.5 * score.pow(2).sum(1)

    return loss.mean()


def langevin_sampling(score_model, n_samples, batch_size, n_steps, tau, device):
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples-i)
            samples = torch.randn((bs, 2), device=device)
            for _ in range(n_steps):
                noise = torch.randn_like(samples, device=device) * np.sqrt(2 * tau)
                samples = samples + tau * score_model(samples) + noise
            all_samples.append(samples)

    return torch.concat(all_samples, 0)


def compute_vector_field(score_model, plot_range, step, device):
    feature_x = np.arange(-plot_range, plot_range+step/2, step)
    feature_y = np.arange(-plot_range, plot_range+step/2, step)
    x, y = np.meshgrid(feature_x, feature_y)

    u, v = score_model(torch.tensor(np.stack([x, y], axis=-1).reshape(-1, 2)).float().to(device)).detach().cpu().numpy().T
    norm = np.linalg.norm(np.array((u, v)), axis=0)
    u = u / norm * step
    v = v / norm * step

    return x, y, u, v


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_samples = get_gmm_samples(200)
    n_samples = 2000
    data_samples = spiral_samples(n_samples)


    score_model = ScoreModel(data_samples.shape[1])
    score_model.to(device)

    dataset = TensorDataset(data_samples)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=2e-4, weight_decay=0.001)

    num_epochs = 5000
    pbar = tqdm(range(num_epochs))
    for _ in pbar:
        losses = []
        for batch in dataloader:
            batch = batch[0]
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = score_matching_loss(score_model, batch)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        avg_loss = np.mean(losses)
        pbar.set_postfix({"loss": avg_loss})


    model_samples = langevin_sampling(score_model, 500, 64, 2000, 0.001, device)
    model_samples = model_samples.detach().cpu().numpy()

    step = 0.1
    plot_range = 2.0


    _, axes = plt.subplots(1, 3)

    for ax in axes:
        ax.set_aspect(1)
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)

    axes[0].scatter(data_samples[:, 0], data_samples[:, 1], alpha=0.5)
    axes[1].scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5)

    x, y, u, v = compute_vector_field(score_model, plot_range, step, device)
    axes[2].quiver(x, y, u, v, units='xy', scale=1.1, color='gray')

    plt.show()


if __name__ == "__main__":
    main()