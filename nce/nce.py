import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.mixture_same_family import MixtureSameFamily
from tqdm import tqdm

from common.toy_data import spiral_samples


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def metropolis(model, n, device):
    x = torch.randn((n, 2), device=device)

    for _ in range(2000):
        x_new = x + 0.01 * torch.randn((n, 2), device=device)
        alpha = torch.exp(model(x_new)) / torch.exp(model(x))
        accept = (torch.rand((n, 1), device=device) < alpha).float()

        x = accept * x_new + (1 - accept) * x

    return x.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_samples = spiral_samples(500)

    batch_size = 64

    data_set = TensorDataset(data_samples)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)

    noise_dist = D.Normal(torch.tensor([0.0, 0.0], device=device), 
                          torch.tensor([4.0, 4.0], device=device))

    model = Model()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    num_epochs = 10000

    running_loss = 0.0
    pbar = tqdm(range(num_epochs))
    for _ in pbar:
        for x in data_loader:
            x = x[0].to(device)
            optimizer.zero_grad()

            y = noise_dist.sample(torch.Size([batch_size])).detach().to(device)

            Gx = model(x) - noise_dist.log_prob(x)
            Gy = model(y) - noise_dist.log_prob(y)

            hx = torch.sigmoid(Gx)
            hy = torch.sigmoid(Gy)

            loss = -(torch.log(hx) + torch.log(1 - hy))
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            running_loss = 0.99 * running_loss + 0.01 * loss.item()
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})

    model_samples = metropolis(model, 500, device)
    model_samples = model_samples.cpu()
    
    _, axes = plt.subplots(1, 2)
    plot_range = 1.5
    for ax in axes:
        ax.set_aspect(1)
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
    
    axes[0].scatter(data_samples[:, 0], data_samples[:, 1])
    axes[1].scatter(model_samples[:, 0], model_samples[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
