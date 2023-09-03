import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import albumentations as A
from einops import rearrange


def retinex_decomposition(img):
    R = img / (img.sum(axis=2, keepdims=True) + 1e-6)
    L = (img / (3 * R + 1e-6))[:, :, 0]
    return R, np.expand_dims(L, -1)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        latent_dim = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(5, latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(latent_dim, 3, kernel_size=1, padding=0)
        )

    def forward(self, x):
        latent = self.backbone(x)
        return latent + x[:, :3, ...]

    def sample(self, x):
        return self.forward(x).detach()

    def loss(self, x, y):
        out = self.forward(x)
        return self.loss_fn(out, y)


class MDN(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        latent_dim = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(5, latent_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.red_mu = nn.Conv2d(latent_dim, K, kernel_size=1)
        self.red_sigma = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softplus())
        self.red_pi = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softmax())

        self.green_mu = nn.Conv2d(latent_dim, K, kernel_size=1)
        self.green_sigma = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softplus())
        self.green_pi = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softmax())

        self.blue_mu = nn.Conv2d(latent_dim, K, kernel_size=1)
        self.blue_sigma = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softplus())
        self.blue_pi = nn.Sequential(nn.Conv2d(latent_dim, K, kernel_size=1), nn.Softmax())

    def forward(self, x):
        latent = self.backbone(x)
        red_mu = self.red_mu(latent) + 1*x[:, 0, ...].unsqueeze(1)
        red_sigma = self.red_sigma(latent)
        red_pi = self.red_pi(latent)

        green_mu = self.green_mu(latent) + 1*x[:, 1, ...].unsqueeze(1)
        green_sigma = self.green_sigma(latent)
        green_pi = self.green_pi(latent)

        blue_mu = self.blue_mu(latent) + 1*x[:, 2, ...].unsqueeze(1)
        blue_sigma = self.blue_sigma(latent)
        blue_pi = self.blue_pi(latent)

        return red_mu, red_sigma, red_pi, green_mu, green_sigma, green_pi, blue_mu, blue_sigma, blue_pi

    def sample_channel(self, mu, sigma, pi):
        pi = torch.distributions.Categorical(pi).sample()
        mu = mu[torch.arange(mu.shape[0]), pi]
        sigma = sigma[torch.arange(sigma.shape[0]), pi]
        return torch.distributions.Normal(mu, sigma).sample()

    def sample(self, x):
        height, width = x.shape[2:]
        red_mu, red_sigma, red_pi, green_mu, green_sigma, green_pi, blue_mu, blue_sigma, blue_pi = self.forward(x)

        red = rearrange(self.sample_channel(
            rearrange(red_mu, 'b c h w -> (b h w) c'),
            rearrange(red_sigma, 'b c h w -> (b h w) c'),
            rearrange(red_pi, 'b c h w -> (b h w) c')
        ), '(b h w) -> b h w', b=x.shape[0], h=height, w=width)

        green = rearrange(self.sample_channel(
            rearrange(green_mu, 'b c h w -> (b h w) c'),
            rearrange(green_sigma, 'b c h w -> (b h w) c'),
            rearrange(green_pi, 'b c h w -> (b h w) c')
        ), '(b h w) -> b h w', b=x.shape[0], h=height, w=width)

        blue = rearrange(self.sample_channel(
            rearrange(blue_mu, 'b c h w -> (b h w) c'),
            rearrange(blue_sigma, 'b c h w -> (b h w) c'),
            rearrange(blue_pi, 'b c h w -> (b h w) c')
        ), '(b h w) -> b h w', b=x.shape[0], h=height, w=width)

        return torch.stack((red, green, blue), dim=1)

    def loglik(self, mu, sigma, pi, y):
        z_score = (y - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            -torch.sum(torch.log(sigma), dim=-1)
        )

        output = torch.logsumexp(torch.log(pi) + normal_loglik.unsqueeze(0), dim=-1)
        return -output

    def loss(self, x, y):
        red_mu, red_sigma, red_pi, green_mu, green_sigma, green_pi, blue_mu, blue_sigma, blue_pi = self.forward(x)

        loglik_red = self.loglik(
            rearrange(red_mu, 'b c h w -> (b h w) c 1'), 
            rearrange(red_sigma, 'b c h w -> (b h w) c 1'),
            rearrange(red_pi, 'b c h w -> (b h w) c'),
            rearrange(y[:, 0, ...], 'b h w -> (b h w) 1 1')
        )

        loglik_green = self.loglik(
            rearrange(green_mu, 'b c h w -> (b h w) c 1'),
            rearrange(green_sigma, 'b c h w -> (b h w) c 1'),
            rearrange(green_pi, 'b c h w -> (b h w) c'),
            rearrange(y[:, 1, ...], 'b h w -> (b h w) 1 1')
        )

        loglik_blue = self.loglik(
            rearrange(blue_mu, 'b c h w -> (b h w) c 1'),
            rearrange(blue_sigma, 'b c h w -> (b h w) c 1'),
            rearrange(blue_pi, 'b c h w -> (b h w) c'),
            rearrange(y[:, 2, ...], 'b h w -> (b h w) 1 1')
        )

        return torch.mean(loglik_red + loglik_green + loglik_blue)


class MDNTransform:
    def __init__(
                self, 
                dim_factor=(0.3, 2), 
                transforms=None, 
                path='dimmers/8shot',
                mdn=True
            ):
        self.transforms = transforms
        self.dim_factor = dim_factor
        self.refl_model = MDN(4) if mdn else DCNN()
        self.refl_model.load_state_dict(torch.load(f'{path}/{"mdn" if mdn else "dcnn"}.pt'))
        self.refl_model.eval()

        self.means = np.load(f'{path}/mean.npy')
        self.stds = np.load(f'{path}/std.npy')

    def manipulate_reflectance(self, R, L, L_dimmed, dim_factor):
        x = np.concatenate((R, L, L_dimmed), axis=2)
        x = torch.from_numpy(x).float()
        x = rearrange(x, 'h w c -> 1 c h w')
        colors = self.refl_model.sample(x).numpy()
        colors = rearrange(colors, '1 c h w -> h w c')
        colors = colors + max(dim_factor - 1.2, 0)*R 
        colors = np.maximum(colors, 0)
        colors = colors / (np.sum(colors, axis=-1, keepdims=True) + 1e-8)
        return colors

    def dim_luminance(self, L, dim_factor, means, stds):
        L_int = (255*L).astype(np.uint16)
        std = np.random.normal(0, 0.05, L_int.shape)
        local_means = means[L_int]
        local_stds = stds[L_int]
        phi = np.clip(dim_factor*local_means + std*local_stds, 0, 1)
        L_dimmed = phi * L
        L_dimmed = np.clip(L_dimmed, 0, 1)
        return L_dimmed 

    def __call__(self, light):
        if self.transforms:
            light = self.transforms(image=light)["image"]
        light = light / 255.0

        R, L = retinex_decomposition(light)
        if type(self.dim_factor) == float:
            rng_dim_factor = self.dim_factor
        else:
            rng_dim_factor = np.random.uniform(self.dim_factor[0], self.dim_factor[1])
        L_dimmed = self.dim_luminance(L, rng_dim_factor, self.means, self.stds)
        R_dimmed = self.manipulate_reflectance(R, L, L_dimmed, rng_dim_factor)

        dark = (255 * 3 * L_dimmed * R_dimmed).astype(np.uint8)

        # histogram equalization
        hist = A.augmentations.functional.equalize(dark) / 255.0

        # color mapping
        c_map = dark / (dark.sum(axis=2, keepdims=True) + 1e-4)

        # normalize
        dark = dark / 255.0

        # compute lightness
        source_lightness = retinex_decomposition(dark)[1].mean()
        target_lightness = retinex_decomposition(light)[1].mean()

        # concatenate all images to a single tensor
        dark = np.concatenate([dark, hist, c_map, L_dimmed], axis=2)

        dark = dark.transpose(2, 0, 1)
        light = light.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(dark).float(),
            "target": torch.from_numpy(light).float(),
            "source_lightness": torch.tensor(source_lightness).float(),
            "target_lightness": torch.tensor(target_lightness).float(),
        }