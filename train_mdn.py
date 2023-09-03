# script for training the MDN model and calculating luminance statistics for further use. 
# note that in dimmers/ directory, there are already pretrained models from this script.

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
from random import randint
from math import isnan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

START_IDX = 40
TRAIN_SIZE = 8
VAL_IMG_INDEX = 9 # never used in training so we can use it for validation (validation is not very important though because we are taking always last version of the model)


def retinex_decomposition(img):
    R = img / (img.sum(axis=2, keepdims=True) + 1e-6)
    L = (img / (3 * R + 1e-6)).max(axis=2, keepdims=True)
    return R, L

# prepare dataset
LOL_PATH = 'data/LOL'

# load LOL images
lol_images = list(sorted(Path(f'{LOL_PATH}/our480/high/').glob('*.png'), key=lambda x: int(x.stem)))
print(lol_images[START_IDX:START_IDX+TRAIN_SIZE])

def load_pair(idx):
    light = cv2.imread(str(lol_images[idx]))[..., ::-1]
    dark = cv2.imread(f'{LOL_PATH}/our480/low/{lol_images[idx].name}')[..., ::-1]

    # prepare dataset
    R_l, L_l = retinex_decomposition(light)
    R_d, L_d = retinex_decomposition(dark)

    # concatenate R_l, L_l and L_d on axis=2
    x = np.concatenate((R_l, L_l, L_d), axis=2)

    y = R_d

    return x, y

x, y = [], []
for i in tqdm(range(START_IDX, START_IDX+TRAIN_SIZE)):
    x_, y_ = load_pair(i)
    x.append(x_)
    y.append(y_)

x_, y_ = load_pair(VAL_IMG_INDEX)
x.append(x_)
y.append(y_)

# stack them into new axis
x = np.stack(x, axis=0)
y = np.stack(y, axis=0)

# permute last axis to second
x = np.moveaxis(x, -1, 1)
y = np.moveaxis(y, -1, 1)

# convert to torch and initialize dataloader
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

x[:, 3] = x[:, 3] / 255
x[:, 4] = x[:, 4] / 255

dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x[:-1], y[:-1]),
    batch_size=1,
    shuffle=True,
)

dl_val = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x[-1:], y[-1:]),
    batch_size=1,
    shuffle=True,
)

class MDN(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        latent_dim = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(5, latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
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
        red_mu = self.red_mu(latent) + x[:, 0, ...].unsqueeze(1)
        red_sigma = self.red_sigma(latent)
        red_pi = self.red_pi(latent)

        green_mu = self.green_mu(latent) + x[:, 1, ...].unsqueeze(1)
        green_sigma = self.green_sigma(latent)
        green_pi = self.green_pi(latent)

        blue_mu = self.blue_mu(latent) + x[:, 2, ...].unsqueeze(1)
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


def test_model(model):
    # read image
    l = cv2.imread(str(lol_images[VAL_IMG_INDEX]))[..., ::-1] / 255.0
    d = cv2.imread(f'{LOL_PATH}/our480/low/{lol_images[VAL_IMG_INDEX].name}')[..., ::-1] / 255.0
    # l = cv2.imread(f'{LOL_PATH}/val5/high/45.png')[..., ::-1] / 255.0
    # d = cv2.imread(f'{LOL_PATH}/val5/low/45.png')[..., ::-1] / 255.0

    # preprocess image
    R_l, L_l = retinex_decomposition(l)
    R_d, L_d = retinex_decomposition(d)

    # concatenate R_l, L_l and L_d on axis=2
    x = np.concatenate((R_l, L_l, L_d), axis=2)

    # convert to torch
    x = torch.from_numpy(x).float()

    # reshape to (h*w, 5)
    x = rearrange(x, 'h w c -> 1 c h w')

    # sample colors
    model.eval()
    colors = model.sample(x.to(device)).cpu()

    # convert to numpy
    colors = colors.numpy()

    # reshape to (h, w, 3)
    colors = rearrange(colors, '1 c h w -> h w c')

    # add
    colors = colors # + R_l
    colors = np.maximum(colors, 0)
    colors = colors / (np.sum(colors, axis=-1, keepdims=True) + 1e-8)
    # print(colors)
    # convert to uint8
    # colors = (colors * 255).astype(np.uint8)

    # plot image and R_d for comparison
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(colors)
    plt.subplot(1, 2, 2)
    plt.imshow(R_d)
    plt.show()

    model.train()

model = MDN(4)
model = model.to(device)

def train():
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 1001):
        losses = []
        for i, (x, y) in enumerate(dl):
            rand_x = randint(0, 400)
            rand_y = randint(0, 200)
            x = x[:, :, rand_y:rand_y+200, rand_x:rand_x+200]
            y = y[:, :, rand_y:rand_y+200, rand_x:rand_x+200]
            loss = model.loss(x.to(device), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())

        # eval
        model.eval()
        for x, y in dl_val:
            loss = model.loss(x.to(device), y.to(device)).cpu().item()

        if epoch % 100 == 0:
            print(f'Epoch {epoch:3d} | Train loss: {np.array(losses).mean():.4f} | Test loss: {loss:.4f}')

    test_model(model)


train()


def estimate_luminance_mapping():
    def load_pair(idx):
        light = cv2.imread(str(lol_images[idx]))[..., ::-1]
        dark = cv2.imread(f'{LOL_PATH}/our480/low/{lol_images[idx].name}')[..., ::-1]

        # prepare dataset
        _, L_l = retinex_decomposition(light)
        _, L_d = retinex_decomposition(dark)

        return L_l, L_d

    x, y = [], []
    MAX_ITER = TRAIN_SIZE

    for i in tqdm(range(START_IDX, START_IDX+MAX_ITER)):
        x_, y_ = load_pair(i)
        x.append(x_)
        y.append(y_)

    # stack them into new axis
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    # calculate mean and std of decrease factor for each 255 value of pixel
    y_flat, x_flat = y.reshape(-1), x.reshape(-1)

    mean = np.zeros(256)
    std = np.zeros(256)

    for i in tqdm(range(1, 256)):
        idx = x_flat.astype(np.uint8) == i
        if np.any(idx):
            mean[i] = (y_flat[idx] / x_flat[idx]).mean()
            std[i] = (y_flat[idx] / x_flat[idx]).std()
        else:
            mean[i] = None
            std[i] = None
    
    for i in range(1, 256):
        if isnan(mean[i]):
            if (i > 0 and not isnan(mean[i-1])) and (i < len(mean)-1 and not isnan(mean[i+1])):
                mean[i] = (mean[i-1] + mean[i+1]) / 2
                std[i] = (std[i-1] + std[i+1]) / 2
            elif i > 0 and not isnan(mean[i-1]):
                mean[i] = mean[i-1]
                std[i] = std[i-1]
            elif i < len(mean) - 1 and not isnan(mean[i+1]):
                mean[i] = mean[i+1]
                std[i] = std[i+1]
            else:
                raise Exception('Cannot estimate mean and std')

    # average mean and std with window size 3
    mean = np.convolve(mean, np.ones(3) / 3, mode='same')
    std = np.convolve(std, np.ones(3) / 3, mode='same')
    # plot mean and std
    plt.plot(mean)
    plt.plot(std)
    plt.show()

    return mean, std

mean, std = estimate_luminance_mapping()

POSTFIX = {
    0: '',
    10: '-b',
    20: '-c',
    30: '-d',
    40: '-e',
}[START_IDX]

POSTFIX = '-new'

# create folder
Path(f'./dimmers/{TRAIN_SIZE}shot{POSTFIX}').mkdir(parents=True, exist_ok=True)

# save model 
torch.save(model.cpu().state_dict(), f'./dimmers/{TRAIN_SIZE}shot{POSTFIX}/mdn.pt')
np.save(f'./dimmers/{TRAIN_SIZE}shot{POSTFIX}/mean.npy', mean)
np.save(f'./dimmers/{TRAIN_SIZE}shot{POSTFIX}/std.npy', std)
