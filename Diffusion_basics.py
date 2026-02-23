import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def gaussian_noise(img, alpha):

    noise = np.random.randn(*img.shape)

    # img = img + t * noise
    img = np.sqrt(alpha) * img + np.sqrt(1 - alpha) * noise * 255

    return np.clip(img, 0, 255)


def test_noise():
    img = np.array(Image.open("data/gear5.jpg"))

    noisy_img = gaussian_noise(img, 0.1)

    out_img = Image.fromarray(noisy_img.astype(np.uint8))
    out_img.save("out.jpg")


def sinusoidal_embedding(t, dim=256):
    """t: (B,) int tensor → (B, dim) float tensor"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t))


class MyResBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim=256, num_groups=8):
        super().__init__()

        self.group_norm1 = nn.GroupNorm(num_groups, in_c)
        self.group_norm2 = nn.GroupNorm(num_groups, out_c)
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.conv3x3_1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c)

        self.res_conv = (
            nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        )

    def forward(self, x, t):
        res = self.res_conv(x)
        x = self.group_norm1(x)
        x = self.silu1(x)
        x = self.conv3x3_1(x)

        t = self.t_proj(t)[:, :, None, None]
        x = x + t

        x = self.group_norm2(x)
        x = self.silu2(x)
        x = self.conv3x3_2(x)
        x = x + res

        return x


class SpatialAttention(nn.Module):
    def __init__(self, C, num_heads=4, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, C)
        self.attn = nn.MultiheadAttention(C, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = self.norm(x)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, T, C)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x + res


class MyUNet(nn.Module):
    def __init__(self, C=64, t_dim=256):
        super().__init__()
        self.t_emb = TimestepEmbedding(t_dim)
        self.conv_initial = nn.Conv2d(3, C, kernel_size=3, padding=1)

        self.res_block1 = MyResBlock(C, C, t_dim)
        self.down1 = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)

        self.res_block2 = MyResBlock(C * 2, C * 2, t_dim)
        self.down2 = nn.Conv2d(C * 2, C * 4, kernel_size=3, stride=2, padding=1)

        self.res_block3 = MyResBlock(C * 4, C * 4, t_dim)

        self.bot_res_block1 = MyResBlock(C * 4, C * 4, t_dim)
        self.bot_attn = SpatialAttention(C * 4)
        self.bot_res_block2 = MyResBlock(C * 4, C * 4, t_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_block4 = MyResBlock(C * 4 + C * 2, C * 2, t_dim)  # cat(x, skip2)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_block5 = MyResBlock(C * 2 + C, C, t_dim)  # cat(x, skip1)

        self.conv_final = nn.Conv2d(C, 3, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.t_emb(t)  # t is (B,)

        x = self.conv_initial(x)

        skip1 = self.res_block1(x, t_emb)  # (B, C, 32, 32)
        x = self.down1(skip1)  # (B, 2C, 16, 16)

        skip2 = self.res_block2(x, t_emb)  # (B, 2C, 16, 16)
        x = self.down2(skip2)  # (B, 4C, 8, 8)

        x = self.res_block3(x, t_emb)  # (B, 4C, 8, 8)

        x = self.bot_res_block1(x, t_emb)
        x = self.bot_attn(x)
        x = self.bot_res_block2(x, t_emb)

        x = self.up1(x)  # (B, 4C, 16, 16)
        x = self.res_block4(torch.cat([x, skip2], dim=1), t_emb)  # in: 6C -> out: 2C

        x = self.up2(x)  # (B, 2C, 32, 32)
        x = self.res_block5(torch.cat([x, skip1], dim=1), t_emb)  # in: 3C -> out: C

        return self.conv_final(x)  # (B, 3, 32, 32)


"""
for i in range (n_steps):
    batch = sample from dataset (batch_size)

    noised_batch, noise = add noise (batch)
    
    optimizer.zero_grad()

    pred = model.forward(noised_batch)

    loss = mse(pred, noise)

    
    loss.backward()
    
    optimizer.step()


"""


def make_ddpm_schedule(T, device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alphabars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphabars


def add_noise(x, alphabars, T):
    B = x.shape[0]
    t = torch.randint(0, T, (B,), device=x.device)
    abar = alphabars[t].view(B, 1, 1, 1)
    noise = torch.randn_like(x)
    noisy = torch.sqrt(abar) * x + torch.sqrt(1 - abar) * noise
    return noisy, noise, t


class TrainUNet:
    def __init__(self, C=64, lr=2e-4, batch_size=64, device="cuda"):
        self.device = torch.device(device)
        self.model = MyUNet(C=C).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.T = 1000
        self.betas, self.alphas, self.alphabars = make_ddpm_schedule(
            self.T, self.device
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        print(
            f"Model params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
        )

    def train(self, n_epochs=100, save_path="diffusion_cifar10.pt"):
        for epoch in tqdm(range(n_epochs)):
            total_loss = 0
            for images, _ in self.dataloader:
                images = images.to(self.device)

                noisy, noise, t = add_noise(images, self.alphabars, self.T)

                self.optimizer.zero_grad()
                pred_noise = self.model(noisy, t)

                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            tqdm.write(f"Epoch {epoch + 1}/{n_epochs} | loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Saved to {save_path}")

    @torch.no_grad()
    def sample(self, n_samples=16, T=1000):
        device = self.device
        self.model.eval()

        betas, alphas, alphabars = make_ddpm_schedule(T, device)
        x = torch.randn(n_samples, 3, 32, 32, device=device)

        for t_val in reversed(range(T)):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            eps = self.model(x, t)

            abar_t = alphabars[t_val]
            a_t = alphas[t_val]
            b_t = betas[t_val]

            # predict x0
            x0 = (x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
            x0 = x0.clamp(-1, 1)

            # DDPM mean for p(x_{t-1} | x_t)
            if t_val > 0:
                abar_prev = alphabars[t_val - 1]
            else:
                abar_prev = torch.tensor(1.0, device=device)

            coef1 = torch.sqrt(abar_prev) * b_t / (1 - abar_t)
            coef2 = torch.sqrt(a_t) * (1 - abar_prev) / (1 - abar_t)
            mean = coef1 * x0 + coef2 * x

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(b_t)  # simple choice
                x = mean + sigma * noise
            else:
                x = mean

        return x

    def save_grid(self, samples, path="samples.png", nrow=4):
        # denormalize [-1,1] -> [0,255]
        samples = ((samples + 1) / 2).clamp(0, 1)
        samples = (samples * 255).byte().cpu().numpy()
        n = len(samples)
        nrow = min(nrow, n)
        ncol = (n + nrow - 1) // nrow
        H, W = 32, 32
        grid = np.zeros((ncol * H, nrow * W, 3), dtype=np.uint8)
        for i, img in enumerate(samples):
            r, c = i // nrow, i % nrow
            grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = img.transpose(1, 2, 0)
        Image.fromarray(grid).save(path)
        print(f"Saved {n} samples to {path}")


def main():
    trainer = TrainUNet()
    # model = MyUNet(C=64).to(device)
    trainer.model.load_state_dict(
        torch.load("diffusion_cifar10.pt", map_location=trainer.device)
    )

    samples = trainer.sample(n_samples=16, T=1000)
    trainer.save_grid(samples, "samples.png")
    # trainer.train(n_epochs=100)


if __name__ == "__main__":
    main()
