import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


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


class MyResBlock(nn.Module):
    def __init__(self, C=32, num_groups=8):
        super().__init__()

        self.group_norm1 = nn.GroupNorm(num_groups, C)
        self.group_norm2 = nn.GroupNorm(num_groups, C)
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.conv3x3_1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(256, C)

    def forward(self, x, t):
        res = x
        x = self.group_norm1(x)
        x = self.silu1(x)
        x = self.conv3x3_1(x)
        t_emb = self.t_proj(t)  # (B, C)
        t_emb = t_emb[:, :, None, None]  # (B, C, 1, 1) — broadcast over H,W
        x = x + t_emb

        x = self.group_norm2(x)
        x = self.silu2(x)
        x = self.conv3x3_2(x)
        x = x + res

        return x


class MyUNet(nn.Module):
    def __init__(self, H=32, W=32, B=32, C=32, num_groups=8):
        super().__init__()
        self.conv_initial = nn.Conv2d(3, C * 2, kernel_size=3, padding=1)

        self.res_block1 = MyResBlock(C * 2)
        self.down1 = nn.Conv2d(C * 2, C * 4, kernel_size=3, stride=2, padding=1)
        self.res_block2 = MyResBlock(C * 4)
        self.down2 = nn.Conv2d(C * 4, C * 8, kernel_size=3, stride=2, padding=1)
        self.res_block3 = MyResBlock(C * 8)

        self.bot_res_block1 = MyResBlock(C * 8)
        self.bot_attn = 0  # attention
        self.bot_res_block2 = MyResBlock(C * 8)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_block4 = MyResBlock(C * 12)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_block5 = MyResBlock(C * 14)
        # self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.res_block6 = MyResBlock(C * 4)

        self.conv_final = nn.Conv2d(C * 14, 3, kernel_size=1, padding=0)

    def forward(self, x, t):
        x = self.conv_initial(x)

        skip1 = self.res_block1(x, t)
        x = self.down1(skip1)
        skip2 = self.res_block2(x, t)
        x = self.down2(skip2)
        x = self.res_block3(x, t)

        x = self.bot_res_block1(x, t)
        # x = self.bot_attn(x)
        x = self.bot_res_block2(x, t)

        x = self.up1(x)
        x = self.res_block4(torch.cat([x, skip2], dim=1), t)
        x = self.up2(x)
        x = self.res_block5(torch.cat([x, skip1], dim=1), t)
        # x = self.up3(x)
        # x = self.res_block6(torch.cat([x, skip1], dim=1), t)

        x = self.conv_final(x)

        return x


def main():
    unet = MyUNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randn(4, 256)
    out = unet(x, t)
    print(out.shape)  # should be (4, 64, 32, 32)


if __name__ == "__main__":
    main()
