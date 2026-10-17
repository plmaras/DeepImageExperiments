import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels=128,
        num_heads=4,
        num_layers=2,
        base_size=16
    ):
        super().__init__()

        # Learnable 2D positional embeddings.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels, base_size, base_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Resize positional embeddings to the current feature map.
        pos = F.interpolate(
            self.pos_embed,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # B,C,H,W -> B,H*W,C
        tokens = (x + pos).flatten(2).transpose(1, 2)

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # B,H*W,C -> B,C,H,W
        x = tokens.transpose(1, 2).reshape(B, C, H, W)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        # Supports odd image dimensions too.
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DIPTransformerUNet(nn.Module):
    """
    CNN encoder + Transformer bottleneck + CNN decoder.

    Input:  (B, input_channels, H, W)
    Output: (B, output_channels, H, W)
    """

    def __init__(
        self,
        input_channels=32,
        output_channels=3,
        base_channels=32,
        num_heads=4,
        num_transformer_layers=2
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.enc1 = ConvBlock(input_channels, c1)
        self.down1 = nn.Conv2d(
            c1, c2, kernel_size=3, stride=2, padding=1
        )

        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(
            c2, c3, kernel_size=3, stride=2, padding=1
        )

        self.enc3 = ConvBlock(c3, c3)
        self.down3 = nn.Conv2d(
            c3, c4, kernel_size=3, stride=2, padding=1
        )

        self.bottleneck_conv = ConvBlock(c4, c4)

        self.transformer = TransformerBottleneck(
            channels=c4,
            num_heads=num_heads,
            num_layers=num_transformer_layers
        )

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.output = nn.Conv2d(
            c1, output_channels, kernel_size=1
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # CNN bottleneck followed by Transformer attention
        b = self.down3(e3)
        b = self.bottleneck_conv(b)
        b = b + self.transformer(b)

        # Decoder with skip connections
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Keep output unconstrained for DIP optimization.
        return self.output(d1)
