
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert non-overlapping image patches into tokens."""

    def __init__(self, in_channels=32, embed_dim=256, patch_size=16):
        super().__init__()

        self.patch_size = patch_size

        patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"Image dimensions {(H, W)} must be divisible "
                f"by patch_size={P}"
            )

        gh, gw = H // P, W // P

        # B,C,H,W -> B,gh,gw,C,P,P
        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, gh * gw, -1)

        tokens = self.projection(patches)

        return tokens, (gh, gw)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-normalized self-attention with residual connection
        h = self.norm1(x)
        x = x + self.attention(
            h, h, h, need_weights=False
        )[0]

        # Feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class FullTransformerDIP(nn.Module):
    """
    Transformer-only DIP generator.

    No Conv2d, ConvTranspose2d, CNN encoder,
    CNN bottleneck, or CNN decoder.

    Input:  (B, input_channels, H, W)
    Output: (B, output_channels, H, W)
    """

    def __init__(
        self,
        input_channels=32,
        output_channels=3,
        embed_dim=256,
        patch_size=16,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
        base_grid=16
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_channels = output_channels

        self.patch_embedding = PatchEmbedding(
            input_channels,
            embed_dim,
            patch_size
        )

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, base_grid, base_grid)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Predict every pixel within each output patch
        self.patch_head = nn.Linear(
            embed_dim,
            output_channels * patch_size * patch_size
        )

    def forward(self, x):
        B, _, H, W = x.shape
        P = self.patch_size

        tokens, (gh, gw) = self.patch_embedding(x)

        # Resize positional embedding to token grid
        pos = F.interpolate(
            self.pos_embed,
            size=(gh, gw),
            mode="bilinear",
            align_corners=False
        )

        pos = pos.flatten(2).transpose(1, 2)
        tokens = tokens + pos

        # Global self-attention across all image patches
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Predict pixel values for every patch
        patches = self.patch_head(tokens)

        patches = patches.reshape(
            B, gh, gw,
            self.output_channels, P, P
        )

        # B,gh,gw,C,P,P -> B,C,gh,P,gw,P
        patches = patches.permute(0, 3, 1, 4, 2, 5)

        # Reconstruct the full image
        out = patches.reshape(
            B, self.output_channels, H, W
        )

        return out
