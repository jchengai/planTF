import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(-1, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(x).squeeze(-1)

        return loc, pi
