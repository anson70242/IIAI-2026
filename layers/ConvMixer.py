import torch
import torch.nn as nn

"""
Input shape: [Batch, Experts, Length, Channel]
Output shape: [Batch, Length, Channel]
"""

class ConvMixer(nn.Module):
    def __init__(self, channels, num_experts, smoothing=False):
        """
        ConvMixer: Responsible for fusing prediction results from different Look-back Window Experts.
        """
        super(ConvMixer, self).__init__()
        self.smoothing = smoothing
        self.channels = channels
        self.num_experts = num_experts

        # 1. Define parameters dynamically based on the smoothing flag
        if self.smoothing:
            # Expert fusion + Temporal smoothing (M, 3)
            kernel_size = (self.num_experts, 3)
            padding = (0, 1)
        else:
            # Expert fusion only (M, 1)
            kernel_size = (self.num_experts, 1)
            padding = 0

        # 2. Instantiate the layer once using those parameters
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='replicate' # Safe to keep for both cases (no effect if padding=0)
            ),
            nn.GELU(),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        B, M, L, D = x.shape
        x = x.permute(0, 3, 1, 2) # [B, D, M, L]
        x = self.conv(x)          # [B, D, 1, L]
        x = x.permute(0, 2, 3, 1) # [B, 1, L, D]
        return x.squeeze(1)       # [B, L, D]