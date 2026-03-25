import torch
import torch.nn as nn

# Assuming these modules are in a 'layers' package or similar
from layers.Autoformer_EncDec import series_decomp
from layers.ConvMixer import ConvMixer
from layers.MaskedLinear import MaskedLinear

"""
Input shape: [Batch, Seq_Len, Channel]
Output shape: [Batch, Pred_Len, Channel]
"""

class Model(nn.Module):
    """
    Multi-Scale DLinear with Parallel Execution and CNN Aggregation.
    
    Architecture Flow:
    1. Series Decomposition (Trend vs Seasonal)
    2. Parallel MaskedLinear Layers (Multi-scale projection for both parts)
    3. Summation (Recombine Trend + Seasonal)
    4. ConvMixer (Fuse predictions from different look-back windows)
    """
    def __init__(self, configs):
        """
        Args:
            configs: Configuration object containing (seq_len, pred_len, enc_in, moving_avg, window_sizes).
        """
        super(Model, self).__init__()
        
        # Configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        # [NEW] 從 configs 讀取 window_sizes (由 argparse 傳入)
        self.window_sizes = configs.window_sizes 
        self.smoothing = configs.smoothing
        
        # 1. Series Decomposition
        # Uses moving average to separate trend and seasonality.
        # Ensure 'series_decomp' class handles padding correctly.
        self.decomposition = series_decomp(configs.moving_avg)
        
        # 2. Parallel Masked Linear Experts
        # We instantiate two separate layers: one for seasonal, one for trend.
        # They handle multiple look-back windows simultaneously via masking.
        self.parallel_seasonal = MaskedLinear(self.seq_len, self.pred_len, self.window_sizes)
        self.parallel_trend = MaskedLinear(self.seq_len, self.pred_len, self.window_sizes)
        
        # 3. Aggregation Layer
        # Fuses the outputs from multiple experts using 2D Convolution.
        self.mixer = ConvMixer(
            channels=self.channels, 
            num_experts=len(self.window_sizes), 
            smoothing=self.smoothing
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: Input tensor [Batch, Seq_Len, Channel]
            
        Returns:
            final_output: Prediction tensor [Batch, Pred_Len, Channel]
        """
        
        # Step 1: Decomposition
        # Input: [B, L_in, D] -> Out: [B, L_in, D], [B, L_in, D]
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        # Step 2: Permute for Linear Projection
        # MaskedLinear expects [Batch, Channel, Seq_Len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # Step 3: Parallel Projection (Experts)
        # Input: [B, D, L_in] -> Output: [B, D, M, P]
        # (M = num_experts, P = pred_len)
        seasonal_output = self.parallel_seasonal(seasonal_init)
        trend_output = self.parallel_trend(trend_init)
        
        # Recombine Seasonal and Trend components
        expert_outputs = seasonal_output + trend_output

        # Step 4: Prepare for Mixer
        # Current: [Batch, Channel, Experts, PredLen]
        # Target for ConvMixer: [Batch, Experts, PredLen, Channel]
        stacked_out = expert_outputs.permute(0, 2, 3, 1)
        
        # Step 5: Aggregation
        # Fuses M experts into a single prediction
        # Output: [B, P, D]
        final_output = self.mixer(stacked_out)
        
        return final_output