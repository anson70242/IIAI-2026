import torch
import torch.nn as nn

from layers.ConvMixer import ConvMixer
from layers.TextProjector import TextProjector
from models.sbert import SbertTextEncoder
from models.LeMoLE_DLinear import Model as DLinearBackbone

"""
Input shape: [Batch, Seq_Len, Channel]
Output shape: [Batch, Pred_Len, Channel]
"""

class Model(nn.Module):
    """
    LeMoLE: Language-enhanced Multi-scale Linear Experts.
    
    Architecture:
    1. Numerical Backbone: Reuses LeMoLE_DLinear to generate a base forecast (y_base).
    2. Textual Backbone: Encodes Static/Dynamic prompts using frozen SBERT.
    3. Conditioning: Projects text embeddings into Gamma (scale) and Beta (shift) parameters.
    4. Fusion: Aggregates the Base (y_base), Static-Modulated (Ys), and Dynamic-Modulated (Yd) forecasts via ConvMixer.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Configuration
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        # [NEW] 從 configs 讀取 window_sizes (與 Backbone 一致)
        self.window_sizes = configs.window_sizes
        self.smoothing = configs.smoothing
        
        # ============================================================
        # Part A: Numerical Backbone (Backbone Reuse)
        # ============================================================
        # Initializes the complete DLinear model (Decomposition + Linear Experts + Mixer).
        # This ensures we don't duplicate code or re-compute decomposition logic.
        # [NEW] 傳入 configs，讓 Backbone 自動讀取裡面的 window_sizes
        self.numerical_backbone = DLinearBackbone(configs)

        # ============================================================
        # Part B: Textual Components (Encoding & Projection)
        # ============================================================
        # 1. Text Encoder: Frozen SBERT (e.g., 'all-MiniLM-L6-v2', dim=384)
        self.text_encoder = SbertTextEncoder(model_name='all-MiniLM-L6-v2', freeze=True)
        self.text_dim = self.text_encoder.sbert_dim

        # 2. Static Projectors: Map Static text -> Modulation parameters
        self.mlp_static_gamma = TextProjector(self.text_dim, self.pred_len, self.channels)
        self.mlp_static_beta = TextProjector(self.text_dim, self.pred_len, self.channels)
        
        # 3. Dynamic Projectors: Map Dynamic text -> Modulation parameters
        self.mlp_dynamic_gamma = TextProjector(self.text_dim, self.pred_len, self.channels)
        self.mlp_dynamic_beta = TextProjector(self.text_dim, self.pred_len, self.channels)

        # ============================================================
        # Part C: Final Fusion Layer
        # ============================================================
        # Mixes 3 expert streams: [Base Forecast, Static Modulated, Dynamic Modulated]
        self.final_mixer = ConvMixer(
            channels=self.channels,
            num_experts=3, 
            smoothing=self.smoothing
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                static_text=None, dynamic_text=None):
        """
        Args:
            x_enc: Numerical Input [Batch, Seq_Len, Channels]
            static_text: List[str] of dataset descriptions (Static Prompts)
            dynamic_text: List[str] of time-specific context (Dynamic Prompts)
        """
        
        # 1. Base Forecast (y_base): Execute backbone logic (Decomp -> Projection -> Mix)
        y_base = self.numerical_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Fallback: Return numerical forecast if text is missing
        if static_text is None or dynamic_text is None:
            return y_base

        # 2. Text Encoding: Convert prompts to embeddings [Batch, Text_Dim]
        emb_static = self.text_encoder(static_text)
        emb_dynamic = self.text_encoder(dynamic_text)

        # 3. Projection: Generate Gamma/Beta parameters [Batch, Pred_Len, Channels]
        gamma_s = self.mlp_static_gamma(emb_static)
        beta_s = self.mlp_static_beta(emb_static)
        
        gamma_d = self.mlp_dynamic_gamma(emb_dynamic)
        beta_d = self.mlp_dynamic_beta(emb_dynamic)

        # 4. Modulation: Apply Affine Transformation (Y = y_base * γ + β)
        # Allows text to scale (γ) or shift (β) the numerical curve
        y_s = (y_base * gamma_s) + beta_s
        y_d = (y_base * gamma_d) + beta_d

        # 5. Fusion: Stack and aggregate the three forecast streams
        # Input to mixer: [Batch, 3, Pred_Len, Channels]
        final_stack = torch.stack([y_base, y_s, y_d], dim=1) 
        output = self.final_mixer(final_stack)

        return output