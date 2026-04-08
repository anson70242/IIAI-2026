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
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Configuration
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.window_sizes = configs.window_sizes
        self.smoothing = configs.smoothing
        
        # ============================================================
        # Part A: Numerical Backbone (Backbone Reuse)
        # ============================================================
        self.numerical_backbone = DLinearBackbone(configs)

        # ============================================================
        # Part B: Textual Components (Encoding & Projection)
        # ============================================================
        # 1. Text Encoder: Frozen SBERT 
        self.text_encoder = SbertTextEncoder(model_name='BAAI/bge-m3', freeze=True)
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
        self.final_mixer = ConvMixer(
            channels=self.channels,
            num_experts=3, 
            smoothing=self.smoothing
        )
        

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                static_text=None, dynamic_text=None, skip_fusion=False, timemmd = False):
        """
        Args:
            x_enc: Numerical Input [Batch, Seq_Len, Channels]
            static_text: List[str] of dataset descriptions (Static Prompts)
            dynamic_text: List[str] of time-specific context (Dynamic Prompts)
            skip_fusion: Boolean, 若为 True 则只运行数值主干网络 (用于 Stage 1)
        """
        
        # 1. Base Forecast (y_base)
        y_base = self.numerical_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # --- [Stage 1 专用] 跳过融合层直接返回数值预测 ---
        if skip_fusion:
            return y_base
            
        # Fallback: Return numerical forecast if text is missing
        if static_text is None or dynamic_text is None:
            return y_base

        # 2. Text Encoding
        emb_static = self.text_encoder(static_text)
        emb_dynamic = self.text_encoder(dynamic_text)

        if timemmd:
            text_proj = self.mlp_dynamic_gamma(emb_dynamic)
            output = y_base + text_proj
            return output

        # 3. Projection
        gamma_s = self.mlp_static_gamma(emb_static)
        beta_s = self.mlp_static_beta(emb_static)
        
        gamma_d = self.mlp_dynamic_gamma(emb_dynamic)
        beta_d = self.mlp_dynamic_beta(emb_dynamic)

        # 4. Modulation
        y_s = (y_base * (1.0 + gamma_s)) + beta_s
        y_d = (y_base * (1.0 + gamma_d)) + beta_d

        # 5. Fusion
        final_stack = torch.stack([y_base, y_s, y_d], dim=1) 
        output = self.final_mixer(final_stack)

        return output