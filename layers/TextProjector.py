import torch
import torch.nn as nn

"""
Input shape: [Batch, Embedding_Dim]
Output shape: [Batch, Pred_Len, Enc_In]
"""

class TextProjector(nn.Module):
    """
    TextProjector: A simple MLP to project Text Embeddings into Time-Series Space.
    
    It maps the semantic embedding from the LLM (e.g., 384 dim) to the 
    forecast horizon shape (Pred_Len * Enc_In) to generate modulation parameters
    like Gamma (scaling) and Beta (shifting).
    """
    def __init__(self, input_dim, pred_len, enc_in, hidden_dim=512):
        """
        Args:
            input_dim (int): Dimension of the text embedding (e.g., 384 for SBERT).
            pred_len (int): Length of the prediction horizon.
            enc_in (int): Number of variates/channels in the time series.
            hidden_dim (int): Hidden dimension for the MLP.
        """
        super(TextProjector, self).__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        self.mlp = nn.Sequential(
            # First projection to hidden space
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # Final projection to flatten time-series space
            nn.Linear(hidden_dim, pred_len * enc_in)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Text embedding tensor of shape [Batch, Text_Dim]
            
        Returns:
            torch.Tensor: Modulation parameter of shape [Batch, Pred_Len, Enc_In]
        """
        # Pass through MLP
        out = self.mlp(x)
        
        # Reshape the output to match the time series prediction shape:
        # [Batch, Flattened_Dim] -> [Batch, Pred_Len, Channels]
        return out.view(x.shape[0], self.pred_len, self.enc_in)