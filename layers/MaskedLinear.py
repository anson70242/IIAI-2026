import torch
import torch.nn as nn

"""
Input shape: [Batch, Channel, Seq_Len]
Output shape: [Batch, Channel, Experts, Pred_Len]
"""

class MaskedLinear(nn.Module):
    """
    Parallel Multi-Scale Linear Layer.
    
    Instead of instantiating multiple nn.Linear layers for different look-back windows,
    this layer uses a single weight tensor with a binary mask to enforce different
    receptive fields (window sizes) for each 'expert'.
    """
    def __init__(self, seq_len, pred_len, window_sizes):
        """
        Args:
            seq_len (int): Length of the input sequence (S).
            pred_len (int): Length of the prediction sequence (P).
            window_sizes (list[int]): A list of look-back window sizes. 
                                      The length of this list determines the number of experts (M).
        """
        super(MaskedLinear, self).__init__()
        self.num_experts = len(window_sizes)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_sizes = window_sizes
        
        # 1. Weights and Bias
        # Shape: [Experts, Seq_Len, Pred_Len]
        # We maintain M independent linear layers in a single tensor.
        self.weights = nn.Parameter(torch.empty(self.num_experts, seq_len, pred_len))

        # Initialize weights using Xavier Uniform
        nn.init.xavier_uniform_(self.weights)
        
        # Shape: [Experts, Pred_Len]
        self.bias = nn.Parameter(torch.zeros(self.num_experts, pred_len))
        
        # 2. Create the Mask
        # Shape: [Experts, Seq_Len, 1] (Last dim is 1 for broadcasting)
        # The mask ensures that the i-th expert only sees the last window_sizes[i] time steps.
        mask = torch.zeros(self.num_experts, seq_len, 1)
        
        for i, w in enumerate(window_sizes):
            if w >= seq_len:
                # If window size covers the whole sequence, keep all weights.
                mask[i, :, :] = 1.0
            else:
                # Otherwise, only keep the weights for the last 'w' time steps.
                mask[i, -w:, :] = 1.0
        
        # Register as a buffer so it becomes part of the state_dict but is not trained.
        self.register_buffer('mask', mask)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channel, Seq_Len]
            
        Returns:
            torch.Tensor: Output tensor of shape [Batch, Channel, Experts, Pred_Len]
        """
        # Apply the mask to the weights.
        # This zeroes out weights corresponding to time steps outside the look-back window.
        masked_w = self.weights * self.mask
        
        # Perform parallel matrix multiplication using Einstein summation.
        # b: Batch, c: Channel, s: Seq_Len
        # m: Experts, p: Pred_Len
        # Input (bcs) * Weights (msp) -> Output (bcmp)
        out = torch.einsum('bcs, msp -> bcmp', x, masked_w)
        
        # Add bias (broadcasted across Batch and Channel dimensions)
        # Bias shape: [M, P] -> [1, 1, M, P]
        out = out + self.bias.unsqueeze(0).unsqueeze(0)
        
        return out