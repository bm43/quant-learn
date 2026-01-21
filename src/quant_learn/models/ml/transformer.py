import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class scaled_dot_product_attention(nn.Module):

    # attention formula:
    # scores = softmax (Q*K.T/sqrt(d_k)) * V
    # where:
    # X is input sequence
    # Q = B x X x W_q = B x (N x d_k) x (d_k x d_k) = B x N x d_k
    # K = B x X x W_k = B x (N x d_k) x (d_k x d_k) = B x N x d_k
    # V = B x X x W_v = B x (N x d_k) x (d_k x d_k) = B x N x d_k
    # d_k = model dimension = no. of embeddings per token
    # mask: B x N x 

    def __init__(self, dk: int):
        super().__init__()
        self.sqrt_dk = np.sqrt(dk)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # attention scores:
        scores = torch.bmm(Q, K.transpose(-2, -1)) / self.sqrt_dk # QK/dk
        # bmm vs matmul:
        # bmm is used when both are batches:
        # B x n x m times B x m x p = B x n x p
        # it's better to use matmul when batch x not batch
        # try to write explicit code: mm, bmm, matmul

        if mask is not None:
            reshaped_mask = mask.reshape(scores.shape)
            scores = scores.masked_fill(reshaped_mask, float('-inf'))

        attention_dist = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_dist, V)

        return context, attention_dist


class multi_head_attention(nn.Module):
    # there are h sets of QKV per transformer
    # mutlihead_attention = concat(head_1, ..., head_h) * W_o
    # where head_i = attention(Q_i, K_i, V_i)
    # implementing h parallel attentions

    def __init__(self, d_k: int = 512, h: int = 8, dropout: Optional[float] = 0.1):
        super().__init__() # call init of parent class
        assert d_k % h == 0, "d_k % h isn't 0"

        self.d_h = d_k // h
        self.h = h
        self.d_k = d_k
        self.sdpa = scaled_dot_product_attention(self.d_h)
        # self.sdpa = scaled_dot_product_attention(self.d_h, dropout)

        self.W_q = nn.Linear(self.d_k, self.d_k)
        self.W_k = nn.Linear(self.d_k, self.d_k)
        self.W_v = nn.Linear(self.d_k, self.d_k)
        self.W_o = nn.Linear(self.d_k, self.d_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B = V.size(0) # batch size
        
        # linear proj
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # view is more memory efficient reshape
        Q = Q.view(B, -1, self.h, self.d_h).transpose(1, 2)
        K = K.view(B, -1, self.h, self.d_h).transpose(1, 2)
        V = V.view(B, -1, self.h, self.d_h).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1) # creates dimension that is 1.

        context, attention_weights = self.sdpa(Q,K,V,mask) # calls __call__ method of nn.Module
        # attention weights - B x h x N x N
        context = context.transpose(1, 2).contiguous().view(B,-1,self.d_k)

        # output
        output = self.W_o(context)
        
        return output