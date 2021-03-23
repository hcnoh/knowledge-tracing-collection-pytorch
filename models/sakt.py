import numpy as np
import torch

from torch.nn import Module, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm


class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads):
        super(SAKT, self).__init__()

        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Embedding(self.n, self.d)

        self.attn = MultiheadAttention(self.d, self.num_attn_heads)
        self.attn_layer_norm = LayerNorm([self.n, self.d])

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Linear(self.d, self.d)
        )
        self.FFN_layer_norm = LayerNorm([self.n, self.d])

        self.pred = Linear(self.d, 1)

    def forward(self, q, r):
        x = q + self.num_q * r

        M = self.M(x)
        E = self.E(q)
        P = self.P.unsqueeze(1)

        M += P

        # mask should be added...

        S, attn_weights = self.attn(E, M, M)
        S = self.attn_layer_norm(S + M)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F))

        return p
