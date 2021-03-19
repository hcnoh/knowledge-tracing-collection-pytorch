import numpy as np
import torch

from torch.nn import Module, Embedding
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.optim import Adam
from sklearn import metrics

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class DKVMN(Module):
    def __init__(self, num_q, dim_k, dim_v, N):
        super(DKVMN, self).__init__()
        self.num_q = num_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.N = N

        self.k_emb_layer = Embedding(self.num_q, self.dim_k)
        self.Mk = torch.Tensor(self.dim_k, self.N)
        self.Mv = torch.Tensor(self.N, self.dim_k)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_v)

    def forward(self, q, r):
        qr = q + self.num_q * r

        k = self.k_emb_layer(q)
        v = self.v_emb_layer(qr)
