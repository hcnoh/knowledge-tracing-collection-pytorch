import numpy as np
import torch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    from torch.cuda import LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
    from torch import LongTensor


class DKT(torch.nn.Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = torch.nn.Embedding(
            self.num_q * 2, self.emb_size
        )
        self.rnn_layer = torch.nn.RNN(self.emb_size, self.hidden_size)
        self.out_layer = torch.nn.Linear(self.hidden_size, self.num_q)

    def forward(self, q, r):
        qr = q + self.num_q * r

        h = self.rnn_layer(self.interaction_emb(qr))
        y = torch.nn.functional.sigmoid(self.out_layer(h))

        return y

    def train(self, q, r):
        y = self(q, r)
        print(y)
