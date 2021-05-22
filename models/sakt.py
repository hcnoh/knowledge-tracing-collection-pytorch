import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super(SAKT, self).__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_layer_norm = LayerNorm([self.n, self.d])

        self.FFN = Sequential(
            Linear(self.d, self.d),
            Dropout(self.dropout),
            ReLU(),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm([self.n, self.d])

        self.pred = Sequential(
            Linear(self.d, 1),
            Dropout(self.dropout),
        )

    def forward(self, q, r, q_shifted):
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(q_shifted).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M += P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(
        self, train_loader, test_loader, num_epochs, learning_rate, opt
    ):
        aucs = []
        loss_means = []

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, t, d, m = data

                self.train()

                p, _ = self(q, r, d)
                p = torch.masked_select(p, m)
                t = torch.masked_select(t, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, t, d, m = data

                    self.eval()

                    p, _ = self(q, r, d)
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(t, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means
