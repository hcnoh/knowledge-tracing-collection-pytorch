import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


def mlp(in_size, out_size):
    return Sequential(
        Linear(in_size, out_size),
        ReLU(),
        Dropout(),
        Linear(out_size, out_size),
    )


class GKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
            num_attn_heads:
                the number of the attention heads in the multi-head attention
                module in this model.
                This argument would be used when the method is MHA.

        Note that this implementation is not exactly the same as the original
        paper. The erase-add gate was not implemented since the reason for
        the gate can't be found. And the batch normalization in MLP was not
        implemented because of the simplicity.
    '''
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__()
        self.num_q = num_q
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.hidden_size)
        self.q_emb = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        kaiming_normal_(self.q_emb)

        self.init_h = Parameter(torch.Tensor(self.num_q, self.hidden_size))

        self.mlp_self = mlp(self.hidden_size * 2, self.hidden_size)

        self.gru = GRU(
            self.hidden_size * 2,
            self.hidden_size,
            batch_first=True
        )

        self.bias = Parameter(torch.Tensor(1, self.num_q, 1))
        self.out_layer = Linear(self.hidden_size, 1, bias=False)

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
                h: the hidden states of the all questions(KCs)
        '''
        batch_size = q.shape[0]

        x = q + self.num_q * r

        x_emb = self.x_emb(x)
        q_emb = self.q_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        q_onehot = one_hot(q, self.num_q)

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h = [ht]
        y = []

        for xt_emb, qt, qt_onehot in zip(
            x_emb.permute(1, 0, 2), q.permute(1, 0), q_onehot.permute(1, 0, 2)
        ):
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, ht)

            ht = self.update(ht, ht_, qt, qt_onehot)
            yt = self.predict(ht)

            h.append(ht)
            y.append(yt)

        h = torch.stack(h, dim=1)
        y = torch.stack(y, dim=1)

        return y, h

    def aggregate(self, xt_emb, qt_onehot, q_emb, ht):
        xt_emb = xt_emb.unsqueeze(1).repeat(1, self.num_q, 1)
        qt_onehot = qt_onehot.unsqueeze(-1)

        ht_ = qt_onehot * torch.cat([ht, xt_emb], dim=-1) + \
            (1 - qt_onehot) * torch.cat([ht, q_emb], dim=-1)

        return ht_

    def f_self(self, ht_):
        return self.mlp_self(ht_)

    def f_neighbor(self, ht_, qt):
        pass

    def update(self, ht, ht_, qt, qt_onehot):
        qt_onehot = qt_onehot.unsqueeze(-1)

        m = qt_onehot * self.f_self(ht_) + \
            (1 - qt_onehot) * self.f_neighbor(ht_, qt)

        ht, _ = self.gru(
            torch.cat([m, ht], dim=-1)
        )

        return ht

    def predict(self, ht):
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze()

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                y, _ = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    y, _ = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means


class PAM(GKT):
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__(num_q, hidden_size, num_attn_heads, method)

        self.A = Parameter(torch.Tensor(self.num_q, self.num_q))
        kaiming_normal_(self.A)

        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

    def f_neighbor(self, ht_, qt):
        batch_size = qt.shape[0]
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1])
        )

        Aij = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.A.shape[-1])
        ).squeeze()

        outgo_part = Aij.unsqueeze(-1) * \
            self.mlp_outgo(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            )

        Aji = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=2,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, self.A.shape[-1], 1)
        ).squeeze()

        income_part = Aji.unsqueeze(-1) * \
            self.mlp_income(
                torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
            )

        return outgo_part + income_part


class MHA(GKT):
    def __init__(self, num_q, hidden_size, num_attn_heads, method):
        super().__init__(num_q, hidden_size, num_attn_heads, method)
        self.num_attn_heads = num_attn_heads

        #############################################################
        # These definitions are due to a bug of PyTorch.
        # Please check the following link if you want to check details:
        # https://github.com/pytorch/pytorch/issues/27623
        # https://github.com/pytorch/pytorch/pull/39402
        self.Q = Linear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False
        )
        self.K = Linear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False
        )
        self.V = Linear(
            self.hidden_size * 4,
            self.hidden_size,
            bias=False
        )
        #############################################################

        self.mha = MultiheadAttention(
            self.hidden_size,
            self.num_attn_heads,
        )

    def f_neighbor(self, ht_, qt):
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1])
        )

        q = self.Q(tgt)
        k = self.K(src)
        v = self.V(
            torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
        )

        _, weights = self.mha(
            q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        )
        # Average attention weights over heads
        # https://github.com/pytorch/pytorch/blob/5fdcc20d8d96a6b42387f57c2ce331516ad94228/torch/nn/functional.py#L5257
        weights = weights.permute(0, 2, 1)

        return weights * v
