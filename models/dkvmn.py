import numpy as np
import torch

from torch.nn import Module, Embedding, Linear
from torch.nn.functional import binary_cross_entropy
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
        self.Mv = torch.Tensor(self.N, self.dim_v)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_v)

        self.f_layer = Linear(self.dim_k * 2, self.dim_k)
        self.p_layer = Linear(self.dim_k, 1)

        self.e_layer = Linear(self.dim_v, self.dim_v)
        self.a_layer = Linear(self.dim_v, self.dim_v)

    def forward(self, q, r):
        qr = q + self.num_q * r
        Mvt = self.Mv.unsqueeze(0)

        p = []
        Mv = []

        for qt, qrt in zip(q.permute(1, 0), qr.permute(1, 0)):
            kt = self.k_emb_layer(qt)
            vt = self.v_emb_layer(qrt)

            wt = torch.softmax(torch.matmul(kt, self.Mk), dim=-1)

            # Read Process
            rt = torch.matmul(wt, self.Mv)
            ft = torch.tanh(self.f_layer(torch.cat([rt, kt], dim=-1)))
            pt = torch.sigmoid(self.p_layer(ft)).squeeze()

            # Write Process
            et = torch.sigmoid(self.e_layer(vt))
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1)))
            at = torch.tanh(self.a_layer(vt))
            Mvt = Mvt + (wt.unsqueeze(-1) * at.unsqueeze(1))

            p.append(pt)
            Mv.append(Mvt)

        p = torch.stack(p, dim=1)
        Mv = torch.stack(Mv, dim=1)

        return p, Mv

    def train_model(
        self, train_loader, test_loader, num_epochs, learning_rate, opt
    ):
        aucs = []
        loss_means = []

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, _, d, m = data

                self.train()

                p, _ = self(q, r)
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            auc = []
            weights = []
            for data in test_loader:
                q, r, t, d, m = data

                self.eval()

                p, _ = self(q, r)
                p = torch.masked_select(p, m).detach().cpu()
                t = torch.masked_select(r, m).float().detach().cpu()

                auc.append(
                    metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )
                )
                weights.append(torch.prod(torch.tensor(t.shape)).cpu().numpy())

            auc = np.average(auc, weights=weights)
            loss_mean = np.mean(loss_mean)

            print(
                "Epoch: {},   AUC: {},   Loss Mean: {}"
                .format(i, auc, loss_mean)
            )

            aucs.append(auc)
            loss_means.append(loss_mean)

        return aucs, loss_means