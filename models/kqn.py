import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class KQN(Module):
    def __init__(self, num_q, dim_v, dim_s, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.dim_v = dim_v
        self.dim_s = dim_s
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.dim_v)
        self.knowledge_encoder = LSTM(self.dim_v, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.dim_s)
        self.dropout_layer = Dropout()

        self.q_emb = Embedding(self.num_q, self.dim_v)
        self.skill_encoder = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.dim_v),
            ReLU()
        )

    def forward(self, q, r, qry):
        # Knowledge State Encoding
        x = q + self.num_q * r
        x = self.x_emb(x)
        h, _ = self.knowledge_encoder(x)
        ks = self.out_layer(h)
        ks = self.dropout_layer(ks)

        # Skill Encoding
        e = self.q_emb(qry)
        o = self.skill_encoder(e)
        s = o / torch.norm(o, p=2)

        p = torch.sigmoid((ks * s).sum(-1))

        return p

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

                p = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    p = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
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
