import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class DKT(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=False
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        qr = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(qr))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

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

                y = self(q, r)
                y = (y * one_hot(d, self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(t, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            for data in test_loader:
                q, r, t, d, m = data

                self.eval()

                y = self(q, r)
                y = (y * one_hot(d, self.num_q)).sum(-1)

                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(t, m)\
                    .detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean)

                print(
                    "Epoch: {},   AUC: {},   Loss Mean: {}"
                    .format(i, auc, loss_mean)
                )

                aucs.append(auc)
                loss_means.append(loss_mean)

        return aucs, loss_means
