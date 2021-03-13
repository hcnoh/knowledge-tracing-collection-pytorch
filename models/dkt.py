import random

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    from torch.cuda import LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
    from torch import LongTensor


class DKT(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        qr = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(qr))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(self, questions, responses, train_config, pad_val=-1e+3):
        batch_size = train_config["batch_size"]
        num_epochs = train_config["num_epochs"]
        train_ratio = train_config["train_ratio"]
        learning_rate = train_config["learning_rate"]

        train_idx = int(len(questions) * train_ratio)

        train_questions = np.array(questions[:train_idx])
        train_responses = np.array(responses[:train_idx])
        test_questions = np.array(questions[train_idx:])
        test_responses = np.array(responses[train_idx:])

        opt = Adam(self.parameters(), learning_rate)

        for i in range(1, num_epochs):
            loss_mean = []
            for _ in range(train_idx // batch_size):
                random_indices = np.random.choice(
                    train_idx, batch_size, replace=False
                )

                q = train_questions[random_indices]
                r = train_responses[random_indices]

                q = [LongTensor(arr).unsqueeze(-1) for arr in q]
                r = [LongTensor(arr).unsqueeze(-1) for arr in r]

                q = pad_sequence(q, padding_value=pad_val).squeeze()
                r = pad_sequence(r, padding_value=pad_val).squeeze()

                mask = (q != pad_val).long()
                q, r = q * mask, r * mask

                delta = one_hot(q[1:], self.num_q)

                q = q[:-1]
                r = r[:-1]
                mask = mask[:-1]

                self.train()

                y = self(q, r)
                # y = self.dropout_layer(y)
                # print(y.shape, delta.shape)
                # print(y[0, 0, :] * delta[0, 0, :])

                # print((y * delta).sum(-1).shape, r.shape)

                opt.zero_grad()
                loss = binary_cross_entropy((y * delta).sum(-1), r.float())\
                    .mean()
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().numpy())

            loss_mean = np.mean(loss_mean)
            print(i, loss_mean)
