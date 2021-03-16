import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from sklearn import metrics

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, LongTensor


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

        train_questions = np.array(questions[:train_idx], dtype=object)
        train_responses = np.array(responses[:train_idx], dtype=object)

        test_questions = [
            LongTensor(q).unsqueeze(-1) for q in questions[train_idx:]
        ]
        test_responses = [
            LongTensor(r).unsqueeze(-1) for r in responses[train_idx:]
        ]

        test_questions = pad_sequence(
            test_questions, padding_value=pad_val
        ).squeeze()
        test_responses = pad_sequence(
            test_responses, padding_value=pad_val
        ).squeeze()

        print(test_questions.shape)
        print(test_responses.shape)
        print(len(responses[train_idx:]))
        print(np.max([arr.shape for arr in responses[train_idx:]]))

        test_masks = (test_questions != pad_val)
        test_questions, test_responses = \
            test_questions * test_masks.long(), \
            test_responses * test_masks.long()

        test_delta = one_hot(test_questions[1:], self.num_q)
        test_targets = test_responses[1:]

        test_questions = test_questions[:-1]
        test_responses = test_responses[:-1]
        test_masks = test_masks[:-1]

        test_targets = torch.masked_select(test_targets, test_masks)

        opt = Adam(self.parameters(), learning_rate)

        aucs = []
        loss_means = []

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

                mask = (q != pad_val)
                q, r = q * mask.long(), r * mask.long()

                delta = one_hot(q[1:], self.num_q)
                target = r[1:]

                q = q[:-1]
                r = r[:-1]
                mask = mask[:-1]

                self.train()

                y = self(q, r)

                opt.zero_grad()
                loss = torch.masked_select(
                    binary_cross_entropy((y * delta).sum(-1), target.float()),
                    mask
                ).mean()
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().numpy())

            self.eval()

            test_y = (self(test_questions, test_responses) * test_delta)\
                .sum(-1)
            test_y = torch.masked_select(test_y, test_masks).detach()

            fpr, tpr, thresholds = metrics.roc_curve(
                test_targets.numpy(), test_y.numpy()
            )
            auc = metrics.auc(fpr, tpr)

            loss_mean = np.mean(loss_mean)

            print(
                "Epoch: {},   AUC: {},   Loss Mean: {}"
                .format(i, auc, loss_mean)
            )

            aucs.append(auc)
            loss_means.append(loss_mean)

        return aucs, loss_means
