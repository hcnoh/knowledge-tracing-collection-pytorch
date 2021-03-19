import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, LongTensor


def match_seq_len(questions, responses, seq_len, pad_val=-1):
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len + 1 < len(q):
            preprocessed_questions.append(q[i:i + seq_len + 1])
            preprocessed_responses.append(r[i:i + seq_len + 1])

            i += seq_len + 1

        preprocessed_questions.append(
            np.concatenate(
                [
                    q[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q)))
                ]
            )
        )
        preprocessed_responses.append(
            np.concatenate(
                [
                    r[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q)))
                ]
            )
        )

    return preprocessed_questions, preprocessed_responses


def collate_fn(batch, pad_val=-1):
    questions = []
    responses = []
    targets = []
    deltas = []

    for q, r in batch:
        questions.append(LongTensor(q[:-1]))
        responses.append(LongTensor(r[:-1]))
        targets.append(FloatTensor(r[1:]))
        deltas.append(LongTensor(q[1:]))

    questions = pad_sequence(
        questions, batch_first=True, padding_value=pad_val
    )
    responses = pad_sequence(
        responses, batch_first=True, padding_value=pad_val
    )
    targets = pad_sequence(
        targets, batch_first=True, padding_value=pad_val
    )
    deltas = pad_sequence(
        deltas, batch_first=True, padding_value=pad_val
    )

    masks = (questions != pad_val) * (deltas != pad_val)

    questions, responses, targets, deltas = \
        questions * masks, responses * masks, targets * masks, \
        deltas * masks

    return questions, responses, targets, deltas, masks
