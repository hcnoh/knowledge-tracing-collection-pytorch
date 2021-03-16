import os
import argparse
import json
import pickle

import numpy as np
import torch

from data_loaders.assistments import AssistmentsLoader
from models.dkt import DKT


def preprocess(loader, path, seq_len=200, pad_val=-1e+3):
    questions = []
    responses = []
    targets = []
    deltas = []

    for q, r in zip(loader.questions, loader.responses):
        q, r = np.array(q), np.array(r)
        i = 0
        while i + 1 + seq_len < len(q):
            questions.append(q[i:i + seq_len])
            responses.append(r[i:i + seq_len])
            targets.append(r[i + 1:i + 1 + seq_len])
            deltas.append(q[i + 1:i + 1 + seq_len])

            i += seq_len

        questions.append(
            np.concatenate(
                [
                    q[i:len(q) - 1],
                    np.array([pad_val] * (i + seq_len - len(q) + 1))
                ]
            )
        )
        responses.append(
            np.concatenate(
                [
                    r[i:len(r) - 1],
                    np.array([pad_val] * (i + seq_len - len(r) + 1))
                ]
            )
        )
        targets.append(
            np.concatenate(
                [
                    r[i + 1:len(r)],
                    np.array([pad_val] * (i + 1 + seq_len - len(r)))
                ]
            )
        )
        deltas.append(
            np.concatenate(
                [
                    q[i + 1:len(r)],
                    np.array([pad_val] * (i + 1 + seq_len - len(r)))
                ]
            )
        )

    questions = np.array(questions)
    responses = np.array(responses)
    targets = np.array(targets)
    deltas = np.array(deltas)
    masks = (questions != pad_val)

    questions, responses, targets, deltas = \
        questions * masks, responses * masks, targets * masks, deltas * masks

    return questions, responses, targets, deltas, masks


def main(model_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")

    ckpt_path = ".ckpts/{}/".format(model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    loader = AssistmentsLoader()

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    with open(ckpt_path + "model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    with open(ckpt_path + "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "DKT":
        if torch.cuda.is_available():
            model = DKT(loader.num_q, **model_config).cuda()
        else:
            model = DKT(loader.num_q, **model_config)

    preprocessed_dataset_path = \
        loader.dataset_dir + "preprocessed_dataset.pkl"
    if os.path.isfile(preprocessed_dataset_path):
        with open(preprocessed_dataset_path, "rb") as f:
            preprocessed_dataset = pickle.load(f)
    else:
        preprocessed_dataset = preprocess(loader, preprocessed_dataset_path)

        with open(preprocessed_dataset_path, "wb") as f:
            pickle.dump(preprocessed_dataset, f)

    questions = preprocessed_dataset[0]
    responses = preprocessed_dataset[1]
    targets = preprocessed_dataset[2]
    deltas = preprocessed_dataset[3]
    masks = preprocessed_dataset[4]

    print(np.concatenate(loader.responses).shape)

    print(questions.shape)
    print(responses.shape)
    print(targets.shape)
    print(masks.shape)

    print(loader.responses[0])
    print(responses[0])
    print(targets[0])
    print(masks[0])

    # print(a)

    print(train_config)
    aucs, loss_means = \
        model.train_model(
            questions, responses, targets, deltas, masks, train_config
        )

    with open(ckpt_path + "aucs.pkl", "wb") as f:
        pickle.dump(aucs, f)
    with open(ckpt_path + "loss_means.pkl", "wb") as f:
        pickle.dump(loss_means, f)

    torch.save(model.state_dict(), ckpt_path + "model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="DKT",
        help="The name of the model to train. The possible models are in [DKT]. \
            The default model is DKT."
    )
    args = parser.parse_args()

    main(args.model_name)
