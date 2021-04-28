import os
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assistments import AssistmentsDataset, DATASET_DIR
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.utils import collate_fn


def main(model_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")

    ckpt_path = ".ckpts/{}/".format(model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    dataset = AssistmentsDataset(seq_len)

    with open(ckpt_path + "model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    with open(ckpt_path + "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        if torch.cuda.is_available():
            model = DKT(dataset.num_q, **model_config).cuda()
        else:
            model = DKT(dataset.num_q, **model_config)
    elif model_name == "dkvmn":
        if torch.cuda.is_available():
            model = DKVMN(dataset.num_q, **model_config).cuda()
        else:
            model = DKVMN(dataset.num_q, **model_config)
    elif model_name == "sakt":
        if torch.cuda.is_available():
            model = SAKT(dataset.num_q, **model_config).cuda()
        else:
            model = SAKT(dataset.num_q, **model_config)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    if os.path.exists("{}train_dataset.pkl".format(DATASET_DIR)):
        with open("{}train_dataset.pkl".format(DATASET_DIR), "rb") as f:
            train_dataset = pickle.load(f)
        with open("{}test_dataset.pkl".format(DATASET_DIR), "rb") as f:
            test_dataset = pickle.load(f)
    else:
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size]
        )
        with open("{}train_dataset.pkl".format(DATASET_DIR), "wb") as f:
            pickle.dump(train_dataset, f)
        with open("{}test_dataset.pkl".format(DATASET_DIR), "wb") as f:
            pickle.dump(test_dataset, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = \
        model.train_model(
            train_loader, test_loader, num_epochs, learning_rate, opt
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
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkvmn, sakt]. \
            The default model is dkt."
    )
    args = parser.parse_args()

    main(args.model_name)
