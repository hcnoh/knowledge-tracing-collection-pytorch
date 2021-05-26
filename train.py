import os
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assistments2009 import Assistments2009Dataset
from data_loaders.assistments2015 import Assistments2015Dataset
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.ssakt import SSAKT
from models.utils import collate_fn


def main(model_name, dataset_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")

    ckpt_path = ".ckpts/{}/".format(model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = ckpt_path + "/{}/".format(dataset_name)
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

    if dataset_name == "assistments2009":
        dataset = Assistments2009Dataset(seq_len)
    elif dataset_name == "assistments2015":
        dataset = Assistments2015Dataset(seq_len)

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
    elif model_name == "ssakt":
        if torch.cuda.is_available():
            model = SSAKT(dataset.num_q, **model_config).cuda()
        else:
            model = SSAKT(dataset.num_q, **model_config)
    else:
        print("The wrong model name was used...")
        return

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    if os.path.exists("{}train_indices.pkl".format(dataset.dataset_dir)):
        with open(
            "{}train_indices.pkl".format(dataset.dataset_dir), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open("{}test_indices.pkl".format(dataset.dataset_dir), "rb") as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(
            "{}train_indices.pkl".format(dataset.dataset_dir), "wb"
        ) as f:
            pickle.dump(train_dataset.indices, f)
        with open("{}test_indices.pkl".format(dataset.dataset_dir), "wb") as f:
            pickle.dump(test_dataset.indices, f)

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
        model.train_model(train_loader, test_loader, num_epochs, opt)

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
            The possible models are in [dkt, dkvmn, sakt, ssakt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="assistments2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in [assistments2009, assistments2015]. \
            The default dataset is assistments2009."
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)
