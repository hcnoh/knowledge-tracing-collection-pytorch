import os
import argparse
import json
import pickle

import torch

from data_loaders.assistments import AssistmentsLoader
from models.dkt import DKT


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

    print(train_config)
    aucs, loss_means = \
        model.train_model(loader.questions, loader.responses, train_config)

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
