import argparse
import json

import numpy as np
import torch

from data_loaders.assistments import AssistmentsLoader
from models.dkt import DKT


def main(model_name):
    loader = AssistmentsLoader()

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    if model_name == "DKT":
        if torch.cuda.is_available():
            model = DKT(loader.num_q, **model_config).cuda()
        else:
            model = DKT(loader.num_q, **model_config)

    print(train_config)
    model.train_model(loader.questions, loader.responses, train_config)


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
