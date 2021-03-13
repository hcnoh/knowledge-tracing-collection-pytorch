import numpy as np
import argparse
import json

from data_loaders.assistments import AssistmentsLoader
from models.dkt import DKT


def main(model_name):
    loader = AssistmentsLoader()

    with open("config.json") as f:
        train_config = json.load(f)[model_name]

    if model_name == "DKT":
        model = DKT(loader.num_q, **train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="DKT",
        help="The name of the model to train. The possible models are in [DKT]. \
            The default model is DKT."
    )

    main(parser.model_name)