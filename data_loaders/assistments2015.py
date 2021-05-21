import os
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = ".datasets/assistments2015/"


class Assistments2015Dataset(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR):
        super(Assistments2015Dataset, self).__init__()
        self.seq_len = seq_len
        self.dataset_dir = dataset_dir

        self._csv_path = os.path.join(
            DATASET_DIR, "2015_100_skill_builders_main_problems.csv"
        )

        self._df = pd.read_csv(self._csv_path, encoding="ISO-8859-1")
        self._df = self._df[
            (self._df["correct"] == 0).values
            + (self._df["correct"] == 1).values
        ]

        self.num_u = np.unique(self._df["user_id"]).shape[0]
        self.num_q = np.unique(self._df["sequence_id"]).shape[0]

        if os.path.isfile(os.path.join(self.dataset_dir, "dataset.pkl")):
            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "rb"
            ) as f:
                self.questions, self.responses, self.user_list, \
                    self.user2idx, self.q_list, self.q2idx = pickle.load(f)
        else:
            self.questions, self.responses, self.user_list, \
                self.user2idx, self.q_list, self.q2idx = \
                self._get_questions_responses(self._df)

        if seq_len:
            self.questions, self.responses = \
                match_seq_len(self.questions, self.responses, seq_len)

        self.len = len(self.questions)

    def __getitem__(self, index):
        return self.questions[index], self.responses[index]

    def __len__(self):
        return self.len

    def _get_questions_responses(self, df):
        user_list = np.unique(df["user_id"].values)
        user2idx = {user_list[idx]: idx for idx, _ in enumerate(user_list)}

        q_list = np.unique(df["sequence_id"].values)
        q2idx = {q_list[idx]: idx for idx, _ in enumerate(q_list)}

        questions = []
        responses = []
        for user in user_list:
            user_data = \
                df[(df["user_id"] == user)].sort_values("log_id")

            question = \
                np.array([q2idx[q] for q in user_data["sequence_id"].values])
            # {0, 1, ..., num_q-1}, Cardinality = num_q
            response = user_data["correct"].values  # {0, 1}, Cardinality = 2

            questions.append(question)
            responses.append(response)

        with open(
            os.path.join(self.dataset_dir, "dataset.pkl"), "wb"
        ) as f:
            pickle.dump(
                (questions, responses, user_list, user2idx, q_list, q2idx), f
            )

        return questions, responses, user_list, user2idx, q_list, q2idx
