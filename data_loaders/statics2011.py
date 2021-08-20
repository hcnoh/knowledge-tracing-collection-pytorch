import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = ".datasets/statics2011/"


class Statics2011(Dataset):
    def __init__(self, seq_len, datset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.seq_len = seq_len

        self.dataset_dir = datset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, os.path.join(
                "ds507_tx_2021_0704_202856",
                "ds507_tx_All_Data_1664_2017_0227_034415.txt"
            )
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)

        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if self.seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, sep="\t")\
            .dropna(subset=["Problem Name", "Step Name", "Outcome"])\
            .sort_values(by=["Time"])
        df = df[df["Attempt At Step"] == 1]
        df = df[df["Student Response Type"] == "ATTEMPT"]

        kcs = []
        for _, row in df.iterrows():
            kcs.append("{}_{}".format(row["Problem Name"], row["Step Name"]))

        df["KC"] = kcs

        u_list = np.unique(df["Anon Student Id"].values)
        q_list = np.unique(df["KC"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []
        for u in u_list:
            u_df = df[df["Anon Student Id"] == u]

            q_seqs.append([q2idx[q] for q in u_df["KC"].values])
            r_seqs.append((u_df["Outcome"].values == "CORRECT").astype(int))

        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx
