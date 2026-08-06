import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter

from dataset import load_dataset, prepare_data

from settings import ACTIVATIONS_FILE, DATA, RESULTS_DIR, N_SENTENCE_FEATS

def load_activations(path, pooling="max"):
    print("Loading activations...")

    acts = torch.load(path)

    pooled = []

    for x in acts:
        x = x.squeeze(0)

        if pooling == "max":
            x = x.max(dim=0).values
        elif pooling == "mean":
            x = x.mean(dim=0)
        else:
            raise ValueError(
                f"Unknown pooling {pooling}"
            )

        pooled.append(x)


    acts = torch.stack(pooled)

    print("Loaded and pooled activations")
    return acts

def load_snli(path, limit=None):
    df = load_dataset(path)
    dataset = prepare_data(df, limit=limit)

    return dataset

def build_features(examples):
    print("Building features...")

    counts = Counter()

    tokenized = []

    for example in examples:
        premise = set(example["premise"].lower().split())
        hypothesis = set(example["hypothesis"].lower().split())

        tokenized.append((premise, hypothesis))

        counts.update(premise)
        counts.update(hypothesis)

    vocab = [word for word, _ in counts.most_common(N_SENTENCE_FEATS)]

    feature_names = []
    for word in vocab:
        feature_names.append(f"word:{word}")

    feature_names += ["overlap", "premise_only", "hypothesis_only"]

    matrix = np.zeros((len(examples), len(feature_names)), dtype=np.bool_)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    for i, (premise, hypothesis) in enumerate(tokenized):
        for word in premise|hypothesis:
            if word in word_to_index:
                matrix[i, word_to_index[word]] = 1

        offset = len(vocab)

        if len(premise & hypothesis) > 0:
            matrix[i, offset] = 1
        if len(premise - hypothesis) > 0:
            matrix[i, offset + 1] = 1
        if len(hypothesis - premise) > 0:
            matrix[i, offset + 2] = 1

    print("Finshed building features")
    return feature_names, matrix


def main():
    acts = load_activations(ACTIVATIONS_FILE)
    examples = load_snli(DATA)

    feature_names, feature_matrix = build_features(examples)

    torch.save(
    {
        "activations": acts,
        "features": feature_matrix,
        "feature_names": feature_names,
        "premises": examples["premise"],
        "hypotheses": examples["hypothesis"],
        "labels": examples["label"],
    },
    f"{RESULTS_DIR}/analysis_data.pt",
)

    print("Saved analysis data")


if __name__ == "__main__":
    main()