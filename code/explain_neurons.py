import torch
import numpy as np
import pandas as pd

from sklearn.metrics import jaccard_score

from settings import *
import multiprocessing as mp

GLOBALS = {}

def explain_single_neuron(neuron):
    binary_acts = GLOBALS["binary_acts"]
    features = GLOBALS["features"]
    feature_names = GLOBALS["feature_names"]

    neuron_activation = binary_acts[:, neuron]

    best_score = 0
    best_feature = None

    for feature_id in range(features.shape[1]):
        score = compute_iou(
            neuron_activation,
            features[:, feature_id]
        )

        if score > best_score:
            best_score = score
            best_feature = feature_names[feature_id]

    return {
        "neuron": neuron,
        "feature": best_feature,
        "iou": best_score
    }

def load_analysis_data(path):
    print("Loading analysis data...")

    data = torch.load(
        path,
        weights_only=False
    )

    activations = data["activations"].numpy()
    features = data["features"]
    feature_names = data["feature_names"]

    print("Loaded analysis data")
    return activations, features, feature_names


def threshold_activations(activations):
    thresholds = np.quantile(
        activations,
        1 - ALPHA,
        axis=0
    )

    return activations > thresholds


def compute_iou(neuron_mask, feature_mask):
    if neuron_mask.sum() == 0:
        return 0

    return jaccard_score(
        neuron_mask,
        feature_mask
    )


def explain_neurons(activations, features, feature_names):
    print("Thresholding activations...")

    binary_acts = threshold_activations(activations)

    num_neurons = binary_acts.shape[1]

    if NEURONS is None:
        neurons = list(range(num_neurons))
    else:
        neurons = NEURONS


    GLOBALS["binary_acts"] = binary_acts
    GLOBALS["features"] = features
    GLOBALS["feature_names"] = feature_names


    print(
        f"Searching {len(neurons)} neurons..."
    )


    workers = PARALLEL

    with mp.Pool(workers) as pool:
        results = list(
            pool.imap(
                explain_single_neuron,
                neurons
            )
        )


    return results



def main():
    activations, features, feature_names = load_analysis_data(f"{RESULTS_DIR}/analysis_data.pt")

    results = explain_neurons(
        activations,
        features,
        feature_names
    )

    df = pd.DataFrame(results)

    output = (
        f"{RESULTS_DIR}/neuron_explanations.csv"
    )

    df.to_csv(output, index=False)

    print(f"Saved explanations to {output}")

    print(
        df.sort_values(
            "iou",
            ascending=False
        ).head(20)
    )

if __name__ == "__main__":
    main()