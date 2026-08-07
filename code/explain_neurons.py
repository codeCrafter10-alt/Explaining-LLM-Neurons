import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.metrics import jaccard_score

from settings import *
from formula import Leaf, And, Or, Not

GLOBALS = {}


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



def get_formula_mask(formula, features):
    if isinstance(formula, Leaf):
        return features[:, formula.val]


    elif isinstance(formula, And):
        return (
            get_formula_mask(formula.left, features) & get_formula_mask(formula.right, features))


    elif isinstance(formula, Or):
        return (
            get_formula_mask(formula.left, features) | get_formula_mask(formula.right, features))


    elif isinstance(formula, Not):
        return ~get_formula_mask(formula.val, features)


    else:
        raise ValueError(
            "Unknown formula"
        )

def compute_iou(formula, neuron_mask, features):

    formula_mask = get_formula_mask(
        formula,
        features
    )

    if formula_mask.sum() == 0:
        return 0

    return jaccard_score(
        neuron_mask,
        formula_mask
    )

def score_formula(formula, neuron_mask, features):

    iou = compute_iou(
        formula,
        neuron_mask,
        features
    )

    return (
        COMPLEXITY_PENALTY ** (len(formula)-1)
    ) * iou

def formula_name(formula):

    feature_names = GLOBALS["feature_names"]

    return formula.to_str(
        lambda x: feature_names[x]
    )



def explain_single_neuron(neuron):

    binary_acts = GLOBALS["binary_acts"]
    features = GLOBALS["features"]

    neuron_activation = binary_acts[:, neuron]


    candidates = {}
    for i in range(features.shape[1]):

        f = Leaf(i)

        score = score_formula(
            f,
            neuron_activation,
            features
        )

        candidates[f] = score


    candidates = dict(
        sorted(
            candidates.items(),
            key=lambda x:x[1],
            reverse=True
        )[:BEAM_SIZE]
    )

    for step in range(MAX_FORMULA_LENGTH - 1):
        new_candidates = {}
        formulas = list(candidates.keys())


        for f1 in formulas:
            for f2 in formulas:
                for op in [And, Or]:
                    new_formula = op(f1, f2)
                    score = score_formula(new_formula, neuron_activation, features)
                    new_candidates[new_formula] = score

        candidates.update(
            new_candidates
        )


        candidates = dict(
            sorted(
                candidates.items(),
                key=lambda x:x[1],
                reverse=True
            )[:BEAM_SIZE]
        )


    best_formula, best_score = max(
        candidates.items(),
        key=lambda x:x[1]
    )


    return {
        "neuron": neuron,
        "formula": formula_name(best_formula),
        "iou": best_score
    }


def explain_neurons(activations, features, feature_names):
    print("Thresholding activations...")

    binary_acts = threshold_activations(activations)


    if NEURONS is None:
        neurons = list(range(binary_acts.shape[1]))

    else:
        neurons = NEURONS


    GLOBALS["binary_acts"] = binary_acts
    GLOBALS["features"] = features
    GLOBALS["feature_names"] = feature_names


    print(f"Searching {len(neurons)} neurons")


    with mp.Pool(PARALLEL) as pool:
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

    output = (f"{RESULTS_DIR}/neuron_explanations.csv")

    df.to_csv(output, index=False)

    print(
        df.sort_values(
            "iou",
            ascending=False
        ).head(20)
    )

if __name__ == "__main__":
    main()