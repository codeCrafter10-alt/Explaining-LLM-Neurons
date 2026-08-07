"""
Settings
"""

import os

CUDA = True
ALPHA = 0.01  # Use "None" to use ReLU threshold (i.e., > 0)
BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 50
COMPLEXITY_PENALTY = 0.9
TOPN = 5

# Choices: iou, precision, recall
METRIC = "iou"

EMBEDDING_NEIGHBORHOOD_SIZE = 5

NEURONS = list(range(50))
PARALLEL = 6

SHUFFLE = False
SAVE_EVERY = 10

# How many "maximally activating" open features to use, PER CATEGORY
MAX_OPEN_FEATS = 5
# Minimum number of activations to analyze a neuron
MIN_ACTS = 200

N_SENTENCE_FEATS = 2000  # how many of the most common sentence lemmas to keep

ACTIVATIONS_FILE = "activations.pt"
RESULTS_DIR = "results"

DATA = "data/snli_1.0/snli_1.0_dev.txt"
assert DATA.endswith(".txt")
VECPATH = "data/glove.6B.300d.txt"
EMBEDDING_DIM = 300

# Overridables
if "MTDISSECT_MAX_FORMULA_LENGTH" in os.environ:
    MAX_FORMULA_LENGTH = int(os.environ["MTDISSECT_MAX_FORMULA_LENGTH"])
if "MTDISSECT_MAX_OPEN_FEATS" in os.environ:
    MAX_OPEN_FEATS = int(os.environ["MTDISSECT_MAX_OPEN_FEATS"])
if "MTDISSECT_METRIC" in os.environ:
    METRIC = os.environ["MTDISSECT_METRIC"]

dbase = os.path.splitext(os.path.basename(DATA))[0]
RESULT = f"{RESULTS_DIR}/snli-mistral/formula_length_{MAX_FORMULA_LENGTH}"

print(RESULT)