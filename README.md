# Explaining-LLM-Neurons
Repository for task 9 of neural explanations LLM - Explaining LLM Neurons

This project adapts the code from [Compositional Explanations's NLI implementation](https://github.com/jayelm/compexp/tree/master/nli) to generate compositional explanations for a fine-tuned Mistral-7B-v0.1 model on the NLI task. 

## Code/Implementation
I extracted activations from the final MLP layer of the fine-tuned Mistral model and searched for compositional logical formulas that best explained when individual neurons activated. Candidate formulas were evaluated using Intersection over Union (iou) values, and the highest-scoring formula for each neuron was selected as its explanation

- `activations.py`:
This file loads the fine-tuned Mistral model and tokenizer, hooks the penultimate layer, and extracts neuron activations for every example (10,000) in the SNLI Dev dataset. The activations are then saved in `activations.pt`

- `analyze.py`:
This file loads the saved activations from `activations.pt`, uses max pooling to convert token-level activations into one activation vector per example, constructs binary sentence features, and saves the data in `analysis_data.pt`

- `explain_neurons.py`:
This is the main file of the entire project as it generates and saves the compositional explanations. It loads the analysis data from `analysis_data.pt` and searches for logical formulas that explain each neuron. Using multiprocessing, for every neuron, it thresholds activations, computes activation masks, searches over all logical formulas and evaluates them using their iou values, and stores the explanation with the highest iou value in `results/neuron_explanations.csv`

- `formula.py`:
This file is copied over from [Compexp's formula.py](https://github.com/jayelm/compexp/blob/master/nli/code/formula.py). It contains the logical formula representations and is used in `explain_neurons.py`

- `dataset.py`:
This is a helper function that loads and preprocesses the SNLI dataset by removing invalid examples, converting labels to integers, and returning formatted examples

- `settings.py`:
Stores all configurables parameters including:
    - activation threshold
    - beam size
    - maximum formula length
    - multiprocessing settings (number of workers)
    - data and results path
    - Number of neurons to search for explanations


## Results
This project successfully generated compositional explanations for 50 neurons of the last MLP layer and stored it under `results/neuron_explanations.csv`.

For the analyzed neurons in the Mistral LLM model, most explanation formulas achieved very low iou values (between 0.05 and 0.2) as compared to the Bowman model. This shows that neurons are not very aligned with human-interpretable explanations. Additionally, [Compexp's research paper](https://arxiv.org/pdf/2006.14032) states that explanability and accuracy are negatively correlated, which justifies these results as this fine-tuned model achieved a higher accuracy than the Bowman model on the SNLI dataset.

Still, one neuron had a very strong explanation with an iou value of 0.729, suggesting that this neuron's activation were very aligned with human-interpretable features. Its specific explanation was `((word:competition AND word:competition.) OR (word:competition AND word:rain.))`. This means that the neuron almost always activates whenever "competition" is in the premise or hypothesis. 

See [`results/neuron_explanations.csv`](results/neuron_explanations.csv) for more neuron explanations.


## Challenges and Solutions
- Learning how to hook activations in a LLM model
    - Found this article on using hooks and read through it, specifically the forward hooks section
- The pytorch docker image for pods has a different version of torch
    - Changed version for other libraries to become compatible
- The activation tensors have different sequence lengths, preventing them from being concatenated
    - Applied max pooling across the sequence dimension (dim=1) to get 1 activation vector per example
- Adapting Compexp’s implementation to Mistral Model
    - Used the same pipeline that Compexp used, but changed the way I extracted activations and features, pooled token-level activations, and constructing a new feature matrix
- OperatorPrecedence was not found in formula.py
    - Found out that it has been replaced by infixNotation and just changed OperatorPrecedence to infixNotation in formula.py

## What I Learned
- Learned how to hook activations in MLP layers of an LLM model
- Learned about exactly how the compositional explanations were implemented, including what each file and function did
- Learned how to adapt existing projects for my specific needs
- Learned how to implement beam search to discover compositional explanations
- Learned how to use multiprocessing to speed up compute time


## How to Run
### Setup
```bash
# Clone the repo
git clone https://github.com/codeCrafter10-alt/Explaining-LLM-Neurons
cd Explaining-LLM-Neurons

# Install dependencies
pip install -r requirements.txt
```

### Download Required Files
Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip, copy `snli_1.0/` and place it under `data/`

Finetune a Mistral-7B-v0.1 model and place the LoRA adapter files and Mistral checkpoint under a new folder `finetuned_model`. See https://github.com/codeCrafter10-alt/open_llms_nli/tree/main/finetuning_llm

### Change Settings for your needs
Change `settings.py` for the specific configuration you need. 

### Extract Activations
```bash
# Hook and save activations
python code/activations.py
```
This will create a new file `activations.pt`

### Build Analysis Data
```bash
# Extract features
python code/analyze.py
```
This will create a new file `results/analysis_data.pt`

### Generate Neuron Explanations
```bash
# Generate and save compositional explanations
python code/explain_neurons.py
```
The final explanations would be saved in `results/neuron_explanations.csv`