"""
Loading and preprocessing of the SNLI dataset for the sequence classification.
"""

import pandas as pd
from datasets import Dataset

LABEL_MAP = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}

def load_dataset(file_path):
    """
    Load the SNLI test dataset.

    Args:
        file_path (str): The path to the SNLI dataset file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the loaded dataset.
    """

    df = pd.read_csv(file_path, sep='\t', header=0)

    return df

def prepare_data(df, limit=None):
    """
    Prepare the dataset for NLI evaluation by selecting relevant columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the loaded dataset.
    
    Returns:
        Dataset: Hugging Face Dataset object containing the prepared dataset.
    """

    # Remove examples where annotators disagreed on the label
    df = df[df['gold_label'] != '-'].copy()

    df = df.dropna(subset=['sentence1', 'sentence2', 'gold_label'])

    if limit is not None:
        df = df.head(limit)

    df['gold_label'] = df['gold_label'].map(LABEL_MAP)
    df = df[['sentence1', 'sentence2', 'gold_label']].reset_index(drop=True)

    df['gold_label'] = df['gold_label'].astype(int)

    df = df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})

    return Dataset.from_pandas(df, preserve_index=False)