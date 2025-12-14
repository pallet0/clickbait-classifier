"""
Main pipeline for clickbait classification.
Combines data loading, preprocessing, and model inference.
"""

import pandas as pd
from naive_baseline import naive_clickbait_classifier

def load_data(filepath, sample_size=1000):
    """Load and sample the clickbait dataset."""
    df = pd.read_csv(filepath)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df


def preprocess(text):
    """Basic text preprocessing."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def evaluate(y_true, y_pred):
    """Calculate accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0


def run_pipeline(data_path):
    """Run the full classification pipeline."""
    # Step 1: Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Step 2: Preprocess
    print("Preprocessing...")
    df['clean_text'] = df['headline'].apply(preprocess)
    
    # Step 3: Run naive baseline
    print("Running naive baseline...")
    df['naive_pred'] = df['clean_text']. apply(naive_clickbait_classifier)
    
    # Step 4: Evaluate
    if 'clickbait' in df.columns:
        accuracy = evaluate(df['clickbait']. tolist(), df['naive_pred'].tolist())
        print(f"Naive Baseline Accuracy: {accuracy:.2%}")
    
    return df


if __name__ == "__main__":
    # Update path to your dataset location
    DATA_PATH = "./data/clickbait_data.csv"
    results = run_pipeline(DATA_PATH)
    print(results.head())