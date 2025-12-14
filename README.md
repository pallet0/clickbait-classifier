# Clickbait Classifier

A simple machine learning project that classifies whether a headline is clickbait or not. 

## Project Overview

This project implements two approaches to classify clickbait headlines: 
1. **Naive Baseline**:  A simple rule-based classifier using keyword matching
2. **Transformer-based Model**: Uses a pre-trained DistilBERT model fine-tuned for text classification

## Dataset

This project uses the [Clickbait Dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) from Kaggle by Aman Anand.

- **Size**: 32,000 headlines (balanced:  16,000 clickbait, 16,000 non-clickbait)
- **Format**:  CSV with columns `headline` and `clickbait` (1 = clickbait, 0 = non-clickbait)
- **Sources**: 
  - Clickbait:  BuzzFeed, Upworthy, ViralNova, etc.
  - Non-clickbait:  WikiNews, New York Times, The Guardian, etc.

## Project Structure

```
clickbait-classifier/
├── README.md
├── requirements.txt
├── clickbait_classifier. ipynb    # Main notebook with full pipeline
├── data/
│   └── clickbait_data. csv        # Dataset (download from Kaggle)
└── models/
    └── (saved model files)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- transformers
- torch
- matplotlib
- seaborn

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)
2. Place `clickbait_data.csv` in the `data/` folder
3. Open and run `clickbait_classifier.ipynb` in Jupyter Notebook

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Baseline | ~65% | ~0.62 | ~0.70 | ~0.66 |
| DistilBERT | ~95% | ~0.94 | ~0.96 | ~0.95 |

## Usage Example

```python
from transformers import pipeline

# Load the classifier
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict
headline = "You Won't Believe What Happened Next!"
result = classifier(headline)
print(result)
```

## Author

Student Project for CAS2105

## License

MIT License