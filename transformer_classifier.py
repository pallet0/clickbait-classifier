"""
Transformer-based clickbait classifier using Hugging Face. 
Uses a pre-trained model for text classification.
"""

from transformers import pipeline

def load_classifier():
    """
    Load a pre-trained text classification model.
    Using a sentiment model as a simple proxy (or you can fine-tune your own).
    """
    # For simplicity, using a general text classifier
    # In practice, you'd fine-tune on clickbait data
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier


def transformer_clickbait_classifier(text, classifier):
    """
    Use transformer model for classification.
    Note: This is a placeholder - ideally fine-tuned on clickbait data. 
    """
    result = classifier(text)[0]
    # This is simplified - a real implementation would use a model
    # fine-tuned specifically on clickbait data
    return result


if __name__ == "__main__":
    classifier = load_classifier()
    
    examples = [
        "10 Shocking Facts You Won't Believe!",
        "Scientists Discover New Species in Amazon",
    ]
    
    for ex in examples: 
        result = transformer_clickbait_classifier(ex, classifier)
        print(f"{ex}\n  -> {result}\n")