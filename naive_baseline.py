"""
Naive rule-based baseline for clickbait detection. 
Uses simple heuristics like presence of clickbait keywords and punctuation patterns.
"""

def naive_clickbait_classifier(text):
    """
    Simple rule-based clickbait detector.
    Returns 1 for clickbait, 0 for non-clickbait. 
    """
    text_lower = text. lower()
    
    # Common clickbait phrases
    clickbait_keywords = [
        "you won't believe", "shocking", "amazing", "incredible",
        "this is why", "what happens next", "number", "secret",
        "they don't want you to know", "mind-blowing", "unbelievable",
        "will shock you", "can't believe", "jaw-dropping"
    ]
    
    score = 0
    
    # Check for clickbait keywords
    for keyword in clickbait_keywords: 
        if keyword in text_lower: 
            score += 1
    
    # Check for excessive punctuation (! or ?)
    if text. count('!') >= 1 or text.count('?') >= 2:
        score += 1
    
    # Check for ALL CAPS words
    words = text. split()
    caps_count = sum(1 for w in words if w. isupper() and len(w) > 2)
    if caps_count >= 1:
        score += 1
    
    # Check for numbers in title (listicles)
    if any(char.isdigit() for char in text):
        score += 0.5
    
    return 1 if score >= 1.5 else 0


# Test examples
if __name__ == "__main__":
    examples = [
        "10 Shocking Facts You Won't Believe!",
        "Scientists Discover New Species in Amazon",
        "You Won't Believe What Happened Next!!! ",
        "Economic Report Shows Steady Growth",
    ]
    
    for ex in examples:
        pred = naive_clickbait_classifier(ex)
        label = "Clickbait" if pred == 1 else "Not Clickbait"
        print(f"{label}: {ex}")