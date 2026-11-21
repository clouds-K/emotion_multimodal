# src/text/preprocessing.py

import re
import string

def clean_text(text):
    """
    Nettoie le texte
    """
    # Minuscules
    text = text.lower()
    
    # Supprimer URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les hashtags #
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Test
    test_text = "Je suis TELLEMENT heureux aujourd'hui !!! 😊 #happy"
    cleaned = clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Nettoyé: {cleaned}")
    print("\n✅ Module de preprocessing texte prêt")