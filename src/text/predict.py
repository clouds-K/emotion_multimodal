# src/text/predict.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
import tensorflow as tf
import numpy as np
import joblib
from src.text.preprocessing import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPredictor:
    def __init__(self, model_path, tokenizer_path, encoder_path):
        print(f"📂 Chargement du modèle texte...")
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"✅ Modèle texte chargé avec succès")
    
    def predict(self, text, max_length=50):
        """Prédit l'émotion d'un texte"""
        print(f"\n🔍 Analyse du texte : '{text[:50]}...'")
        
        # Nettoyer le texte
        cleaned_text = clean_text(text)
        print(f"   Texte nettoyé : '{cleaned_text[:50]}...'")
        
        # Convertir en séquence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        
        # Padding
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Prédire
        predictions = self.model.predict(padded, verbose=0)[0]
        
        # Décoder l'émotion
        emotion_idx = np.argmax(predictions)
        emotion = self.label_encoder.classes_[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Afficher résultat
        print(f"\n🎭 Résultat :")
        print(f"   Émotion détectée : {emotion}")
        print(f"   Confiance : {confidence*100:.2f}%")
        
        print(f"\n📊 Toutes les probabilités :")
        for i, emo in enumerate(self.label_encoder.classes_):
            bar = "█" * int(predictions[i] * 30)
            print(f"   {emo:10s} : {predictions[i]*100:5.2f}% {bar}")
        
        # Retourner aussi toutes les probabilités pour les graphiques
        all_probs = {
            emo: float(predictions[i]) 
            for i, emo in enumerate(self.label_encoder.classes_)
        }
        
        return emotion, float(confidence), all_probs

if __name__ == "__main__":
    model_path = os.path.join(config.MODEL_DIR, 'text_model.h5')
    tokenizer_path = os.path.join(config.MODEL_DIR, 'text_tokenizer.pkl')
    encoder_path = os.path.join(config.MODEL_DIR, 'text_label_encoder.pkl')
    
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé. Entraînez d'abord avec train.py")
        exit(1)
    
    predictor = TextPredictor(model_path, tokenizer_path, encoder_path)
    
    # Tests
    test_texts = [
        "I am so happy today! Everything is wonderful!",
        "I feel very sad and lonely right now.",
        "This makes me so angry! I can't believe it!",
        "I'm scared of what might happen next.",
    ]
    
    for text in test_texts:
        print("\n" + "="*60)
        emotion, confidence, probs = predictor.predict(text)
        print("="*60)