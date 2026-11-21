# src/audio/predict.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
import tensorflow as tf
from src.audio.feature_extraction import extract_all_features
import numpy as np
import joblib

class AudioPredictor:
    def __init__(self, model_path, encoder_path):
        print(f"📂 Chargement du modèle audio...")
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"✅ Modèle audio chargé avec succès")
    
    def predict(self, audio_path):
        """Prédit l'émotion d'un fichier audio"""
        print(f"\n🔍 Analyse de : {os.path.basename(audio_path)}")
        
        # Extraire les features
        features = extract_all_features(audio_path)
        
        if features is None:
            return None, 0.0, {}
        
        # Reshape pour le modèle
        features = np.expand_dims(features, axis=0)
        
        # Prédire
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Décoder l'émotion
        emotion_idx = np.argmax(predictions)
        emotion = self.label_encoder.classes_[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Afficher
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
    model_path = os.path.join(config.MODEL_DIR, 'audio_model.h5')
    encoder_path = os.path.join(config.MODEL_DIR, 'audio_label_encoder.pkl')
    
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé. Entraînez d'abord avec train.py")
        exit(1)
    
    predictor = AudioPredictor(model_path, encoder_path)
    
    # Test sur un fichier RAVDESS
    test_audio = os.path.join(
        config.RAW_DATA_DIR,
        'ravdess',
        'Actor_01',
        '03-01-03-01-01-01-01.wav'  # Happy
    )
    
    if os.path.exists(test_audio):
        emotion, conf, probs = predictor.predict(test_audio)
    else:
        print("❌ Fichier de test non trouvé")