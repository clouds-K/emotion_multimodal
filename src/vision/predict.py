# src/vision/predict.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

class VisionPredictor:
    def __init__(self, model_path):
        print(f"📂 Chargement du modèle vision : {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = config.EMOTIONS
        print(f"✅ Modèle vision chargé avec succès")
    
    def preprocess_image(self, image_path):
        """Prétraite une image pour le modèle"""
        # Charger l'image en niveaux de gris
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")
        
        # Détecter le visage
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        if len(faces) == 0:
            print("⚠️  Aucun visage détecté, utilisation de l'image entière")
            face = img
        else:
            # Prendre le plus grand visage
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face = img[y:y+h, x:x+w]
        
        # Redimensionner à 48x48
        face = cv2.resize(face, config.IMG_SIZE)
        
        # Normaliser [0, 1]
        face = face.astype('float32') / 255.0
        
        # Ajouter dimensions pour le modèle
        face = np.expand_dims(face, axis=-1)  # Ajouter canal
        face = np.expand_dims(face, axis=0)   # Ajouter batch
        
        return face
    
    def predict(self, image_path):
        """Prédit l'émotion d'une image"""
        print(f"\n🔍 Analyse de : {os.path.basename(image_path)}")
        
        # Prétraiter
        processed = self.preprocess_image(image_path)
        
        # Prédire
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Trouver l'émotion dominante
        emotion_idx = np.argmax(predictions)
        emotion = self.emotions[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Afficher résultat principal
        print(f"\n🎭 Résultat :")
        print(f"   Émotion détectée : {emotion}")
        print(f"   Confiance : {confidence*100:.2f}%")
        
        # Afficher toutes les probabilités
        print(f"\n📊 Toutes les probabilités :")
        for i, emo in enumerate(self.emotions):
            bar = "█" * int(predictions[i] * 30)
            print(f"   {emo:10s} : {predictions[i]*100:5.2f}% {bar}")
        
        # Retourner aussi toutes les probabilités pour les graphiques
        all_probs = {
            self.emotions[i]: float(predictions[i]) 
            for i in range(len(self.emotions))
        }
        
        return emotion, float(confidence), all_probs

if __name__ == "__main__":
    # Vérifier que le modèle existe
    model_path = os.path.join(config.MODEL_DIR, 'vision_model.h5')
    
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé!")
        print(f"   Entraînez d'abord le modèle avec : python src/vision/train.py")
        exit(1)
    
    # Charger le prédicteur
    predictor = VisionPredictor(model_path)
    
    # Tester sur une image du dataset test
    test_dir = os.path.join(config.RAW_DATA_DIR, 'fer2013', 'test')
    
    # Prendre une image happy
    happy_dir = os.path.join(test_dir, 'happy')
    test_images = os.listdir(happy_dir)
    
    if test_images:
        test_image = os.path.join(happy_dir, test_images[0])
        emotion, conf, probs = predictor.predict(test_image)
    else:
        print("❌ Aucune image de test trouvée")