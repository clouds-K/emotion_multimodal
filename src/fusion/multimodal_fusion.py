import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class MultimodalFusion:
    """
    Classe pour fusionner les prédictions de plusieurs modalités
    """
    
    def __init__(self, vision_model_path, audio_model_path, text_model_path):
        """
        Charge les 3 modèles
        """
        print("📂 Chargement des modèles...")
        self.vision_model = keras.models.load_model(vision_model_path)
        self.audio_model = keras.models.load_model(audio_model_path)
        self.text_model = keras.models.load_model(text_model_path)
        print("✅ Modèles chargés")
    
    def predict_weighted_average(self, vision_input, audio_input, text_input,
                                 weights=[0.4, 0.3, 0.3]):
        """
        Fusion par moyenne pondérée
        
        Args:
            vision_input: Image prétraitée
            audio_input: Features audio
            text_input: Texte prétraité
            weights: Poids [vision, audio, text]
        
        Returns:
            (emotion, confidence, all_probs)
        """
        # Prédictions individuelles
        vision_pred = self.vision_model.predict(vision_input, verbose=0)[0]
        audio_pred = self.audio_model.predict(audio_input, verbose=0)[0]
        text_pred = self.text_model.predict(text_input, verbose=0)[0]
        
        # Fusion pondérée
        fused_pred = (
            weights[0] * vision_pred +
            weights[1] * audio_pred +
            weights[2] * text_pred
        )
        
        # Normaliser
        fused_pred = fused_pred / np.sum(fused_pred)
        
        # Trouver l'émotion dominante
        emotion_idx = np.argmax(fused_pred)
        confidence = fused_pred[emotion_idx]
        
        return emotion_idx, float(confidence), fused_pred
    
    def predict_voting(self, vision_input, audio_input, text_input):
        """
        Fusion par vote majoritaire
        """
        # Prédictions individuelles
        vision_pred = np.argmax(self.vision_model.predict(vision_input, verbose=0)[0])
        audio_pred = np.argmax(self.audio_model.predict(audio_input, verbose=0)[0])
        text_pred = np.argmax(self.text_model.predict(text_input, verbose=0)[0])
        
        # Vote
        votes = [vision_pred, audio_pred, text_pred]
        emotion_idx = max(set(votes), key=votes.count)
        
        return emotion_idx
    
    def predict_max_confidence(self, vision_input, audio_input, text_input):
        """
        Fusion en prenant la prédiction la plus confiante
        """
        # Prédictions individuelles
        vision_pred = self.vision_model.predict(vision_input, verbose=0)[0]
        audio_pred = self.audio_model.predict(audio_input, verbose=0)[0]
        text_pred = self.text_model.predict(text_input, verbose=0)[0]
        
        # Confidences
        vision_conf = np.max(vision_pred)
        audio_conf = np.max(audio_pred)
        text_conf = np.max(text_pred)
        
        # Choisir la plus confiante
        max_conf = max(vision_conf, audio_conf, text_conf)
        
        if max_conf == vision_conf:
            return np.argmax(vision_pred), vision_conf, "vision"
        elif max_conf == audio_conf:
            return np.argmax(audio_pred), audio_conf, "audio"
        else:
            return np.argmax(text_pred), text_conf, "text"

def create_late_fusion_model(vision_model, audio_model, text_model, num_classes=7):
    """
    Crée un modèle de fusion tardive (late fusion)
    Les 3 modèles sont combinés et fine-tunés ensemble
    """
    # Extraire les features de chaque modèle (avant la couche finale)
    vision_base = keras.Model(
        inputs=vision_model.input,
        outputs=vision_model.layers[-3].output  # Avant softmax
    )
    audio_base = keras.Model(
        inputs=audio_model.input,
        outputs=audio_model.layers[-3].output
    )
    text_base = keras.Model(
        inputs=text_model.input,
        outputs=text_model.layers[-3].output
    )
    
    # Geler les bases
    vision_base.trainable = False
    audio_base.trainable = False
    text_base.trainable = False
    
    # Inputs
    vision_input = layers.Input(shape=vision_model.input.shape[1:], name='vision_input')
    audio_input = layers.Input(shape=audio_model.input.shape[1:], name='audio_input')
    text_input = layers.Input(shape=text_model.input.shape[1:], name='text_input')
    
    # Extraire features
    vision_features = vision_base(vision_input)
    audio_features = audio_base(audio_input)
    text_features = text_base(text_input)
    
    # Concatener
    combined = layers.Concatenate()([vision_features, audio_features, text_features])
    
    # Couches de fusion
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Modèle final
    model = keras.Model(
        inputs=[vision_input, audio_input, text_input],
        outputs=outputs
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("✅ Module de fusion multimodale prêt")