import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.audio.model import create_audio_model_mlp
from src.audio.feature_extraction import extract_all_features, extract_emotion_from_filename
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import glob
from tqdm import tqdm

def load_ravdess_data(data_dir):
    """
    Charge et prétraite toutes les données RAVDESS
    """
    print("📂 Chargement des données RAVDESS...")
    
    features = []
    labels = []
    
    # Trouver tous les fichiers .wav
    audio_files = glob.glob(os.path.join(data_dir, '*', '*.wav'))
    
    print(f"   Trouvé {len(audio_files)} fichiers audio")
    
    for audio_file in tqdm(audio_files, desc="Extraction des features"):
        # Extraire les features
        feat = extract_all_features(audio_file)
        
        if feat is not None:
            # Extraire l'émotion
            emotion = extract_emotion_from_filename(audio_file)
            
            if emotion:
                features.append(feat)
                labels.append(emotion)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\n✅ Chargé {len(features)} échantillons")
    print(f"   Forme des features: {features.shape}")
    
    return features, labels

def train_audio_model():
    print("🚀 ENTRAÎNEMENT DU MODÈLE AUDIO")
    print("="*60)
    
    # 1. Vérifier que les données existent
    ravdess_path = os.path.join(config.RAW_DATA_DIR, 'ravdess')
    
    if not os.path.exists(ravdess_path):
        print(f"❌ Dossier RAVDESS non trouvé!")
        print(f"   Chemin attendu: {ravdess_path}")
        print(f"\n📥 Téléchargez RAVDESS depuis:")
        print(f"   https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio")
        return
    
    # 2. Charger les données
    X, y = load_ravdess_data(ravdess_path)
    
    # 3. Encoder les labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    print(f"\n📊 Classes:")
    for i, emotion in enumerate(label_encoder.classes_):
        count = np.sum(y == emotion)
        print(f"   {emotion:10s}: {count:4d} échantillons")
    
    # 4. Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\n📂 Split:")
    print(f"   Train: {X_train.shape[0]}")
    print(f"   Validation: {X_val.shape[0]}")
    
    # 5. Créer le modèle
    print(f"\n🏗️  Création du modèle...")
    model = create_audio_model_mlp(
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_)
    )
    
    # 6. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'audio_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 7. Entraînement
    print(f"\n🎯 Début de l'entraînement...")
    print(f"   Epochs: 50")
    print(f"   Batch size: 32")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Évaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n📈 Résultats finaux:")
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")
    
    # 9. Sauvegarder
    model_path = os.path.join(config.MODEL_DIR, 'audio_model.h5')
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé: {model_path}")
    
    # 10. Sauvegarder le label encoder
    import joblib
    encoder_path = os.path.join(config.MODEL_DIR, 'audio_label_encoder.pkl')
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder sauvegardé: {encoder_path}")
    
    return model, history

if __name__ == "__main__":
    train_audio_model()