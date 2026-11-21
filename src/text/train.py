
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.text.model import create_text_model_lstm
from src.text.preprocessing import clean_text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def train_text_model():
    print("🚀 ENTRAÎNEMENT DU MODÈLE TEXTE")
    print("="*60)
    
    # 1. Charger le dataset
    dataset_path = os.path.join(config.RAW_DATA_DIR, 'text', 'emotion_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé!")
        print(f"   Créez d'abord le dataset avec:")
        print(f"   python src/text/create_dataset.py")
        return
    
    print(f"📂 Chargement du dataset...")
    df = pd.read_csv(dataset_path)
    
    print(f"   Total : {len(df)} exemples")
    print(f"\n📊 Distribution:")
    print(df['emotion'].value_counts())
    
    # 2. Nettoyer les textes
    print(f"\n🧹 Nettoyage des textes...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 3. Tokenization
    print(f"\n🔤 Tokenization...")
    max_words = 5000
    max_len = 50
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['clean_text'])
    
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=max_len)
    
    print(f"   Vocabulaire : {len(tokenizer.word_index)} mots")
    print(f"   Longueur max : {max_len}")
    
    # 4. Encoder les labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['emotion'])
    y = to_categorical(y_encoded)
    
    # 5. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📂 Split:")
    print(f"   Train: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    
    # 6. Créer le modèle
    print(f"\n🏗️  Création du modèle...")
    model = create_text_model_lstm(
        vocab_size=max_words,
        max_length=max_len,
        num_classes=len(label_encoder.classes_)
    )
    
    # 7. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'text_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 8. Entraînement
    print(f"\n🎯 Début de l'entraînement...")
    print(f"   Epochs: 30")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Évaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n📈 Résultats finaux:")
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")
    
    # 10. Sauvegarder
    model_path = os.path.join(config.MODEL_DIR, 'text_model.h5')
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé: {model_path}")
    
    # Sauvegarder tokenizer et encoder
    import joblib
    tokenizer_path = os.path.join(config.MODEL_DIR, 'text_tokenizer.pkl')
    encoder_path = os.path.join(config.MODEL_DIR, 'text_label_encoder.pkl')
    
    joblib.dump(tokenizer, tokenizer_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"✅ Tokenizer sauvegardé: {tokenizer_path}")
    print(f"✅ Label encoder sauvegardé: {encoder_path}")
    
    return model, history

if __name__ == "__main__":
    train_text_model()