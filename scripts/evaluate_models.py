"""
Évalue les trois modèles déjà entraînés (sans les réentraîner) et écrit
le résultat dans docs/resultats.md.

Usage, depuis la racine du projet :
    python scripts/evaluate_models.py

- Vision : précision sur le jeu de test officiel FER2013 (data/raw/fer2013/test)
- Audio  : précision sur le jeu de validation de train.py (20 % de RAVDESS, random_state=42)
- Texte  : précision sur le jeu de validation de train.py (20 % du CSV, random_state=42)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

SEED = 42


def evaluate_vision():
    test_dir = os.path.join(config.RAW_DATA_DIR, "fer2013", "test")
    model_path = os.path.join(config.MODEL_DIR, "vision_model.h5")
    if not os.path.isdir(test_dir) or not os.path.exists(model_path):
        print("Vision : dossier de test FER2013 ou modèle introuvable, ignoré.")
        return None

    generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=False,
    )
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(generator, verbose=0).argmax(axis=1)
    y_true = generator.classes
    names = list(generator.class_indices.keys())

    print("\n=== VISION (FER2013, jeu de test) ===")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    return {
        "Modèle": "Vision (CNN)",
        "Jeu d'évaluation": "FER2013 test",
        "Exemples": len(y_true),
        "Précision": accuracy_score(y_true, y_pred),
    }


def evaluate_audio():
    from src.audio.train import load_ravdess_data

    ravdess_dir = os.path.join(config.RAW_DATA_DIR, "ravdess")
    model_path = os.path.join(config.MODEL_DIR, "audio_model.h5")
    encoder_path = os.path.join(config.MODEL_DIR, "audio_label_encoder.pkl")
    if not os.path.isdir(ravdess_dir) or not os.path.exists(model_path):
        print("Audio : dossier RAVDESS ou modèle introuvable, ignoré.")
        return None

    X, y = load_ravdess_data(ravdess_dir)
    encoder = joblib.load(encoder_path)
    y_enc = encoder.transform(y)
    _, X_val, _, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
    )

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

    print("\n=== AUDIO (RAVDESS, jeu de validation) ===")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, zero_division=0))
    return {
        "Modèle": "Audio (MLP)",
        "Jeu d'évaluation": "RAVDESS validation (20 %)",
        "Exemples": len(y_val),
        "Précision": accuracy_score(y_val, y_pred),
    }


def evaluate_text():
    from src.text.preprocessing import clean_text

    csv_path = os.path.join(config.RAW_DATA_DIR, "text", "emotion_dataset.csv")
    model_path = os.path.join(config.MODEL_DIR, "text_model.h5")
    tokenizer_path = os.path.join(config.MODEL_DIR, "text_tokenizer.pkl")
    encoder_path = os.path.join(config.MODEL_DIR, "text_label_encoder.pkl")
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print("Texte : dataset ou modèle introuvable, ignoré.")
        print("        (le dataset se génère avec : python src/text/create_dataset.py)")
        return None

    df = pd.read_csv(csv_path)
    df["clean_text"] = df["text"].apply(clean_text)
    tokenizer = joblib.load(tokenizer_path)
    encoder = joblib.load(encoder_path)

    X = pad_sequences(tokenizer.texts_to_sequences(df["clean_text"]), maxlen=50)
    y_enc = encoder.transform(df["emotion"])
    _, X_val, _, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
    )

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

    print("\n=== TEXTE (jeu de validation) ===")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, zero_division=0))
    return {
        "Modèle": "Texte (LSTM)",
        "Jeu d'évaluation": "CSV validation (20 %)",
        "Exemples": len(y_val),
        "Précision": accuracy_score(y_val, y_pred),
    }


def main():
    results = [r for r in (evaluate_vision(), evaluate_audio(), evaluate_text()) if r]
    if not results:
        print("Aucun modèle évalué.")
        return

    lines = [
        "# Résultats",
        "",
        "| Modèle | Jeu d'évaluation | Exemples | Précision |",
        "|--------|------------------|----------|-----------|",
    ]
    for r in results:
        dataset_name = r["Jeu d'évaluation"]
        accuracy_pct = r["Précision"] * 100
        lines.append(
            f"| {r['Modèle']} | {dataset_name} | {r['Exemples']} | {accuracy_pct:.1f} % |"
        )

    out_dir = os.path.join(config.BASE_DIR, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resultats.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nRésultats écrits dans {out_path}")


if __name__ == "__main__":
    main()
