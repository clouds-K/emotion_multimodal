# config.py

import os

# ==================== CHEMINS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# Créer les dossiers
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== PARAMÈTRES ====================
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# ==================== ÉMOTIONS ====================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTIONS_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Tristesse', 'Surprise', 'Neutre']

# ==================== APPLICATION ====================
APP_TITLE = "🎭 Détecteur Multi-modal d'Émotions"

DISCLAIMER = """
⚠️ AVERTISSEMENT IMPORTANT
Cette application détecte les émotions apparentes.
Elle NE PEUT PAS diagnostiquer de problèmes de santé mentale.
"""

RANDOM_SEED = 42

print("✅ Configuration chargée")