

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.vision.model import create_simple_cnn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

def train_vision_model():
    print(" ENTRAÎNEMENT DU MODÈLE VISION")
    print("="*60)
    
    # 1. Vérifier que les données existent
    train_dir = os.path.join(config.RAW_DATA_DIR, 'fer2013', 'train')
    
    if not os.path.exists(train_dir):
        print(f" ERREUR : Dossier train non trouvé!")
        print(f"   Chemin attendu: {train_dir}")
        print(f"\n📥 Téléchargez FER2013 depuis:")
        print(f"   https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    # 2. Créer des générateurs de données
    print("\n Chargement des données...")
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Générateur d'entraînement
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode='grayscale',
        class_mode='sparse',
        subset='training'
    )
    
    # Générateur de validation
    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode='grayscale',
        class_mode='sparse',
        subset='validation'
    )
    
    print(f"\n Données chargées:")
    print(f"   Train: {train_generator.samples} images")
    print(f"   Validation: {val_generator.samples} images")
    print(f"   Classes: {list(train_generator.class_indices.keys())}")
    
    # 3. Créer le modèle
    print(f"\n Création du modèle...")
    num_classes = len(train_generator.class_indices)
    model = create_simple_cnn(num_classes=num_classes)
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'vision_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. Entraînement
    print(f"\n Début de l'entraînement...")
    print(f"   Epochs: 10 (test rapide)")
    print(f"   Batch size: {config.BATCH_SIZE}")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,  
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Évaluation finale
    print(f"\n Évaluation finale:")
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")
    
    # 7. Sauvegarder le modèle final
    model_path = os.path.join(config.MODEL_DIR, 'vision_model.h5')
    model.save(model_path)
    print(f"\n Modèle sauvegardé: {model_path}")
    
    return model, history

if __name__ == "__main__":
    train_vision_model()