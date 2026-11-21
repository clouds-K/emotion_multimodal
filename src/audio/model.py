from tensorflow import keras
from tensorflow.keras import layers

def create_audio_model_mlp(input_dim=376, num_classes=7):
    """
    Modèle MLP (Multi-Layer Perceptron) pour les features audio
    
    Args:
        input_dim: Nombre de features (80 MFCC + autres)
        num_classes: Nombre d'émotions (7)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Bloc 1
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Bloc 2
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Bloc 3
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Sortie
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test
    model = create_audio_model_mlp()
    model.summary()
    print("\n✅ Modèle audio créé avec succès!")