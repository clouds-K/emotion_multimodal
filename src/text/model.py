# src/text/model.py

from tensorflow import keras
from tensorflow.keras import layers

def create_text_model_lstm(vocab_size=5000, max_length=50, embedding_dim=128, num_classes=7):
    """
    Modèle LSTM pour l'analyse de sentiment textuel
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.SpatialDropout1D(0.2),
        
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = create_text_model_lstm()
    model.summary()
    print("\n✅ Modèle texte créé avec succès!")