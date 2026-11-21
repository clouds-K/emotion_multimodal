import librosa
import numpy as np
import os

def extract_mfcc(audio_path, duration=3, sr=22050, n_mfcc=40):
    """
    Extrait les MFCC (Mel-frequency cepstral coefficients) d'un fichier audio
    
    Les MFCC capturent les caractéristiques tonales de la voix
    """
    try:
        # Charger l'audio
        y, sr = librosa.load(audio_path, duration=duration, sr=sr)
        
        # Extraire MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculer moyenne et écart-type pour chaque coefficient
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Concaténer
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        return features
    
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
        return None

def extract_all_features(audio_path, duration=3, sr=22050):
    """
    Extrait toutes les features audio utiles
    """
    try:
        # Charger l'audio
        y, sr = librosa.load(audio_path, duration=duration, sr=sr)
        
        # 1. MFCC (40 coefficients × 2 stats = 80 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Chroma (12 features × 2 = 24)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Mel Spectrogram (128 features × 2 = 256)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        
        # 4. Spectral Contrast (7 features × 2 = 14)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # 5. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Concaténer toutes les features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std,
            [zcr_mean, zcr_std]
        ])
        
        return features
    
    except Exception as e:
        print(f"Erreur : {e}")
        return None

def extract_emotion_from_filename(filename):
    """
    Extrait l'émotion depuis le nom du fichier RAVDESS
    
    Format: 03-01-05-01-01-01-01.wav
    Position 3 = émotion
    """
    parts = os.path.basename(filename).split('-')
    if len(parts) >= 3:
        emotion_code = int(parts[2])
        
        # Mapping RAVDESS -> nos émotions
        emotion_map = {
            1: 'neutral',   # neutral
            2: 'neutral',   # calm -> neutral
            3: 'happy',     # happy
            4: 'sad',       # sad
            5: 'angry',     # angry
            6: 'fear',      # fearful
            7: 'disgust',   # disgust
            8: 'surprise'   # surprised
        }
        
        return emotion_map.get(emotion_code, 'neutral')
    
    return None

if __name__ == "__main__":
    print("✅ Module d'extraction de features audio prêt")
    
    # Test si librosa fonctionne
    print(f"Librosa version: {librosa.__version__}")