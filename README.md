#  Détecteur multimodal d'émotions

Application de détection d'émotions apparentes à partir de trois sources : **le visage**, **la voix** et **le texte**. Chaque modalité a son propre modèle de deep learning (TensorFlow / Keras) et l'interface Streamlit permet de tester chacune d'elles et de comparer les résultats.

>  Cette application détecte des émotions *apparentes*. Elle ne peut en aucun cas diagnostiquer un problème de santé mentale.

## Fonctionnalités

| Modalité | Entrée | Modèle | Dataset |
|----------|--------|--------|---------|
|  Vision | Photo (jpg, png) | CNN (images 48×48, niveaux de gris) | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) |
|  Audio | Fichier wav / mp3 | MLP sur features audio (MFCC, etc.) | [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio) |
|  Texte | Phrase en français | LSTM | Petit dataset généré par `src/text/create_dataset.py` |

Les 7 émotions : colère, dégoût, peur, joie, tristesse, surprise, neutre.

L'interface affiche des graphiques (radar, barres, camembert) pour comparer les prédictions par modalité.

## Structure du projet

```
emotion_multimodal/
├── app/
│   └── simple_app.py          # Application Streamlit
├── src/
│   ├── vision/                # model.py, train.py, predict.py, preprocessing.py
│   ├── audio/                 # model.py, train.py, predict.py, feature_extraction.py
│   ├── text/                  # model.py, train.py, predict.py, preprocessing.py, create_dataset.py
│   └── fusion/                # multimodal_fusion.py
├── scripts/
│   └── check_datasets.py      # Vérifie la structure du dataset FER2013
├── config.py                  # Chemins, hyperparamètres, liste des émotions
├── requirements.txt
└── data/                      # NON versionné (datasets + modèles entraînés)
```

## Installation

Prérequis : Python 3 (version à préciser, par exemple 3.11) et `git`.

```bash
git clone https://github.com/clouds-K/emotion_multimodal.git
cd emotion_multimodal

# Environnement virtuel
python -m venv venv

# Activation (Windows PowerShell)
venv\Scripts\Activate.ps1
# Activation (Linux / macOS)
# source venv/bin/activate

pip install -r requirements.txt
```

## Données

Le dossier `data/` n'est pas dans le dépôt (trop volumineux). Il est créé automatiquement par `config.py` ; il faut y placer les datasets :

```
data/
├── raw/
│   ├── fer2013/        # train/ et test/ (un sous-dossier par émotion)
│   ├── ravdess/        # fichiers audio .wav
│   └── text/           # emotion_dataset.csv (généré, voir ci-dessous)
├── processed/
└── models/             # modèles entraînés (.h5, .pkl)
```

1. **Vision** : télécharger [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) et le décompresser dans `data/raw/fer2013/`. Vérification : `python scripts/check_datasets.py`
2. **Audio** : télécharger [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio) et le décompresser dans `data/raw/ravdess/`
3. **Texte** : générer le dataset avec `python src/text/create_dataset.py`

## Entraînement

Depuis la racine du projet :

```bash
python src/vision/train.py
python src/audio/train.py
python src/text/train.py
```

Les modèles sont enregistrés dans `data/models/` (`vision_model.h5`, `audio_model.h5`, `text_model.h5`, ainsi que les tokenizers et encodeurs `.pkl`).

## Lancer l'application

```bash
streamlit run app/simple_app.py
```

L'application charge les modèles depuis `data/models/`. Les modèles entraînés sont fournis dans le dépôt : l'application fonctionne donc directement après l'installation.


## Résultats

| Modèle | Précision (jeu de test) |
|--------|------------------------|
| Vision (CNN) | à compléter |
| Audio (MLP) | à compléter |
| Texte (LSTM) | à compléter |

## Limites

- Le dataset texte est petit et généré à la main : le modèle texte est une démonstration, pas un modèle robuste.
- FER2013 contient des images basse résolution (48×48) et des étiquettes parfois bruitées.
- RAVDESS est un corpus d'acteurs en anglais, en conditions de studio.
- Les émotions détectées sont apparentes et ne reflètent pas forcément l'état réel d'une personne.

## Auteur

Khouloud ([@clouds-K](https://github.com/clouds-K))
