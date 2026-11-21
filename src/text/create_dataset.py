# src/text/create_dataset.py

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

def create_simple_emotion_dataset():
    """
    Crée un dataset texte simple pour les émotions
    """
    
    data = {
        'text': [
            # HAPPY
            "Je suis tellement heureux aujourd'hui",
            "C'est une excellente nouvelle",
            "Je me sens joyeux et plein d'énergie",
            "Quelle merveilleuse journée",
            "Je suis ravi de cette opportunité",
            "C'est génial, je suis aux anges",
            "Je me sens incroyablement bien",
            "Tout va parfaitement bien",
            "Je suis content de voir mes amis",
            "Cette journée est fantastique",
            
            # SAD
            "Je me sens triste et seul",
            "C'est vraiment déprimant",
            "Je suis malheureux en ce moment",
            "Tout va mal dans ma vie",
            "Je me sens vide et sans énergie",
            "C'est difficile de continuer",
            "Je n'ai plus envie de rien",
            "Je me sens abandonné",
            "C'est trop dur pour moi",
            "Je pleure souvent ces derniers temps",
            
            # ANGRY
            "Je suis vraiment en colère",
            "C'est inacceptable et révoltant",
            "Je suis furieux de cette situation",
            "Ça me met hors de moi",
            "Je n'en peux plus de cette injustice",
            "C'est insupportable",
            "Je suis énervé par ce comportement",
            "Ça suffit maintenant",
            "Je suis outré par ce manque de respect",
            "Cette situation me rend fou",
            
            # FEAR
            "J'ai vraiment peur de ce qui va arriver",
            "Je suis terrifié par cette situation",
            "Ça m'angoisse énormément",
            "J'ai peur de l'avenir",
            "Cette situation m'effraie",
            "Je suis anxieux et stressé",
            "J'ai des frissons rien que d'y penser",
            "Ça me fait vraiment peur",
            "Je suis inquiet pour demain",
            "J'ai peur de ne pas y arriver",
            
            # SURPRISE
            "Je ne m'attendais vraiment pas à ça",
            "Quelle surprise incroyable",
            "Je suis complètement stupéfait",
            "C'est inattendu et étonnant",
            "Je n'arrive pas à y croire",
            "Quelle révélation surprenante",
            "Je suis bouche bée",
            "C'est totalement inattendu",
            "Quel choc",
            "Je ne l'avais pas vu venir",
            
            # DISGUST
            "C'est vraiment dégoûtant",
            "Ça me dégoûte profondément",
            "C'est répugnant",
            "Je trouve ça écœurant",
            "C'est vraiment nauséabond",
            "Ça me soulève le cœur",
            "C'est immonde",
            "Je trouve ça révoltant",
            "C'est vraiment sale",
            "Ça me répugne",
            
            # NEUTRAL
            "C'est une journée normale",
            "Je vais au travail comme d'habitude",
            "Il fait beau aujourd'hui",
            "J'ai mangé un sandwich",
            "La réunion est à 14h",
            "Je lis un livre",
            "C'est mercredi",
            "J'ai fait les courses",
            "Il est 15 heures",
            "Je prends le bus",
        ],
        'emotion': [
            'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
            'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
            'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
            'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
            'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
            'disgust', 'disgust', 'disgust', 'disgust', 'disgust', 'disgust', 'disgust', 'disgust', 'disgust', 'disgust',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Sauvegarder
    output_path = os.path.join(config.RAW_DATA_DIR, 'text', 'emotion_dataset.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Dataset créé : {output_path}")
    print(f"   Total : {len(df)} exemples")
    print(f"\n📊 Distribution:")
    print(df['emotion'].value_counts())
    
    return output_path

if __name__ == "__main__":
    create_simple_emotion_dataset()
