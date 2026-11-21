

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

def check_datasets():
    print("🔍 VÉRIFICATION COMPLÈTE DES DATASETS")
    print("="*60)
    
    fer2013_path = os.path.join(config.RAW_DATA_DIR, 'fer2013')
    
    # 1. Vérifier que fer2013 existe
    if not os.path.exists(fer2013_path):
        print(f" Dossier fer2013 non trouvé!")
        print(f"   Chemin attendu: {fer2013_path}")
        return
    
    print(f" Dossier fer2013 trouvé\n")
    
    # 2. Lister le contenu de fer2013
    fer2013_content = os.listdir(fer2013_path)
    print(f" Contenu de fer2013/:")
    for item in sorted(fer2013_content):
        item_path = os.path.join(fer2013_path, item)
        if os.path.isdir(item_path):
            print(f"    {item}/")
        else:
            print(f"    {item}")
    
    # 3. Vérifier TRAIN
    print(f"\n{'='*60}")
    train_path = os.path.join(fer2013_path, 'train')
    
    if os.path.exists(train_path):
        print(f" TRAIN trouvé")
        emotions = os.listdir(train_path)
        total_train = 0
        print(f"\n Contenu de train/:")
        for emotion in sorted(emotions):
            emotion_path = os.path.join(train_path, emotion)
            if os.path.isdir(emotion_path):
                count = len(os.listdir(emotion_path))
                total_train += count
                print(f"   {emotion:10s}: {count:5d} images")
        print(f"   {'─'*30}")
        print(f"   {'TOTAL':10s}: {total_train:5d} images")
    else:
        print(f" TRAIN non trouvé!")
    
    # 4. Vérifier TEST
    print(f"\n{'='*60}")
    test_path = os.path.join(fer2013_path, 'test')
    
    if os.path.exists(test_path):
        print(f"TEST trouvé")
        emotions = os.listdir(test_path)
        total_test = 0
        print(f"\n Contenu de test/:")
        for emotion in sorted(emotions):
            emotion_path = os.path.join(test_path, emotion)
            if os.path.isdir(emotion_path):
                count = len(os.listdir(emotion_path))
                total_test += count
                print(f"   {emotion:10s}: {count:5d} images")
        print(f"   {'─'*30}")
        print(f"   {'TOTAL':10s}: {total_test:5d} images")
    else:
        print(f"\nTEST non trouvé!")
        print(f"   Chemin attendu: {test_path}")
        print(f"     Le dossier TEST est manquant!")
    
    # 5. Résumé final
    print(f"\n{'='*60}")
    print(f" RÉSUMÉ:")
    
    has_train = os.path.exists(train_path)
    has_test = os.path.exists(test_path)
    
    if has_train and has_test:
        print(f"    Structure COMPLÈTE (train + test)")
        print(f"    Prêt pour l'entraînement!")
    elif has_train and not has_test:
        print(f"     Train OK, mais TEST MANQUANT")
        print(f"     Vous pouvez quand même entraîner")
        print(f"     La validation sera faite sur une partie de train")
    else:
        print(f"    Structure incomplète")

if __name__ == "__main__":
    check_datasets()