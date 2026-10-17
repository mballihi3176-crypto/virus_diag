import pandas as pd
from sklearn.model_selection import train_test_split
import os


def charger_et_preparer_dataset():
    """
    Charge le fichier CSV, sépare les features (X) de la cible (y),
    et divise les données en ensemble d'entraînement et de test.
    """
    # 1. Définir le chemin vers le fichier CSV
    # On remonte d'un dossier (..) pour aller dans 'data'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, 'data', 'patient_data.csv')

    # 2. Lire les données avec pandas
    df = pd.read_csv(filepath)

    # 3. Séparer les caractéristiques (X) et la cible (y)
    # X : on garde l'âge, la température et les symptômes. On enlève patient_id et infected.
    X = df.drop(columns=['patient_id', 'infected'])

    # y : c'est ce qu'on veut prédire (1 = infecté, 0 = sain)
    y = df['infected']

    # 4. Diviser les données : 80% pour l'entraînement, 20% pour le test
    # random_state=42 permet d'avoir toujours la même division si on relance le code
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Retourner les données prêtes à l'emploi
    return X_train, X_test, y_train, y_test


# --- Bloc de test (pour vérifier que ça marche) ---
if __name__ == "__main__":
    print("Chargement des données...")
    X_train, X_test, y_train, y_test = charger_et_preparer_dataset()

    print(f"Taille de l'ensemble d'entraînement : {len(X_train)} patients")
    print(f"Taille de l'ensemble de test : {len(X_test)} patients")
    print("\nAperçu des données d'entraînement (X_train) :")
    print(X_train.head())