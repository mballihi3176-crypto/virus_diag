import pandas as pd


def load_patient_data(filepath="../data/patient_data.csv"):
    """
    Charge le fichier CSV et sépare les caractéristiques (X) de la cible (y).
    """
    # Lecture du fichier CSV
    df = pd.read_csv(filepath)

    # X contient les features (on enlève l'ID et la cible 'infected')
    X = df.drop(columns=['patient_id', 'infected'])

    # y contient la cible (1 ou 0)
    y = df['infected']

    return X, y


# Test rapide si on exécute ce fichier directement
if __name__ == "__main__":
    X, y = load_patient_data()
    print("Aperçu des données (X) :")
    print(X.head())
    print("\nAperçu des cibles (y) :")
    print(y.head())