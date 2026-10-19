import sys
import os

# Permet d'importer les dossiers core et utils si on lance depuis pipeline/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import charger_et_preparer_dataset
from sklearn.linear_model import LogisticRegression  # On utilise scikit-learn directement pour tester


class Trainer:
    def __init__(self):
        # On initialise le modèle (ici la régression logistique)
        self.model = LogisticRegression()

    def entrainer(self):
        print("--- 1. Chargement des données ---")
        X_train, X_test, y_train, y_test = charger_et_preparer_dataset()

        print("\n--- 2. Entraînement du modèle ---")
        self.model.fit(X_train, y_train)
        print("Modèle entraîné avec succès !")

        # On retourne le modèle entraîné et les données de test pour l'évaluation
        return self.model, X_test, y_test