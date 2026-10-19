import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import evaluate_metrics


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluer(self, X_test, y_test):
        print("\n--- 3. Évaluation du modèle ---")

        # Le modèle prédit les classes (0 pour sain, 1 pour infecté)
        predictions = self.model.predict(X_test)

        # On calcule les métriques via utils/metrics.py
        resultats = evaluate_metrics(y_test, predictions)

        print("Résultats de l'évaluation :")
        for nom_metrique, valeur in resultats.items():
            print(f"- {nom_metrique} : {valeur:.2f}")

        return resultats