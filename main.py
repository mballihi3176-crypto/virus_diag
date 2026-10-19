import sys
import os
import pandas as pd

# Permet à Python de trouver les dossiers (app, pipeline, etc.) depuis la racine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator


# --- EXERCICE 1 : La classe est maintenant ici, dans le fichier principal ---
class ClinicalPredictor:
    def __init__(self, model):
        self.model = model

    def diagnose(self, patient_data):
        # On récupère la probabilité via le modèle
        if hasattr(self.model, "predict_proba"):
            prediction = self.model.predict_proba(patient_data)[:, 1][0]
        else:
            prediction = self.model.predict(patient_data)[0]

        # On applique la règle de l'exercice
        if prediction >= 0.5:
            return "Infecté"
        else:
            return "Sain"


# ---------------------------------------------------------------------------

def main():
    print("=== DEMARRAGE DU SYSTEME DE DIAGNOSTIC ===")

    # 1. PHASE D'ENTRAINEMENT
    trainer = Trainer()
    model, X_test, y_test = trainer.entrainer()

    # 2. PHASE D'EVALUATION
    evaluator = Evaluator(model)
    evaluator.evaluer(X_test, y_test)

    # 3. PHASE DE PREDICTION CLINIQUE
    print("\n=== TEST DE PREDICTION CLINIQUE ===")

    # On instancie notre classe avec le modèle entraîné
    predictor = ClinicalPredictor(model)

    # On simule un nouveau patient (avec les mêmes colonnes que le CSV)
    nouveau_patient = pd.DataFrame({
        'age': [55],
        'temperature': [39.0],
        'fievre': [1],
        'toux': [1],
        'fatigue': [1],
        'difficulte_respiratoire': [0]
    })

    # On lance le diagnostic
    resultat = predictor.diagnose(nouveau_patient)
    print(f"Le diagnostic pour ce nouveau patient est : {resultat}")


if __name__ == "__main__":
    main()