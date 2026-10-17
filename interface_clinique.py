class ClinicalPredictor:
    def __init__(self, model):
        self.model = model

    def diagnose(self, patient_data):
        # 1. On demande au modèle de faire une prédiction
        prediction = self.model.predict(patient_data)

        # 2. On extrait la valeur si nécessaire
        if isinstance(prediction, (list, tuple)) or hasattr(prediction, '__len__'):
            if len(prediction) > 0:
                prediction = prediction[0]

        # 3. On applique la règle
        if prediction >= 0.5:
            return "Infecté"
        else:
            return "Sain"


# --- BLOC DE TEST --- (Ce bloc doit être TOUT À GAUCHE)
if __name__ == "__main__":
    # On crée une classe qui simule un vrai modèle d'IA
    class FakeModel:
        def predict(self, data):
            return 0.75


    class FakeModelSain:
        def predict(self, data):
            return 0.2


    # --- Test 1 : Patient malade ---
    print("--- Test avec un patient à 0.75 ---")
    modele1 = FakeModel()
    predicteur1 = ClinicalPredictor(modele1)

    patient_test = {"age": 45, "temperature": 38.5}
    resultat1 = predicteur1.diagnose(patient_test)
    print("Résultat :", resultat1)  # Doit afficher "Infecté"

    # --- Test 2 : Patient sain ---
    print("\n--- Test avec un patient à 0.20 ---")
    modele2 = FakeModelSain()
    predicteur2 = ClinicalPredictor(modele2)

    resultat2 = predicteur2.diagnose(patient_test)
    print("Résultat :", resultat2)  # Doit afficher "Sain"