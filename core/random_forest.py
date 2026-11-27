from typing import Any

from sklearn.ensemble import RandomForestClassifier
from core.model import BaseModel


class RandomForestModel(BaseModel):
    """
    Modèle de classification basé sur l'algorithme Random Forest.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> None:
        """
        Initialise le classificateur Random Forest.

        Args:
            n_estimators: Nombre d'arbres dans la forêt.
            random_state: Graine pour garantir la reproductibilité.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def train(self, X_train: Any, y_train: Any) -> RandomForestClassifier:
        """
        Entraîne le modèle sur les données d'apprentissage.

        Args:
            X_train: Variables explicatives.
            y_train: Étiquettes des classes.

        Returns:
            Le modèle entraîné.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, patient_data: Any) -> Any:
        """
        Prédit la classe des données d'un patient.

        Args:
            patient_data: Données du patient à classifier.

        Returns:
            Les classes prédites.
        """
        return self.model.predict(patient_data)