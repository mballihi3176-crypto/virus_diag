from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Classe abstraite qui définit le 'contrat' que tous les modèles d'IA doivent respecter.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Méthode obligatoire pour entraîner le modèle.
        """
        pass

    @abstractmethod
    def predict(self, patient_data):
        """
        Méthode obligatoire pour faire une prédiction.
        """
        pass