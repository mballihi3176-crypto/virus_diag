
from sklearn.ensemble import RandomForestClassifier
from core.model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, patient_data):
        return self.model.predict(patient_data)
