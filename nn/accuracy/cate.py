# Accuracy calculation for classification model
from .parent import Accuracy
class Accuracy_Categorical(Accuracy):
    # No initialization is needed
    def init(self, y):
        pass
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
        # Accuracy calculation for regression model