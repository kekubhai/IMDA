import shap
import numpy as np

class SHAPExplainer:
    """
    Uses SHAP to explain predictions from the fault predictor model.
    """
    def __init__(self, model):
        self.model = model
        self.explainer = shap.Explainer(self.model)

    def explain(self, data: np.ndarray):
        """
        Returns SHAP values for the input data.
        Args:
            data (np.ndarray): Input features.
        Returns:
            shap.Explanation: SHAP explanation object.
        """
        shap_values = self.explainer(data)
        return shap_values 