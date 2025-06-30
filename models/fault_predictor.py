import torch
import numpy as np

class FaultPredictor:
    """
    Loads a PyTorch model and predicts fault probability from telemetry data.
    """
    def __init__(self, model_path: str):
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()

    def predict(self, data: np.ndarray) -> float:
        """
        Predicts the probability of a fault given telemetry data.
        Args:
            data (np.ndarray): Input features.
        Returns:
            float: Probability of fault.
        """
        with torch.no_grad():
            input_tensor = torch.tensor(data, dtype=torch.float32)
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
        return prob 