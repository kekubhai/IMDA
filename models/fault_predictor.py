import torch
import numpy as np
import os

class FaultPredictor:
    """
    Loads a PyTorch model and predicts fault probability from telemetry data.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load with weights_only=False for now, but in production should use safe loading
            self.model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def predict(self, data: np.ndarray) -> float:
        """
        Predicts the probability of a fault given telemetry data.
        Args:
            data (np.ndarray): Input features. Expected shape: (n_samples, n_features) or (n_features,)
        Returns:
            float: Probability of fault.
        """
        try:
            # Handle different input shapes
            if data.ndim == 1:
                # Single sample
                input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            elif data.ndim == 2:
                # Multiple samples - take the mean for now
                mean_data = np.mean(data, axis=0)
                input_tensor = torch.tensor(mean_data, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.sigmoid(output).item()
            
            return prob
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0.5  # Return neutral probability on error 