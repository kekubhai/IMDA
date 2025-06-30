import shap
import numpy as np
import torch

class SHAPExplainer:
    """
    Uses SHAP to explain predictions from the fault predictor model.
    """
    def __init__(self, model):
        self.model = model
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(data):
                """Wrapper function to make the model compatible with SHAP"""
                if isinstance(data, np.ndarray):
                    data_tensor = torch.tensor(data, dtype=torch.float32)
                else:
                    data_tensor = data
                
                with torch.no_grad():
                    output = self.model(data_tensor)
                    prob = torch.sigmoid(output)
                    return prob.numpy()
            
            self.model_wrapper = model_wrapper
            
            # Generate background data for SHAP
            background_data = np.random.randn(100, 3)  # Assuming 3 features
            self.explainer = shap.Explainer(self.model_wrapper, background_data)
            
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None

    def explain(self, data: np.ndarray):
        """
        Returns SHAP values for the input data.
        Args:
            data (np.ndarray): Input features.
        Returns:
            str: SHAP explanation as text or error message.
        """
        if self.explainer is None:
            return "SHAP explanation not available"
        
        try:
            # Handle different input shapes
            if data.ndim == 1:
                # Single sample
                shap_input = data.reshape(1, -1)
            elif data.ndim == 2:
                # Multiple samples - take the mean
                shap_input = np.mean(data, axis=0).reshape(1, -1)
            else:
                return f"Unexpected data shape: {data.shape}"
            
            shap_values = self.explainer(shap_input)
            
            # Format SHAP values as readable text
            feature_names = ["Temperature", "Pressure", "Vibration"]
            if len(shap_input[0]) != len(feature_names):
                feature_names = [f"Feature_{i}" for i in range(len(shap_input[0]))]
            
            explanation_text = "SHAP Feature Importance:\n"
            for i, (feature, value, shap_val) in enumerate(zip(feature_names, shap_input[0], shap_values.values[0])):
                explanation_text += f"• {feature}: {value:.3f} (SHAP: {shap_val:.4f})\n"
            
            return explanation_text
            
        except Exception as e:
            return f"Error generating SHAP explanation: {e}" 