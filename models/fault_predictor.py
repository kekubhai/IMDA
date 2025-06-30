import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings

class AdvancedFaultPredictionModel(nn.Module):
    """
    Enhanced neural network for fault prediction with multiple architectures and features.
    Supports different model types: simple, attention-based, and ensemble.
    """
    def __init__(self, input_size=3, hidden_size=64, num_classes=1, model_type='simple', dropout_rate=0.2):
        super(AdvancedFaultPredictionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        
        if model_type == 'simple':
            self._build_simple_network()
        elif model_type == 'deep':
            self._build_deep_network()
        elif model_type == 'attention':
            self._build_attention_network()
        elif model_type == 'ensemble':
            self._build_ensemble_network()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_simple_network(self):
        """Original simple network"""
        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    def _build_deep_network(self):
        """Deeper network with residual connections"""
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    def _build_attention_network(self):
        """Attention-based network"""
        self.feature_embedding = nn.Linear(self.input_size, self.hidden_size)
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    def _build_ensemble_network(self):
        """Ensemble of multiple smaller networks"""
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1)
            ) for _ in range(3)
        ])
        
        self.combiner = nn.Linear(3, self.num_classes)
    
    def forward(self, x):
        if self.model_type == 'simple':
            return self.network(x)
        
        elif self.model_type == 'deep':
            x = F.relu(self.input_layer(x))
            
            # Residual connection 1
            residual = x
            x = self.res_block1(x)
            x = F.relu(x + residual)
            
            # Residual connection 2
            residual = x
            x = self.res_block2(x)
            x = F.relu(x + residual)
            
            return self.output_layer(x)
        
        elif self.model_type == 'attention':
            x = self.feature_embedding(x)  # [batch, features] -> [batch, hidden]
            x = x.unsqueeze(1)  # [batch, 1, hidden] for attention
            
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # Feed forward
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
            
            x = x.squeeze(1)  # [batch, hidden]
            return self.classifier(x)
        
        elif self.model_type == 'ensemble':
            outputs = [network(x) for network in self.networks]
            combined = torch.cat(outputs, dim=1)
            return self.combiner(combined)
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate
        }

class EnhancedFaultPredictor:
    """
    Enhanced fault predictor with advanced features:
    - Multiple model architectures
    - Confidence estimation
    - Feature importance
    - Model interpretation
    - Performance monitoring
    - Data preprocessing
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model = None
        self.scaler = None
        self.config = None
        self.feature_names = ['temperature', 'pressure', 'vibration']
        self.prediction_history = []
        self.performance_metrics = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def create_model(self, input_size=3, hidden_size=64, model_type='simple', **kwargs):
        """Create a new model with specified architecture"""
        self.model = AdvancedFaultPredictionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            model_type=model_type,
            **kwargs
        )
        return self.model
    
    def load_model(self, model_path: str):
        """Load model with proper error handling"""
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Checkpoint format with metadata
                self.model = checkpoint['model']
                self.scaler = checkpoint.get('scaler', None)
                self.config = checkpoint.get('config', {})
                self.feature_names = checkpoint.get('feature_names', self.feature_names)
                self.performance_metrics = checkpoint.get('performance_metrics', {})
            else:
                # Legacy format - just the model
                self.model = checkpoint
            
            self.model.eval()
            print(f"✓ Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def save_model(self, model_path: str, include_metadata=True):
        """Save model with metadata"""
        try:
            if include_metadata:
                checkpoint = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'config': self.config,
                    'feature_names': self.feature_names,
                    'performance_metrics': self.performance_metrics,
                    'save_time': datetime.now().isoformat(),
                    'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
                }
            else:
                checkpoint = self.model
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)
            print(f"✓ Model saved successfully to {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving model: {e}")
    
    def preprocess_data(self, data: np.ndarray, fit_scaler=False) -> np.ndarray:
        """Advanced data preprocessing with scaling and validation"""
        try:
            # Convert to numpy if needed
            if isinstance(data, (list, tuple)):
                data = np.array(data)
            
            # Handle different input shapes
            original_shape = data.shape
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Validate data
            if data.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {data.shape[1]}")
            
            # Check for missing values
            if np.isnan(data).any():
                warnings.warn("Data contains NaN values. Imputing with median values.")
                data = self._impute_missing_values(data)
            
            # Scale data
            if fit_scaler or self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)
            
            # Return original shape if input was 1D
            if len(original_shape) == 1:
                scaled_data = scaled_data.flatten()
            
            return scaled_data
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return data
    
    def _impute_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Simple median imputation for missing values"""
        for i in range(data.shape[1]):
            col_data = data[:, i]
            if np.isnan(col_data).any():
                median_val = np.nanmedian(col_data)
                data[:, i] = np.where(np.isnan(col_data), median_val, col_data)
        return data
    
    def predict(self, data: np.ndarray, return_confidence=False) -> Union[float, Tuple[float, float]]:
        """Enhanced prediction with confidence estimation"""
        try:
            if self.model is None:
                raise RuntimeError("No model loaded. Please load or create a model first.")
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Handle different input shapes
            if processed_data.ndim == 1:
                input_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0)
            elif processed_data.ndim == 2:
                if processed_data.shape[0] > 1:
                    # Multiple samples - take the mean for prediction
                    mean_data = np.mean(processed_data, axis=0)
                    input_tensor = torch.tensor(mean_data, dtype=torch.float32).unsqueeze(0)
                else:
                    input_tensor = torch.tensor(processed_data, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected data shape: {processed_data.shape}")
            
            with torch.no_grad():
                self.model.eval()
                output = self.model(input_tensor)
                prob = torch.sigmoid(output).item()
                
                # Calculate confidence using entropy
                confidence = self._calculate_confidence(prob)
            
            # Store prediction for monitoring
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'input': processed_data.tolist() if processed_data.ndim > 1 else processed_data.tolist(),
                'probability': prob,
                'confidence': confidence
            })
            
            if return_confidence:
                return prob, confidence
            return prob
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            if return_confidence:
                return 0.5, 0.0
            return 0.5
    
    def _calculate_confidence(self, prob: float) -> float:
        """Calculate prediction confidence using entropy"""
        # Binary entropy - higher entropy means lower confidence
        entropy = -prob * np.log(prob + 1e-8) - (1 - prob) * np.log(1 - prob + 1e-8)
        max_entropy = np.log(2)  # Maximum entropy for binary classification
        confidence = 1 - (entropy / max_entropy)
        return confidence
    
    def predict_batch(self, data_batch: np.ndarray) -> Dict:
        """Batch prediction with statistics"""
        try:
            if self.model is None:
                raise RuntimeError("No model loaded.")
            
            processed_data = self.preprocess_data(data_batch)
            input_tensor = torch.tensor(processed_data, dtype=torch.float32)
            
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                probs = torch.sigmoid(outputs).squeeze().numpy()
                
                if probs.ndim == 0:
                    probs = np.array([probs])
            
            # Calculate statistics
            results = {
                'predictions': probs.tolist(),
                'mean_probability': float(np.mean(probs)),
                'std_probability': float(np.std(probs)),
                'max_probability': float(np.max(probs)),
                'min_probability': float(np.min(probs)),
                'high_risk_count': int(np.sum(probs > 0.7)),
                'medium_risk_count': int(np.sum((probs > 0.4) & (probs <= 0.7))),
                'low_risk_count': int(np.sum(probs <= 0.4))
            }
            
            return results
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, data: np.ndarray, method='gradient') -> Dict:
        """Calculate feature importance using gradients or permutation"""
        try:
            if self.model is None:
                raise RuntimeError("No model loaded.")
            
            processed_data = self.preprocess_data(data)
            if processed_data.ndim == 1:
                processed_data = processed_data.reshape(1, -1)
            
            if method == 'gradient':
                return self._gradient_feature_importance(processed_data)
            elif method == 'permutation':
                return self._permutation_feature_importance(processed_data)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return {}
    
    def _gradient_feature_importance(self, data: np.ndarray) -> Dict:
        """Calculate feature importance using gradients"""
        input_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        
        self.model.eval()
        output = self.model(input_tensor)
        prob = torch.sigmoid(output)
        
        # Calculate gradients
        prob.backward(torch.ones_like(prob))
        gradients = input_tensor.grad.abs().mean(dim=0).detach().numpy()
        
        # Normalize
        gradients = gradients / (gradients.sum() + 1e-8)
        
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, gradients)
        }
        
        return importance_dict
    
    def _permutation_feature_importance(self, data: np.ndarray) -> Dict:
        """Calculate feature importance using permutation"""
        baseline_pred = self.predict(data)
        importance_scores = []
        
        for i in range(data.shape[1]):
            # Permute feature i
            data_permuted = data.copy()
            np.random.shuffle(data_permuted[:, i])
            
            # Calculate prediction change
            permuted_pred = self.predict(data_permuted)
            importance = abs(baseline_pred - permuted_pred)
            importance_scores.append(importance)
        
        # Normalize
        total_importance = sum(importance_scores)
        if total_importance > 0:
            importance_scores = [score / total_importance for score in importance_scores]
        
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, importance_scores)
        }
        
        return importance_dict
    
    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        try:
            predictions = self.predict_batch(test_data)['predictions']
            binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': float(accuracy_score(test_labels, binary_predictions)),
                'precision': float(precision_score(test_labels, binary_predictions, zero_division=0)),
                'recall': float(recall_score(test_labels, binary_predictions, zero_division=0)),
                'f1_score': float(f1_score(test_labels, binary_predictions, zero_division=0)),
                'auc_score': float(roc_auc_score(test_labels, predictions)),
                'num_samples': len(test_labels),
                'evaluation_time': datetime.now().isoformat()
            }
            
            self.performance_metrics.update(metrics)
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'error': str(e)}
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics from prediction history"""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        probs = [p['probability'] for p in self.prediction_history]
        confidences = [p['confidence'] for p in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'mean_probability': float(np.mean(probs)),
            'std_probability': float(np.std(probs)),
            'mean_confidence': float(np.mean(confidences)),
            'high_risk_predictions': len([p for p in probs if p > 0.7]),
            'medium_risk_predictions': len([p for p in probs if 0.4 < p <= 0.7]),
            'low_risk_predictions': len([p for p in probs if p <= 0.4]),
            'last_prediction_time': self.prediction_history[-1]['timestamp']
        }
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history = []
        print("Prediction history cleared")
    
    def export_history(self, filepath: str):
        """Export prediction history to JSON"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'prediction_history': self.prediction_history,
                    'performance_metrics': self.performance_metrics,
                    'feature_names': self.feature_names,
                    'export_time': datetime.now().isoformat()
                }, f, indent=2)
            print(f"History exported to {filepath}")
        except Exception as e:
            print(f"Error exporting history: {e}")

# Legacy class for backward compatibility
class FaultPredictionModel(AdvancedFaultPredictionModel):
    """Backward compatibility wrapper"""
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__(input_size, hidden_size, model_type='simple')

class FaultPredictor(EnhancedFaultPredictor):
    """Backward compatibility wrapper"""
    def __init__(self, model_path: str):
        super().__init__(model_path)