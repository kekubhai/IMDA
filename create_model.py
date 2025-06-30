#!/usr/bin/env python3
"""
Script to create and save a simple PyTorch model for fault prediction.
This creates a basic neural network that can predict fault probability from telemetry data.
"""
import torch
import torch.nn as nn
import numpy as np
import os

class FaultPredictionModel(nn.Module):
    """
    Simple neural network for fault prediction from telemetry data.
    Expects 3 input features: temperature, pressure, vibration
    """
    def __init__(self, input_size=3, hidden_size=64):
        super(FaultPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def create_sample_model():
    """
    Creates and saves a sample trained model for demonstration.
    In a real scenario, this would be replaced with proper training.
    """
    # Create model
    model = FaultPredictionModel()
    
    # Generate some sample training data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Sample telemetry data (temperature, pressure, vibration)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)
    # Simple rule: fault if temperature > 1 OR vibration > 1.5
    y = ((X[:, 0] > 1) | (X[:, 2] > 1.5)).astype(float)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Simple training loop
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
    
    # Set to evaluation mode
    model.eval()
    
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Save the model
    torch.save(model, 'assets/fault_model.pt')
    print("Model saved to assets/fault_model.pt")

if __name__ == "__main__":
    create_sample_model()
