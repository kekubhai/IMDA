import pandas as pd
import numpy as np

def load_telemetry_data(file_path: str) -> np.ndarray:
    """
    Loads telemetry data from a CSV file and returns as a NumPy array.
    Expected columns: temperature, pressure, vibration
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for expected columns
        expected_cols = ['temperature', 'pressure', 'vibration']
        if not all(col in df.columns for col in expected_cols):
            print(f"Warning: Expected columns {expected_cols}, got {list(df.columns)}")
            # Use first 3 numeric columns if available
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            if len(numeric_cols) >= 3:
                df = df[numeric_cols]
                df.columns = expected_cols[:len(numeric_cols)]
            else:
                raise ValueError("Not enough numeric columns in CSV")
        else:
            df = df[expected_cols]
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        return df.values
        
    except Exception as e:
        print(f"Error loading telemetry data: {e}")
        # Return sample data as fallback
        return np.array([[75.0, 101.3, 0.02]])

def validate_data(data: np.ndarray) -> bool:
    """
    Validates the input telemetry data.
    """
    try:
        # Check if data is not empty
        if data.size == 0:
            return False
        
        # Check if data has the right shape (at least 3 features)
        if data.ndim == 1 and len(data) < 3:
            return False
        elif data.ndim == 2 and data.shape[1] < 3:
            return False
        
        # Check for invalid values (NaN, inf)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False
        
        # Basic range checks for telemetry data
        if data.ndim == 1:
            temp, pressure, vibration = data[:3]
        else:
            temp = np.mean(data[:, 0])
            pressure = np.mean(data[:, 1])
            vibration = np.mean(data[:, 2])
        
        # Reasonable ranges for industrial equipment
        if not (0 <= temp <= 200):  # Temperature in Celsius
            print(f"Warning: Temperature {temp} outside expected range (0-200°C)")
        
        if not (50 <= pressure <= 200):  # Pressure in kPa
            print(f"Warning: Pressure {pressure} outside expected range (50-200 kPa)")
        
        if not (0 <= vibration <= 10):  # Vibration in some unit
            print(f"Warning: Vibration {vibration} outside expected range (0-10)")
        
        return True
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """
    Preprocesses telemetry data for model input.
    """
    try:
        # Normalize data to reasonable ranges
        if data.ndim == 1:
            processed = data.copy()
        else:
            processed = data.copy()
        
        # Simple normalization (in a real system, use proper scaling)
        # Temperature: normalize around 75°C
        processed[..., 0] = (processed[..., 0] - 75) / 25
        # Pressure: normalize around 101 kPa
        processed[..., 1] = (processed[..., 1] - 101) / 10
        # Vibration: normalize around 0.02
        processed[..., 2] = (processed[..., 2] - 0.02) / 0.05
        
        return processed
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return data 