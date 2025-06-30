---
title: IMDA - Intelligent Maintenance Decision Agent
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered predictive maintenance with fault prediction, SHAP explanations, and LLM repair suggestions
---

# IMDA – Intelligent Maintenance Decision Agent

An AI-powered tool for predictive maintenance in industrial settings. Upload telemetry data to get fault predictions, explanations, and repair suggestions.

## Features
- 🎯 **Fault Prediction**: PyTorch-based ML model predicts equipment failures
- 📊 **Explainable AI**: SHAP values explain why predictions were made  
- 🔧 **Smart Suggestions**: LLM generates contextual repair recommendations
- 💾 **Memory System**: Stores and retrieves similar past cases
- 📝 **User Feedback**: Continuous improvement through user input

## Usage
1. Upload a CSV file with columns: `temperature`, `pressure`, `vibration`
2. Click "Analyze Equipment" to get predictions and suggestions
3. Review the fault probability, SHAP explanation, and repair recommendations
4. Provide feedback to help improve the system

## Example Data
```csv
temperature,pressure,vibration
75,101.3,0.02
80,102.1,0.03
90,99.8,0.05
```
