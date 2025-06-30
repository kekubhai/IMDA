# IMDA – Intelligent Maintenance Decision Agent

IMDA is an AI-powered tool for predictive maintenance in industrial settings. It takes telemetry data (temperature, pressure, vibration, etc.), predicts faults using a PyTorch model, explains predictions with SHAP, and suggests repair actions using a Hugging Face LLM. The system uses LangChain for memory/context and optionally stores past cases in a Chroma vector database. A Gradio UI allows users to upload data, view predictions and explanations, see suggestions, and provide feedback.

## Features
- Fault prediction from telemetry data (PyTorch)
- Explainable AI with SHAP
- LLM-powered repair suggestions (Hugging Face, LangChain)
- Memory/context retention (LangChain, Chroma)
- Gradio UI for interaction and feedback
- Modular, well-commented code
- Ready for Hugging Face Spaces deployment

## Setup
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```

## Usage
- Upload telemetry data (CSV)
- View fault prediction and SHAP explanation
- Get LLM-generated repair suggestions
- Provide feedback on suggestions

## Deployment
- Designed for easy deployment on [Hugging Face Spaces](https://huggingface.co/spaces)

---

For more details, see code comments and module docstrings. 