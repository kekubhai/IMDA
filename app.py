import gradio as gr
import numpy as np
import os
from models.fault_predictor import FaultPredictor
from models.shap_explainer import SHAPExplainer
from llm.repair_suggester import RepairSuggester
from memory.memory_manager import MemoryManager
from utils.data_utils import load_telemetry_data, validate_data

# Paths and configs (update as needed)
MODEL_PATH = "assets/fault_model.pt"  # Placeholder path
HUGGINGFACE_MODEL = "meta-llama/Llama-3-8b"  # Example model
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Initialize modules
fault_predictor = FaultPredictor(MODEL_PATH)
shap_explainer = SHAPExplainer(fault_predictor.model)
repair_suggester = RepairSuggester(HUGGINGFACE_MODEL, HUGGINGFACE_TOKEN)
memory_manager = MemoryManager()

def process_file(file):
    data = load_telemetry_data(file.name)
    if not validate_data(data):
        return "Invalid data!", None, None, None
    prob = fault_predictor.predict(data)
    shap_values = shap_explainer.explain(data)
    explanation = str(shap_values)
    context = f"Telemetry: {data.tolist()}, Fault probability: {prob:.2f}"
    similar_cases = memory_manager.retrieve_similar(context)
    suggestion = repair_suggester.suggest(context)
    memory_manager.add_case(context + "\nSuggestion: " + suggestion)
    return f"Fault probability: {prob:.2f}", explanation, suggestion, str(similar_cases)

def feedback_fn(feedback, context):
    # Store feedback for future improvement
    # (Extend as needed: e.g., log to file or DB)
    return "Thank you for your feedback!"

with gr.Blocks() as demo:
    gr.Markdown("# IMDA – Intelligent Maintenance Decision Agent")
    with gr.Row():
        file_input = gr.File(label="Upload Telemetry CSV")
        predict_btn = gr.Button("Predict & Suggest")
    output_pred = gr.Textbox(label="Fault Prediction")
    output_expl = gr.Textbox(label="SHAP Explanation")
    output_sugg = gr.Textbox(label="Repair Suggestion")
    output_cases = gr.Textbox(label="Similar Past Cases")
    with gr.Row():
        feedback = gr.Textbox(label="Feedback on Suggestion")
        feedback_btn = gr.Button("Submit Feedback")
    hidden_context = gr.State()

    predict_btn.click(
        process_file,
        inputs=[file_input],
        outputs=[output_pred, output_expl, output_sugg, output_cases]
    )
    feedback_btn.click(
        feedback_fn,
        inputs=[feedback, hidden_context],
        outputs=[]
    )

demo.launch() 