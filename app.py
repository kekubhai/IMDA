import gradio as gr
import numpy as np
import os
import traceback
from dotenv import load_dotenv
from models.fault_predictor import FaultPredictor
from models.shap_explainer import SHAPExplainer
from llm.repair_suggester import RepairSuggester
from memory.memory_manager import MemoryManager
from utils.data_utils import load_telemetry_data, validate_data, preprocess_data

# Load environment variables
load_dotenv()

# Paths and configs (update as needed)
MODEL_PATH = "assets/fault_model.pt"
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium")  # More accessible model
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Initialize modules with error handling
def initialize_components():
    """Initialize all components with proper error handling"""
    components = {}
    
    # Initialize fault predictor
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}. Creating a sample model...")
            # Run the model creation script
            os.system("python create_model.py")
        
        components['fault_predictor'] = FaultPredictor(MODEL_PATH)
        print("✓ Fault predictor initialized")
    except Exception as e:
        print(f"✗ Error initializing fault predictor: {e}")
        components['fault_predictor'] = None
    
    # Initialize SHAP explainer
    try:
        if components['fault_predictor']:
            components['shap_explainer'] = SHAPExplainer(components['fault_predictor'].model)
            print("✓ SHAP explainer initialized")
        else:
            components['shap_explainer'] = None
    except Exception as e:
        print(f"✗ Error initializing SHAP explainer: {e}")
        components['shap_explainer'] = None
    
    # Initialize repair suggester
    try:
        components['repair_suggester'] = RepairSuggester(HUGGINGFACE_MODEL, HUGGINGFACE_TOKEN)
        print("✓ Repair suggester initialized")
    except Exception as e:
        print(f"✗ Error initializing repair suggester: {e}")
        components['repair_suggester'] = None
    
    # Initialize memory manager
    try:
        components['memory_manager'] = MemoryManager()
        print("✓ Memory manager initialized")
    except Exception as e:
        print(f"✗ Error initializing memory manager: {e}")
        components['memory_manager'] = None
    
    return components

# Initialize components
components = initialize_components()

def process_file(file):
    """Process uploaded telemetry file and return predictions and suggestions"""
    try:
        if file is None:
            return "No file uploaded!", "", "", ""
        
        # Load and validate data
        data = load_telemetry_data(file.name)
        if not validate_data(data):
            return "Invalid data format!", "", "", ""
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Get fault prediction
        if components['fault_predictor']:
            prob = components['fault_predictor'].predict(processed_data)
            prediction_text = f"Fault Probability: {prob:.3f} ({prob*100:.1f}%)"
            
            # Add risk level
            if prob > 0.7:
                risk_level = "🔴 HIGH RISK"
            elif prob > 0.4:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            prediction_text = f"{prediction_text}\nRisk Level: {risk_level}"
        else:
            prob = 0.5
            prediction_text = "Fault prediction unavailable (model not loaded)"
        
        # Get SHAP explanation
        if components['shap_explainer']:
            explanation = components['shap_explainer'].explain(processed_data)
        else:
            explanation = "SHAP explanation unavailable"
        
        # Create context for LLM
        # Format data nicely
        if data.ndim == 1:
            data_summary = f"Temperature: {data[0]:.1f}°C, Pressure: {data[1]:.1f} kPa, Vibration: {data[2]:.3f}"
        else:
            data_summary = f"Temperature: {np.mean(data[:, 0]):.1f}°C, Pressure: {np.mean(data[:, 1]):.1f} kPa, Vibration: {np.mean(data[:, 2]):.3f}"
        
        context = f"Industrial Equipment Telemetry - {data_summary}. Fault probability: {prob:.3f}"
        
        # Get similar cases
        similar_cases = []
        if components['memory_manager']:
            similar_cases = components['memory_manager'].retrieve_similar(context)
        
        similar_cases_text = "Previous Similar Cases:\n"
        if similar_cases:
            for i, case in enumerate(similar_cases[:3], 1):
                similar_cases_text += f"{i}. {case[:100]}...\n"
        else:
            similar_cases_text += "No similar cases found in memory."
        
        # Get repair suggestion
        if components['repair_suggester']:
            suggestion = components['repair_suggester'].suggest(context)
        else:
            suggestion = "Repair suggestions unavailable"
        
        # Store case in memory
        if components['memory_manager']:
            case_record = f"{context}\nSuggestion: {suggestion}"
            components['memory_manager'].add_case(case_record)
        
        return prediction_text, explanation, suggestion, similar_cases_text
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return error_msg, "", "", ""

def feedback_fn(feedback, prediction, suggestion):
    """Store user feedback for future improvement"""
    try:
        if feedback and feedback.strip():
            # In a real system, this would go to a database
            feedback_record = f"Feedback: {feedback}\nPrediction: {prediction}\nSuggestion: {suggestion}\n---"
            
            # Save to file for persistence
            os.makedirs("feedback", exist_ok=True)
            with open("feedback/user_feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback_record + "\n")
            
            return "Thank you for your feedback! It will help improve the system."
        else:
            return "Please enter some feedback before submitting."
    except Exception as e:
        return f"Error saving feedback: {e}"

# Create Gradio interface
with gr.Blocks(title="IMDA - Intelligent Maintenance Decision Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 IMDA – Intelligent Maintenance Decision Agent
    
    Upload telemetry data from industrial equipment to get AI-powered fault predictions, 
    explanations, and repair suggestions.
    
    **Expected CSV format:** `temperature,pressure,vibration`
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📁 Upload Telemetry CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            predict_btn = gr.Button("🔍 Analyze Equipment", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### Sample Data Format
            ```csv
            temperature,pressure,vibration
            75,101.3,0.02
            80,102.1,0.03
            90,99.8,0.05
            ```
            """)
    
    with gr.Row():
        with gr.Column():
            output_pred = gr.Textbox(
                label="🎯 Fault Prediction",
                lines=3,
                interactive=False
            )
            output_expl = gr.Textbox(
                label="📊 SHAP Explanation",
                lines=6,
                interactive=False
            )
        
        with gr.Column():
            output_sugg = gr.Textbox(
                label="🔧 Repair Suggestions",
                lines=8,
                interactive=False
            )
            output_cases = gr.Textbox(
                label="📋 Similar Past Cases",
                lines=5,
                interactive=False
            )
    
    gr.Markdown("---")
    gr.Markdown("### 💬 Feedback")
    
    with gr.Row():
        with gr.Column(scale=3):
            feedback = gr.Textbox(
                label="Your feedback on the suggestions",
                placeholder="Was this helpful? Any suggestions for improvement?",
                lines=2
            )
        with gr.Column(scale=1):
            feedback_btn = gr.Button("📝 Submit Feedback", variant="secondary")
    
    feedback_output = gr.Textbox(label="Feedback Status", visible=False)
    
    # Store outputs for feedback
    prediction_state = gr.State()
    suggestion_state = gr.State()
    
    # Event handlers
    predict_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[output_pred, output_expl, output_sugg, output_cases]
    ).then(
        fn=lambda pred, sugg: (pred, sugg),
        inputs=[output_pred, output_sugg],
        outputs=[prediction_state, suggestion_state]
    )
    
    feedback_btn.click(
        fn=feedback_fn,
        inputs=[feedback, prediction_state, suggestion_state],
        outputs=[feedback_output]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[feedback_output]
    )

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting IMDA - Intelligent Maintenance Decision Agent")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",  # For Hugging Face Spaces
        server_port=7860,
        share=False,
        show_error=True
    ) 