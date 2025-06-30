from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os

class RepairSuggester:
    """
    Uses a Hugging Face LLM (via LangChain) to suggest repair actions based on fault context.
    """
    def __init__(self, model_name: str, api_token: str):
        if not api_token:
            # Fallback to a simple rule-based system if no token
            self.llm = None
            print("Warning: No Hugging Face token provided. Using fallback suggestions.")
        else:
            try:
                self.llm = HuggingFaceHub(
                    repo_id=model_name,
                    huggingfacehub_api_token=api_token,
                    model_kwargs={"temperature": 0.7, "max_length": 512}
                )
            except Exception as e:
                print(f"Error initializing Hugging Face LLM: {e}")
                self.llm = None
        
        self.prompt = PromptTemplate(
            input_variables=["fault_context"],
            template="""Given the following industrial equipment fault context, provide specific and actionable repair recommendations:

Context: {fault_context}

Please provide:
1. Immediate actions to take
2. Preventive measures
3. When to schedule maintenance

Repair Recommendations:"""
        )

    def suggest(self, fault_context: str) -> str:
        """
        Generates repair suggestions from the LLM or fallback system.
        Args:
            fault_context (str): Description of the fault and context.
        Returns:
            str: Suggested repair actions.
        """
        if self.llm:
            try:
                prompt_text = self.prompt.format(fault_context=fault_context)
                response = self.llm.invoke(prompt_text)
                return response
            except Exception as e:
                print(f"Error getting LLM suggestion: {e}")
                return self._fallback_suggestion(fault_context)
        else:
            return self._fallback_suggestion(fault_context)
    
    def _fallback_suggestion(self, fault_context: str) -> str:
        """
        Provides basic rule-based suggestions when LLM is not available.
        """
        suggestions = []
        
        # Extract key information from context
        if "temperature" in fault_context.lower():
            if "high" in fault_context.lower() or any(str(i) in fault_context for i in range(85, 150)):
                suggestions.append("• Check cooling system and ventilation")
                suggestions.append("• Inspect for heat sources or blockages")
        
        if "pressure" in fault_context.lower():
            suggestions.append("• Check pressure relief valves")
            suggestions.append("• Inspect seals and gaskets for leaks")
        
        if "vibration" in fault_context.lower():
            suggestions.append("• Check bearing alignment and lubrication")
            suggestions.append("• Inspect for loose mountings or imbalance")
        
        if "fault probability" in fault_context.lower():
            prob_str = fault_context.split("Fault probability:")[-1].split()[0]
            try:
                prob = float(prob_str)
                if prob > 0.7:
                    suggestions.append("• HIGH PRIORITY: Schedule immediate inspection")
                    suggestions.append("• Consider stopping operation if critical")
                elif prob > 0.4:
                    suggestions.append("• Schedule maintenance within 24-48 hours")
                else:
                    suggestions.append("• Continue monitoring, schedule routine maintenance")
            except:
                pass
        
        if not suggestions:
            suggestions = [
                "• Perform visual inspection of equipment",
                "• Check all connections and fasteners",
                "• Review maintenance logs for patterns",
                "• Consider consulting equipment manual"
            ]
        
        return "\n".join(suggestions) 