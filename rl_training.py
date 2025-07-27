import json
import random
from langchain_groq import ChatGroq

class RLTrainer:
    def __init__(self):
        self.feedback_file = "feedback_log.json"
        self.feedback_data = self._load_feedback()
        
    def _load_feedback(self):
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
            
    def save_feedback(self, century, user_input, continuation, feedback):
        """Store user feedback"""
        self.feedback_data.append({
            "century": century,
            "user_input": user_input,
            "continuation": continuation,
            "feedback": feedback
        })
        with open(self.feedback_file, "w") as f:
            json.dump(self.feedback_data, f, indent=2)
            
    def optimize_prompt(self, prompt_template):
        """Simple prompt optimization based on feedback"""
        if len(self.feedback_data) < 5:
            return prompt_template
            
        # Analyze positive feedback patterns
        good_examples = [e for e in self.feedback_data if e["feedback"] == 1]
        if not good_examples:
            return prompt_template
            
        # Enhance prompt with successful patterns
        example = random.choice(good_examples)
        enhanced_prompt = f"{prompt_template}\n\n**Successful Pattern Example**\n"\
                          f"Era: {example['century']}\n"\
                          f"User Input: {example['user_input']}\n"\
                          f"Good Continuation: {example['continuation']}"
        
        return enhanced_prompt
        
    def generate_with_feedback(self, prompt):
        """Generate text using feedback-enhanced prompt"""
        llm = ChatGroq(model_name="Llama3-70b-8192", temperature=0.8)
        return llm.invoke(prompt).content