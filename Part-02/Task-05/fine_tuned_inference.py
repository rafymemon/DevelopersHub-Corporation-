from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Dict

# Load fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("./flan_t5_finetuned")
tokenizer = T5Tokenizer.from_pretrained("./flan_t5_finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def fine_tuned_predict(ticket: str, categories: List[str]) -> List[Dict[str, float]]:
    prompt = f"Classify this ticket: {ticket} Categories: {', '.join(categories)}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7, num_beams=5, num_return_sequences=3)
    result = []
    
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        try:
            category = response.split("Category: ")[1].strip()
            if category in categories:
                # Assign confidence based on beam search ranking (simplified)
                confidence = 1.0 / (len(result) + 1)  # Higher rank = higher confidence
                result.append({"category": category, "confidence": confidence})
        except:
            continue
    
    # Ensure top 3, fill with defaults if needed
    while len(result) < 3:
        remaining = [c for c in categories if not any(r["category"] == c for r in result)]
        if remaining:
            result.append({"category": remaining[0], "confidence": 0.1})
    return result[:3]

# Example usage
categories = ["Billing", "Technical", "Account", "Product Inquiry", "Other"]
sample_ticket = "Cannot log into my account, password reset not working."
fine_tuned_results = fine_tuned_predict(sample_ticket, categories)
print("Fine-Tuned Results:", fine_tuned_results)