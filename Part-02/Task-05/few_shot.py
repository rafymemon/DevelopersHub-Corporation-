def few_shot_prompt(ticket: str, examples: List[Dict], categories: List[str]) -> str:
    prompt = "Classify the following support ticket into one of these categories: " + ", ".join(categories) + ".\n"
    prompt += "Provide the top 3 most probable categories with their confidence scores (0 to 1).\n"
    prompt += "Examples:\n"
    for ex in examples:
        prompt += f"Ticket: {ex['text']}\nCategories: {ex['label']}\n\n"
    prompt += f"Ticket: {ticket}\nOutput format: Category: <category>, Confidence: <score>\n"
    return prompt

def few_shot_predict(ticket: str, examples: List[Dict], categories: List[str]) -> List[Dict[str, float]]:
    prompt = few_shot_prompt(ticket, examples, categories)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse response
    result = []
    lines = response.split("\n")
    for line in lines:
        if "Category:" in line and "Confidence:" in line:
            try:
                category = line.split("Category: ")[1].split(",")[0].strip()
                confidence = float(line.split("Confidence: ")[1].strip())
                if category in categories:
                    result.append({"category": category, "confidence": confidence})
            except:
                continue
    while len(result) < 3:
        remaining = [c for c in categories if not any(r["category"] == c for r in result)]
        if remaining:
            result.append({"category": remaining[0], "confidence": 0.1})
    return result[:3]

# Select 5 examples (1 per category)
examples = [
    {"text": "Charged twice for my subscription.", "label": "Billing"},
    {"text": "App keeps crashing on launch.", "label": "Technical"},
    {"text": "Need to update my account email.", "label": "Account"},
    {"text": "What are the features of your premium plan?", "label": "Product Inquiry"},
    {"text": "Suggestion for improving the UI.", "label": "Other"}
]

# Example usage
few_shot_results = few_shot_predict(sample_ticket, examples, categories)
print("Few-Shot Results:", few_shot_results)