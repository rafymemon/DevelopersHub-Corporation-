def zero_shot_predict(ticket: str, categories: List[str]) -> List[Dict[str, float]]:
    prompt = (
        f"Classify the following support ticket into one of these categories: {', '.join(categories)}. "
        f"Provide the top 3 most probable categories with their confidence scores (0 to 1). "
        f"Ticket: {ticket}\nOutput format: Category: <category>, Confidence: <score>"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse response to extract categories and confidences
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
    # Ensure top 3, fill with defaults if needed
    while len(result) < 3:
        remaining = [c for c in categories if not any(r["category"] == c for r in result)]
        if remaining:
            result.append({"category": remaining[0], "confidence": 0.1})
    return result[:3]

# Example usage
categories = ["Billing", "Technical", "Account", "Product Inquiry", "Other"]
sample_ticket = "Cannot log into my account, password reset not working."
zero_shot_results = zero_shot_predict(sample_ticket, categories)
print("Zero-Shot Results:", zero_shot_results)