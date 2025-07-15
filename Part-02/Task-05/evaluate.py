from sklearn.metrics import precision_recall_f1_support
import pandas as pd

# Load test dataset
test_df = pd.read_csv("test_tickets.csv")
categories = ["Billing", "Technical", "Account", "Product Inquiry", "Other"]

# Evaluation function
def evaluate_model(predict_fn, examples=None):
    y_true = test_df["label"].tolist()
    y_pred = []
    top_3_results = []
    
    for ticket in test_df["text"]:
        if examples:
            predictions = predict_fn(ticket, examples, categories)
        else:
            predictions = predict_fn(ticket, categories)
        top_3_results.append(predictions)
        y_pred.append(predictions[0]["category"])  # Top-1 for evaluation
    
    precision, recall, f1, _ = precision_recall_f1_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}, top_3_results

# Evaluate zero-shot
zero_shot_metrics, zero_shot_top_3 = evaluate_model(zero_shot_predict)
print("Zero-Shot Metrics:", zero_shot_metrics)

# Evaluate few-shot
examples = [
    {"text": "Charged twice for my subscription.", "label": "Billing"},
    {"text": "App keeps crashing on launch.", "label": "Technical"},
    {"text": "Need to update my account email.", "label": "Account"},
    {"text": "What are the features of your premium plan?", "label": "Product Inquiry"},
    {"text": "Suggestion for improving the UI.", "label": "Other"}
]
few_shot_metrics, few_shot_top_3 = evaluate_model(few_shot_predict, examples)
print("Few-Shot Metrics:", few_shot_metrics)

# Evaluate fine-tuned
fine_tuned_metrics, fine_tuned_top_3 = evaluate_model(fine_tuned_predict)
print("Fine-Tuned Metrics:", fine_tuned_metrics)

# Save top-3 predictions for a few sample tickets
sample_results = pd.DataFrame({
    "Ticket": test_df["text"].head(5),
    "Zero-Shot Top 3": [str(res) for res in zero_shot_top_3[:5]],
    "Few-Shot Top 3": [str(res) for res in few_shot_top_3[:5]],
    "Fine-Tuned Top 3": [str(res) for res in fine_tuned_top_3[:5]],
    "True Label": test_df["label"].head(5)
})
sample_results.to_csv("sample_predictions.csv", index=False)