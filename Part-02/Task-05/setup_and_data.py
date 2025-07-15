import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_f1_support, classification_report
import torch
import random
from typing import List, Dict
import uuid

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Synthetic dataset creation
def create_synthetic_dataset(n_samples=1000):
    categories = ["Billing", "Technical", "Account", "Product Inquiry", "Other"]
    tickets = []
    for _ in range(n_samples):
        if random.random() < 0.2:
            ticket = "Issue with payment processing, charged twice."
            label = "Billing"
        elif random.random() < 0.4:
            ticket = "App crashes on startup, error code 503."
            label = "Technical"
        elif random.random() < 0.6:
            ticket = "Cannot log into my account, password reset not working."
            label = "Account"
        elif random.random() < 0.8:
            ticket = "Question about product features and pricing."
            label = "Product Inquiry"
        else:
            ticket = "General feedback about the service."
            label = "Other"
        tickets.append({"text": ticket, "label": label})
    return pd.DataFrame(tickets)

# Create and split dataset
df = create_synthetic_dataset()
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Save dataset for reproducibility
train_df.to_csv("train_tickets.csv", index=False)
test_df.to_csv("test_tickets.csv", index=False)

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)