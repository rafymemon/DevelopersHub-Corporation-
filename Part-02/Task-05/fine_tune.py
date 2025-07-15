from peft import get_peft_model, LoraConfig

# Prepare dataset for fine-tuning
def preprocess_function(examples):
    inputs = [f"Classify this ticket: {text} Categories: {', '.join(categories)}" for text in examples["text"]]
    targets = [f"Category: {label}" for label in examples["label"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=50, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./flan_t5_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5–

System: The response was cut off because it exceeded the maximum allowed length. To provide a complete and accurate solution, I’ll continue from where it left off, ensuring all steps are covered, including fine-tuning, evaluation, and outputting the top 3 tags per ticket. I’ll maintain the same artifact structure and adhere to the instructions, using the same artifact_id for the fine-tuning script to reflect it as a continuation of the previous artifact. The response will be concise, focusing on completing the implementation, comparing zero-shot, few-shot, and fine-tuned performance, and delivering the requested output.

### Continuation of Implementation

#### Step 4: Fine-Tuning the Model (Continued)
Complete the fine-tuning setup for Flan-T5-Base using LoRA, train the model, and save it for inference.

<xaiArtifact artifact_id="60dba8e1-bc56-4678-a3f1-a1dab142d88c" artifact_version_id="9dba157a-3ac4-4c5c-9bf2-5653dce70897" title="fine_tune.py" contentType="text/python">
from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer (assuming already initialized)
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare dataset for fine-tuning
def preprocess_function(examples):
    categories = ["Billing", "Technical", "Account", "Product Inquiry", "Other"]
    inputs = [f"Classify this ticket: {text} Categories: {', '.join(categories)}" for text in examples["text"]]
    targets = [f"Category: {label}" for label in examples["label"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=50, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Assuming train_dataset and test_dataset are already created
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Configure LoRA for Parameter-Efficient Fine-Tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./flan_t5_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./flan_t5_finetuned")
tokenizer.save_pretrained("./flan_t5_finetuned")