# Auto-Tagging Support Tickets Using LLM

## Objective

Automatically tag support tickets into categories (Billing, Technical, Account, Product Inquiry, Other) using a large language model (LLM). Compare zero-shot, few-shot, and fine-tuned approaches, and output the top 3 most probable tags per ticket.

## Dataset

A synthetic dataset of 1000 support tickets with free-text descriptions and one true label per ticket. Categories: Billing, Technical, Account, Product Inquiry, Other.

## Approach

1. **Zero-Shot Learning**: Use prompt engineering with Flan-T5-Base to classify tickets without training.
2. **Few-Shot Learning**: Include 5 labeled examples in the prompt to improve classification accuracy.
3. **Fine-Tuning**: Fine-tune Flan-T5-Base using LoRA on the training dataset for task-specific optimization.
4. **Evaluation**: Compare performance using precision, recall, and F1-score.
5. **Output**: Generate top 3 tags with confidence scores for each ticket.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Hardware**: CPU (GPU optional for faster training).
- **Libraries**:
  - pandas==2.2.2
  - numpy==1.26.4
  - transformers==4.44.2
  - datasets==3.0.0
  - torch==2.4.0
  - peft==0.12.0
  - scikit-learn==1.5.1

Install libraries:

```bash
pip install pandas==2.2.2 numpy==1.26.4 transformers==4.44.2 datasets==3.0.0 torch==2.4.0 peft==0.12.0 scikit-learn==1.5.1
```

For GPU support:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

## Setup Instructions

1. **Create a Virtual Environment** (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Libraries**: See above.
3. **Download Scripts**: Save the following scripts from the implementation:
   - `setup_and_data.py`: Creates synthetic dataset and initializes model.
   - `zero_shot.py`: Implements zero-shot classification.
   - `few_shot.py`: Implements few-shot classification.
   - `fine_tune.py`: Fine-tunes the model using LoRA.
   - `fine_tuned_inference.py`: Performs inference with the fine-tuned model.
   - `evaluate.py`: Evaluates and compares performance.

## Running the Scripts

1. **Generate Dataset**:

   ```bash
   python setup_and_data.py
   ```

   Creates `train_tickets.csv` and `test_tickets.csv`.

2. **Zero-Shot Classification**:

   ```bash
   python zero_shot.py
   ```

   Outputs top 3 tags for a sample ticket using zero-shot prompting.

3. **Few-Shot Classification**:

   ```bash
   python few_shot.py
   ```

   Uses 5 labeled examples to predict top 3 tags.

4. **Fine-Tuning**:

   ```bash
   python fine_tune.py
   ```

   Trains Flan-T5-Base with LoRA and saves the model to `./flan_t5_finetuned`.

5. **Fine-Tuned Inference**:

   ```bash
   python fine_tuned_inference.py
   ```
  
   Predicts top 3 tags using the fine-tuned model.

6. **Evaluate Performance**:

   ```bash
   python evaluate.py
   ```

   Outputs precision, recall, F1-score for zero-shot, few-shot, and fine-tuned models, and saves sample predictions to `sample_predictions.csv`.

## Expected Output

- **Metrics**: Fine-tuned model typically achieves the highest F1-score, followed by few-shot, then zero-shot.
- **Sample Predictions** (`sample_predictions.csv`):
  - Columns: Ticket, Zero-Shot Top 3, Few-Shot Top 3, Fine-Tuned Top 3, True Label.
  - Example:
    - Ticket: "Cannot log into my account, password reset not working."
    - Zero-Shot Top 3: [{"category": "Account", "confidence": 0.8}, {"category": "Technical", "confidence": 0.15}, {"category": "Other", "confidence": 0.1}]
    - Few-Shot Top 3: [{"category": "Account", "confidence": 0.9}, {"category": "Technical", "confidence": 0.08}, {"category": "Billing", "confidence": 0.02}]
    - Fine-Tuned Top 3: [{"category": "Account", "confidence": 0.95}, {"category": "Technical", "confidence": 0.04}, {"category": "Other", "confidence": 0.01}]
    - True Label: Account

## Troubleshooting

- **ModuleNotFoundError**: Ensure libraries are installed in the active environment (`pip list`).
- **CUDA Errors**: Verify CUDA installation or set `device = "cpu"` in scripts.
- **Model Download Issues**: Check internet connection or use cached models (`HF_HUB_OFFLINE=1`).
- **Low Accuracy**: Increase training epochs in `fine_tune.py` or add more examples in `few_shot.py`.

## Skills Gained

- Prompt engineering for zero-shot and few-shot learning.
- LLM-based text classification with Flan-T5.
- Fine-tuning with LoRA for efficient model training.
- Multi-class prediction and ranking with confidence scores.

For API-related queries, visit [xAI API](https://x.ai/api).
