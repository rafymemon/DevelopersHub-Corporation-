# 📰 News Topic Classifier Using BERT

This project fine-tunes a transformer model (`bert-base-uncased`) to classify news headlines into topic categories using the AG News Dataset. The model is trained using Hugging Face Transformers and deployed using Gradio for live interaction.

---

## 📌 Task Overview

**Objective**  
Build a robust text classification system that identifies the topic category of news headlines.

**Dataset**  
[AG News Dataset](https://huggingface.co/datasets/ag_news) — contains news headlines categorized into:

- 🌍 World
- ⚽ Sports
- 💼 Business
- 🧪 Sci/Tech

---

## 🚀 Project Pipeline

### 1. 📊 Dataset Loading & Exploration

- Load AG News using Hugging Face `datasets` library
- Examine class balance and sample entries

### 2. 🧪 Data Preprocessing

- Tokenization using `BertTokenizerFast`
- Truncation & padding for input compatibility with BERT
- Conversion to PyTorch tensors

### 3. 🤖 Model Fine-tuning

- Model: `bert-base-uncased` with classification head
- Loss Function: Cross-Entropy
- Optimizer: AdamW
- Scheduler: Linear learning rate decay
- Trained for 3 epochs on 4 classes

### 4. 🧠 Evaluation

- Evaluation metrics:
  - **Accuracy**
  - **F1-score (weighted)**
- Used Hugging Face `Trainer` API with `compute_metrics()`

### 5. 🌐 Deployment

- Deployed as a live demo using [Gradio](https://gradio.app/)
- Interactive UI for classifying custom news headlines

---

## 📈 Results

| Metric      | Score |
|-------------|-------|
| Accuracy    | 93-96% (depending on hyperparameters) |
| F1-Score    | 0.93+ (weighted average) |

---

## 💡 Skills Gained

- 🧠 NLP using Transformer Models (BERT)
- 🔁 Transfer Learning & Fine-tuning
- 📊 Evaluation of Text Classification Models
- 🚀 Lightweight Deployment with Gradio

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| [Transformers](https://huggingface.co/transformers/) | Pretrained BERT & fine-tuning |
| [Datasets](https://huggingface.co/docs/datasets/) | Loading AG News |
| [PyTorch](https://pytorch.org/) | Model backend |
| [Scikit-learn](https://scikit-learn.org/) | Evaluation metrics |
| [Gradio](https://gradio.app/) | Interactive model deployment |

---

## 🛠 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/news-topic-classifier
   cd news-topic-classifier
