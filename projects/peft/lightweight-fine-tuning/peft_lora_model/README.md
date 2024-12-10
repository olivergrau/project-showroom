---
base_model: bert-base-uncased
library_name: peft
---

# Model Card for Lightweight Fine-Tuned BERT with LoRA

This model is a lightweight fine-tuned version of `bert-base-uncased` using Parameter-Efficient Fine-Tuning (PEFT) with the LoRA method. It is designed for sequence classification tasks, such as sentiment analysis, and fine-tuned on a subset of the IMDb movie reviews dataset.

## Model Details

### Model Description

This model leverages LoRA (Low-Rank Adaptation) to fine-tune the BERT-base architecture efficiently. LoRA reduces the computational and memory overhead of fine-tuning by introducing trainable low-rank matrices to specific model layers (e.g., attention layers). 

- **Model type:** Transformer-based sequence classification model.
- **Language(s) (NLP):** English.
- **License:** Apache 2.0 (inherited from `bert-base-uncased`).
- **Finetuned from model:** `bert-base-uncased`.

### Model Sources

- **Repository:** [Hugging Face Model Hub](https://huggingface.co/bert-base-uncased)
- **Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Uses

### Direct Use

This model can be directly used for tasks requiring sentiment classification on English text, such as classifying movie reviews as positive or negative. 

### Downstream Use

The model is intended for fine-tuning or inference in NLP tasks where lightweight and efficient training is desired.

### Out-of-Scope Use

The model is not suitable for tasks outside its pretraining domain or in languages other than English.

## Bias, Risks, and Limitations

While this model is fine-tuned on IMDb data, it may inherit biases from the original `bert-base-uncased` model or the fine-tuning dataset (e.g., movie reviews predominantly in English). 

### Recommendations

Users should be aware of the potential biases and validate the model on diverse datasets before deployment in critical applications.

## How to Get Started with the Model

Use the code below to get started with the model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
peft_model = PeftModel.from_pretrained(model, "./peft_lora_model")

# Tokenize input
text = "The movie was amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Make predictions
outputs = peft_model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(dim=-1).item()
print("Predicted class:", predicted_class)
```

## Training Details

## Training Details

### Training Data

This model was fine-tuned on a subset of the IMDb dataset containing 1,000 training examples and 500 validation examples.

### Training Procedure

The model was trained for 5 epochs with the following configuration:
- Batch size: 16
- Learning rate: 5e-5
- Weight decay: 0.01
- Loss function: CrossEntropyLoss

#### Training Hyperparameters

- **Training regime:** fp32 precision
- **Optimizer:** AdamW

#### Speeds, Sizes, Times

- **Training time:** Approximately 20 minutes on an NVIDIA 4070 laptop GPU.
- **Checkpoint size:** Lightweight LoRA weights (~1 MB).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on a held-out test set from the IMDb dataset, consisting of 500 examples.

#### Metrics

Accuracy was used as the primary metric to evaluate the model's performance.

### Results

- **Validation Accuracy:** ~84%
- **Validation Loss:** ~0.51

#### Summary

The lightweight fine-tuning approach (LoRA) achieved competitive performance with minimal computational overhead, making it suitable for resource-constrained environments.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA 4070 laptop GPU.
- **Hours used:** ~0.5 hours.
- **Cloud Provider:** Local hardware.
- **Compute Region:** N/A.
- **Carbon Emitted:** Negligible.

## Technical Specifications

### Model Architecture and Objective

This model uses the BERT-base architecture with LoRA adapters applied to attention layers (`query` and `value`).

### Compute Infrastructure

#### Hardware

- NVIDIA GeForce RTX 4070 Laptop GPU with 8 GB VRAM.

#### Software

- Python 3.9
- PyTorch 2.0
- Hugging Face Transformers 4.33
- PEFT 0.13.2