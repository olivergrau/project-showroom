# inference_peft.py
# Usage example:
#   python inference_peft.py --base bert-base-uncased --adapters outputs_peft/adapters_lora --text "Great film!"

import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def load_model(base: str, adapters_dir: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)
    peft_model = PeftModel.from_pretrained(base_model, adapters_dir).to(device)
    peft_model.eval()
    return tok, peft_model, device

def classify(text: str, tok, model, device: str):
    with torch.no_grad():
        enc = tok(text, truncation=True, max_length=256, return_tensors='pt').to(device)
        out = model(**enc)
        pred = out.logits.argmax(dim=-1).item()
    return pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='base model name or path')
    ap.add_argument('--adapters', required=True, help='path to saved adapters')
    ap.add_argument('--text', required=True, help='input text to classify')
    args = ap.parse_args()

    tok, model, device = load_model(args.base, args.adapters)
    pred = classify(args.text, tok, model, device)
    print(f'Prediction (0=neg,1=pos): {pred}')
