# Model Card — IMDb Sentiment (BERT + LoRA/QLoRA)

**Date:** 2025-08-09  
**Base model:** `bert-base-uncased`  
**Task:** Binary sentiment classification (IMDb)  
**Method:** PEFT (LoRA / QLoRA)  

## Training
- LoRA rank: 8, alpha: 32, dropout: 0.1
- Target modules: query, key, value
- Optimizer: default Trainer (AdamW) @ 5e-5
- Epochs: 3
- Batch size: 16
- Max length: 256
- Dataset subsampling: train=5000, test=2500 (toggle as needed)

## Evaluation
Report Accuracy and Macro-F1 on the test split. Include confusion matrix and error notes.
(Place your actual numbers here after a run.)

## Artifacts
- Adapters: `outputs_peft/adapters_lora/` (or `adapters_qlora/` when QLoRA)
- (Optional) Merged model: `outputs_peft/merged_lora_model/` (not for QLoRA)

## Usage
Load base + adapters for inference:
```bash
python inference_peft.py --base bert-base-uncased --adapters outputs_peft/adapters_lora --text "Great film!"
```

## Limitations & Risks
- Binary sentiment simplifies nuance. Consider calibration and threshold analysis.
- For broader tasks (instructions, tool use), switch to instruction SFT and add safety eval.

## License & Attribution
- Base model license: see Hugging Face hub for `bert-base-uncased`.
- Dataset: IMDb (check its terms).

## Changelog
- 2025-08-09: Initial upgraded release.
