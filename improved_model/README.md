# Improved Transformer Variant

This directory hosts a deeper-but-narrower transformer that keeps compute roughly constant while adding capacity. The encoder depth doubles (3 → 6 layers) and the model width is trimmed (192 → 160 with 5 heads) to balance attention cost. To prevent the early overfitting seen in the first attempt, this variant adds stronger dropout, decoupled weight decay, a cosine+warmup LR schedule, and early stopping.

## Train
```bash
conda activate 612proj
python improved_model/train.py
```
Artifacts:
- `improved_model/best_model_deepened.pth`
- Training/validation loss logs in the console

## Evaluate
```bash
python improved_model/evaluate.py
```
Uses the shared `processed_data.npz` at the repo root and prints regression plus rain-event metrics.
