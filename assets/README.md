# Assets

This directory contains architecture diagrams and result figures for the thesis.

## Placeholder — Files to Add

| File | Description |
|------|-------------|
| `architecture.png` | Full model architecture figure from thesis Chapter 3 |
| `moae_block.png` | Sparse Mixture-of-Adapter Experts block diagram |
| `results/dfew_split*_confusion_matrix.png` | DFEW per-fold confusion matrices |
| `results/dfew_split*_training_curves.png` | DFEW per-fold training curves |
| `results/ferv39k_split*_confusion_matrix.png` | FERV39K per-fold confusion matrices |
| `results/mafw_split*_confusion_matrix.png` | MAFW per-fold confusion matrices |

> To generate the result figures, run:
> ```bash
> python tools/plot_5folds.py --dataset dfew --log_dir ./logs/dfew --output_dir ./assets/results
> ```
