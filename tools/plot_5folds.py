#!/usr/bin/env python3
"""
tools/plot_5folds.py
─────────────────────
Generates per-fold training curves and normalized confusion matrices
from S4D/GroupMamba-DFER training logs.

Usage:
    python tools/plot_5folds.py --dataset dfew --output_dir ./results/plots
    python tools/plot_5folds.py --dataset ferv39k --output_dir ./results/plots
"""
import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

LABELS_7 = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
LABELS_11 = ['Anger', 'Anxiety', 'Contempt', 'Disgust', 'Embarrass', 'Fear',
             'Happiness', 'Neutral', 'Pain', 'Sadness', 'Surprise']


def parse_log(log_path):
    train_loss, val_loss, war, uar = [], [], [], []
    best_cm_lines = []

    with open(log_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('Confusion Matrix:'):
            best_cm_lines = lines[i + 1:i + 15]
        if line.startswith('Averaged stats:'):
            m = re.search(r'loss: \d+\.\d+ \(([\d\.]+)\)', line)
            if m:
                train_loss.append(float(m.group(1)))
        if line.startswith('* Acc@1') and 'SFER' not in lines[max(0, i-1)]:
            m = re.search(r'loss ([\d\.]+)', line)
            if m:
                val_loss.append(float(m.group(1)))
        if line.startswith('* WAR') and 'SFER' not in lines[max(0, i-1)]:
            m = re.search(r'WAR ([\d\.]+) UAR ([\d\.]+)', line)
            if m:
                war.append(float(m.group(1)) * 100)
                uar.append(float(m.group(2)) * 100)

    cm = []
    for line in best_cm_lines:
        row = [int(x) for x in re.findall(r'\d+', line)]
        if row:
            cm.append(row)

    return (np.array(train_loss), np.array(val_loss),
            np.array(war), np.array(uar), np.array(cm) if cm else None)


def plot_split(split, train_loss, val_loss, war, uar, cm, labels, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix
    if cm is not None and len(cm) == len(labels):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=labels, yticklabels=labels, ax=ax, square=True)
        ax.set_title(f'Normalized Confusion Matrix — {dataset.upper()} Split {split}', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_split{split}_confusion_matrix.png'), dpi=300)
        plt.close()

    # Training curves
    epochs = np.arange(len(war))
    min_len = min(len(train_loss), len(val_loss), len(war))
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs[:min_len], train_loss[:min_len], label='Train Loss', linewidth=2)
    ax_loss.plot(epochs[:min_len], val_loss[:min_len], label='Val Loss', linewidth=2)
    ax_loss.set_title(f'Loss — {dataset.upper()} Split {split}', fontsize=12)
    ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss')
    ax_loss.legend(frameon=True, shadow=True); ax_loss.grid(False)

    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.plot(epochs[:min_len], war[:min_len], label='WAR', color='firebrick', linewidth=2)
    ax_acc.plot(epochs[:min_len], uar[:min_len], label='UAR', color='seagreen', linewidth=2)
    ax_acc.set_title(f'WAR & UAR — {dataset.upper()} Split {split}', fontsize=12)
    ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend(frameon=True, shadow=True); ax_acc.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_split{split}_training_curves.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dfew', 'ferv39k', 'mafw'], required=True)
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Root dir containing split1..split5 subdirectories with train.log')
    parser.add_argument('--output_dir', type=str, default='./results/plots')
    args = parser.parse_args()

    labels = LABELS_11 if args.dataset == 'mafw' else LABELS_7
    wars, uars = [], []

    for split in range(1, 6):
        log_path = os.path.join(args.log_dir, f'split{split}', 'train.log')
        if not os.path.exists(log_path):
            # Try flat layout
            log_path = os.path.join(args.log_dir, f'split{split}_train.log')
        if not os.path.exists(log_path):
            print(f'  [!] Log not found for {args.dataset} Split {split}, skipping.')
            continue

        train_loss, val_loss, war, uar, cm = parse_log(log_path)
        print(f'  Split {split}: max WAR={max(war):.2f}%  max UAR={max(uar):.2f}%')
        wars.append(max(war)); uars.append(max(uar))
        plot_split(split, train_loss, val_loss, war, uar, cm, labels, args.output_dir, args.dataset)

    if wars:
        print(f'\n{args.dataset.upper()} Summary:')
        print(f'  Mean WAR = {np.mean(wars):.2f}% ± {np.std(wars, ddof=1):.2f}')
        print(f'  Mean UAR = {np.mean(uars):.2f}% ± {np.std(uars, ddof=1):.2f}')


if __name__ == '__main__':
    main()
