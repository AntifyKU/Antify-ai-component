# scripts/evaluate_model.py
# Standalone evaluation & visualization for BioCLIP models.
# Generates: confusion matrix, F1-score charts, prediction grid,
# training history plots (if available), and a text report.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

# Allow imports from the scripts directory
sys.path.insert(0, os.path.dirname(__file__))
from train_bioclip import AntDataset, stratified_split, maybe_filter_valid_images
from bioclip_model import BioCLIPClassifier

# ─── Styling ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
COLORS = {"train": "#2196F3", "val": "#FF5722"}


# ─── Helpers ────────────────────────────────────────────────────────
def load_checkpoint(model_path, device):
    """Load checkpoint and return (checkpoint_dict, model)."""
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    classes = ckpt.get("classes")
    if classes is None:
        class_file = os.path.join(os.path.dirname(model_path), "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            raise RuntimeError("Cannot determine class list from checkpoint or classes.txt")

    model = BioCLIPClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return ckpt, model, classes


def run_inference(model, loader, device, classes):
    """Run inference and return (all_preds, all_labels, sample_images)."""
    all_preds = []
    all_labels = []
    sample_images = []
    sample_true = []
    sample_pred = []

    # CLIP normalization for de-normalizing display images
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running Inference"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Collect samples for the prediction grid (first 16)
            if len(sample_images) < 16:
                imgs_denorm = images * std + mean
                imgs_denorm = torch.clamp(imgs_denorm, 0, 1)
                for i in range(min(16 - len(sample_images), len(images))):
                    sample_images.append(imgs_denorm[i].cpu())
                    sample_true.append(classes[labels[i]])
                    sample_pred.append(classes[predicted[i]])

    return (
        np.array(all_preds),
        np.array(all_labels),
        sample_images,
        sample_true,
        sample_pred,
    )


# ─── Plot: Training History ────────────────────────────────────────
def plot_training_history(history, output_dir):
    """3-panel plot: loss, accuracy, F1-score across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", color=COLORS["train"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s-", color=COLORS["val"], label="Val Loss", linewidth=2)
    ax.set_title("Loss per Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "o-", color=COLORS["train"], label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"], "s-", color=COLORS["val"], label="Val Acc", linewidth=2)
    ax.set_title("Accuracy per Epoch (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    # F1
    ax = axes[2]
    ax.plot(epochs, history["train_f1"], "o-", color=COLORS["train"], label="Train F1", linewidth=2)
    ax.plot(epochs, history["val_f1"], "s-", color=COLORS["val"], label="Val F1", linewidth=2)
    ax.set_title("Macro F1-Score per Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-Score")
    ax.legend()

    plt.suptitle("Training History", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training history → {path}")


# ─── Plot: Overall Performance ─────────────────────────────────────
def plot_overall_performance(acc, macro_f1, weighted_f1, output_dir):
    """Simple bar chart showing overall accuracy and F1 scores."""
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    # Convert acc out of 100 to out of 1.0 for the same scale
    values = [acc / 100.0, macro_f1, weighted_f1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall Model Performance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
                
    plt.tight_layout()
    path = os.path.join(output_dir, "overall_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved overall performance → {path}")


# ─── Plot: Confusion Matrix ────────────────────────────────────────
def plot_confusion_matrix(cm, classes, output_dir, top_n=None):
    """Full confusion matrix heatmap, optionally filtered to top_n most confused classes."""
    if top_n and len(classes) > top_n:
        # Find the most confused classes (highest off-diagonal totals)
        off_diag = cm.copy().astype(float)
        np.fill_diagonal(off_diag, 0)
        confusion_per_class = off_diag.sum(axis=0) + off_diag.sum(axis=1)
        top_indices = np.argsort(confusion_per_class)[-top_n:]
        top_indices = np.sort(top_indices)
        cm_sub = cm[np.ix_(top_indices, top_indices)]
        labels_sub = [classes[i] for i in top_indices]
        suffix = f"_top{top_n}"
        title = f"Confusion Matrix (Top {top_n} Most Confused Classes)"
    else:
        cm_sub = cm
        labels_sub = classes
        suffix = ""
        title = "Confusion Matrix"

    n = len(labels_sub)
    fig_size = max(10, n * 0.35)  # scale figure with num classes
    annot = n <= 30  # annotate cells only if not too many

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_sub,
        annot=annot,
        fmt="d" if annot else "",
        cmap="Blues",
        xticklabels=labels_sub,
        yticklabels=labels_sub,
        ax=ax,
        linewidths=0.5 if annot else 0,
        square=True,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=90, fontsize=max(6, 10 - n // 20))
    plt.yticks(rotation=0, fontsize=max(6, 10 - n // 20))
    plt.tight_layout()

    path = os.path.join(output_dir, f"confusion_matrix{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {path}")


# ─── Plot: Per-Class F1 Bar Chart ──────────────────────────────────
def plot_f1_per_class(all_labels, all_preds, classes, output_dir, top_n=10):
    """Horizontal bar chart of per-class F1-scores, showing only the best and worst N classes."""
    f1_per = f1_score(
        all_labels, all_preds, average=None,
        labels=range(len(classes)), zero_division=0,
    )

    # Build dataframe and sort
    df = pd.DataFrame({"class": classes, "f1": f1_per})
    
    # Filter out classes that have 0 support in the true labels (they skew the 'worst' list)
    true_counts = np.bincount(all_labels, minlength=len(classes))
    df['support'] = true_counts
    df_valid = df[df['support'] > 0].copy()
    
    if len(df_valid) > top_n * 2:
        df_sorted = df_valid.sort_values("f1", ascending=True)
        # Take worst N and best N
        worst = df_sorted.head(top_n)
        best = df_sorted.tail(top_n)
        df_plot = pd.concat([worst, best])
        title_suffix = f" (Worst {top_n} & Best {top_n})"
    else:
        df_plot = df_valid.sort_values("f1", ascending=True)
        title_suffix = ""

    # Colour by score: red (<0.5) → yellow (0.5-0.8) → green (>0.8)
    colors = []
    for v in df_plot["f1"]:
        if v < 0.5:
            colors.append("#EF5350")
        elif v < 0.8:
            colors.append("#FFC107")
        else:
            colors.append("#4CAF50")

    n = len(df_plot)
    fig_height = max(6, n * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(range(n), df_plot["f1"].values, color=colors, edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels(df_plot["class"].values, fontsize=max(8, 10 - n // 25))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title(f"Per-Class F1-Score{title_suffix}", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="0.8 threshold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(output_dir, "f1_scores_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved F1 per class → {path}")


# ─── Plot: Prediction Grid ─────────────────────────────────────────
def plot_prediction_grid(sample_images, sample_true, sample_pred, output_dir):
    """4×4 grid of sample predictions with green (correct) / red (wrong) titles."""
    n = min(len(sample_images), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = sample_images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            is_correct = sample_true[i] == sample_pred[i]
            color = "green" if is_correct else "red"
            icon = "✓" if is_correct else "✗"
            ax.set_title(
                f"{icon}  True: {sample_true[i]}\n    Pred: {sample_pred[i]}",
                color=color,
                fontsize=8,
                fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle("Validation Predictions Sample", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "val_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved predictions grid → {path}")


# ─── Text Report ───────────────────────────────────────────────────
def write_report(
    output_dir,
    model_path,
    csv_file,
    num_val,
    classes,
    all_labels,
    all_preds,
    history,
    ckpt,
):
    """Write a comprehensive text report."""
    all_label_ids = list(range(len(classes)))
    acc = 100.0 * np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=all_label_ids, zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", labels=all_label_ids, zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", labels=all_label_ids, zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", labels=all_label_ids, zero_division=0)

    cls_report = classification_report(
        all_labels, all_preds, labels=all_label_ids,
        target_names=classes, digits=4, zero_division=0,
    )

    lines = []
    lines.append("=" * 70)
    lines.append("  BioCLIP Model Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"  Model:             {model_path}")
    lines.append(f"  Dataset:           {csv_file}")
    lines.append(f"  Num Classes:       {len(classes)}")
    lines.append(f"  Val Samples:       {num_val}")
    if "epoch" in ckpt:
        lines.append(f"  Best Epoch:        {ckpt['epoch'] + 1}")
    if "best_acc" in ckpt:
        lines.append(f"  Best Val Acc:      {ckpt['best_acc']:.2f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  Overall Metrics (on current evaluation)")
    lines.append("-" * 70)
    lines.append(f"  Accuracy:          {acc:.2f}%")
    lines.append(f"  Macro F1:          {macro_f1:.4f}")
    lines.append(f"  Weighted F1:       {weighted_f1:.4f}")
    lines.append(f"  Macro Precision:   {macro_prec:.4f}")
    lines.append(f"  Macro Recall:      {macro_rec:.4f}")

    if history:
        lines.append("")
        lines.append("-" * 70)
        lines.append("  Per-Epoch Training History")
        lines.append("-" * 70)
        header = f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Train F1':>8}  {'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}"
        lines.append(header)
        for i in range(len(history["train_loss"])):
            row = (
                f"  {i+1:>5}  "
                f"{history['train_loss'][i]:>10.4f}  "
                f"{history['train_acc'][i]:>8.2f}%  "
                f"{history['train_f1'][i]:>8.4f}  "
                f"{history['val_loss'][i]:>8.4f}  "
                f"{history['val_acc'][i]:>6.2f}%  "
                f"{history['val_f1'][i]:>6.4f}"
            )
            lines.append(row)

    lines.append("")
    lines.append("-" * 70)
    lines.append("  Per-Class Classification Report")
    lines.append("-" * 70)
    lines.append(cls_report)
    lines.append("=" * 70)

    report = "\n".join(lines)
    path = os.path.join(output_dir, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved report → {path}")

    # Also print summary to console
    print("\n" + "=" * 50)
    print(f" Accuracy:      {acc:.2f}%")
    print(f" Macro F1:      {macro_f1:.4f}")
    print(f" Weighted F1:   {weighted_f1:.4f}")
    print("=" * 50)


# ─── Main ───────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt, model, classes = load_checkpoint(args.model_path, device)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Load dataset
    print(f"Loading labels from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # Filter to classes present in the model
    df_filtered = df[df["scientific_name"].isin(classes)].copy()
    print(f"Filtered dataset: {len(df)} total → {len(df_filtered)} matching model classes.")

    if args.drop_missing_images:
        df_filtered = maybe_filter_valid_images(df_filtered, args.base_path)

    # Use validation split (same seed as training for consistency)
    _, val_df = stratified_split(df_filtered, args.val_ratio, args.seed)
    print(f"Validation samples: {len(val_df)}")

    # Optionally limit for speed
    if args.max_samples > 0 and len(val_df) > args.max_samples:
        val_df = val_df.sample(n=args.max_samples, random_state=args.seed)
        print(f"Limited to {args.max_samples} samples for evaluation.")

    # DataLoader
    _, preprocess_val = model.get_transforms()
    val_dataset = AntDataset(val_df, preprocess_val, class_to_idx, args.base_path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    # Run inference
    all_preds, all_labels, sample_images, sample_true, sample_pred = run_inference(
        model, val_loader, device, classes
    )

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))

    # Check if checkpoint has training history
    history = ckpt.get("history")

    # ─── Generate all outputs ───
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating outputs to {args.output_dir}/")

    # 1. Training history (if available)
    if history and len(history.get("train_loss", [])) > 0:
        plot_training_history(history, args.output_dir)
    else:
        print("  ⚠ No training history in checkpoint — skipping history plot.")
        print("    (Retrain with the updated train_bioclip.py to get epoch history)")

    # 2. Confusion matrix (top-10 most confused)
    plot_confusion_matrix(cm, classes, args.output_dir, top_n=10)

    # 3. F1 per class (top 10 best and worst)
    plot_f1_per_class(all_labels, all_preds, classes, args.output_dir, top_n=10)

    # 4. Overall performance bar chart
    acc = 100.0 * np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=range(len(classes)), zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", labels=range(len(classes)), zero_division=0)
    plot_overall_performance(acc, macro_f1, weighted_f1, args.output_dir)

    # 5. Prediction grid
    plot_prediction_grid(sample_images, sample_true, sample_pred, args.output_dir)

    # 6. Text report
    write_report(
        args.output_dir,
        args.model_path,
        args.csv_file,
        len(val_df),
        classes,
        all_labels,
        all_preds,
        history,
        ckpt,
    )

    print(f"\n✓ All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate & visualize a BioCLIP classifier model"
    )
    parser.add_argument(
        "--model-path",
        default="models/bioclip_finetuned.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--csv-file",
        default="combined_labels.csv",
        help="Path to CSV with image_path and scientific_name columns",
    )
    parser.add_argument("--base-path", default=".", help="Base path for images")
    parser.add_argument("--output-dir", default="results", help="Directory for outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Max val samples (0=all)")
    parser.add_argument(
        "--drop-missing-images",
        action="store_true",
        help="Drop rows with missing image files",
    )

    args = parser.parse_args()
    evaluate(args)
