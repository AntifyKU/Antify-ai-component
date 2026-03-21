# scripts/train_bioclip.py
# Full version: stratified split + trainable params print + 2-LR optimizer + logit_scale support (in model)
# + warmup freeze schedule + class-weighted loss (optional) + early stopping (optional)

import os
import argparse
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip

from bioclip_model import BioCLIPClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AntDataset(Dataset):
    def __init__(self, dataframe, preprocess, class_to_idx, base_path):
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocess = preprocess
        self.class_to_idx = class_to_idx
        self.base_path = base_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.base_path, row["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            # Better: pre-filter invalid images; this is fallback
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 224, 224), dtype=torch.float32)

        label_name = row["scientific_name"]
        label_idx = self.class_to_idx[label_name]
        return image, label_idx


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(model: BioCLIPClassifier, args):
    """Two-group LR: backbone_lr + head_lr (classifier + logit_scale)."""
    head_params = list(model.classifier.parameters())
    if getattr(model, "logit_scale", None) is not None:
        head_params += [model.logit_scale]

    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ]
    return optim.AdamW(param_groups, weight_decay=args.weight_decay)


def maybe_filter_valid_images(df, base_path):
    """Drop rows whose image file doesn't exist."""
    abs_paths = df["image_path"].apply(lambda p: os.path.join(base_path, p))
    exists = abs_paths.apply(os.path.exists)
    dropped = int((~exists).sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows with missing image files.")
    return df[exists].copy()


def make_criterion(train_df, classes, class_to_idx, device, args):
    """CrossEntropyLoss with optional class weighting (helps long-tail)."""
    if not args.class_weighting:
        return nn.CrossEntropyLoss()

    counts = train_df["scientific_name"].value_counts()
    weights = torch.zeros(len(classes), dtype=torch.float32)
    for cls, c in counts.items():
        idx = class_to_idx[cls]
        weights[idx] = 1.0 / (float(c) ** float(args.class_weight_power))

    weights = weights / weights.mean().clamp(min=1e-6)

    print(
        f"Using class-weighted CE (power={args.class_weight_power}). "
        f"min_w={weights.min().item():.4f} max_w={weights.max().item():.4f}"
    )
    return nn.CrossEntropyLoss(weight=weights.to(device))


def stratified_split(df, val_ratio, seed):
    """Stratified split with sklearn; fallback to random if impossible."""
    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:
        print("scikit-learn not found. Install with: pip install scikit-learn")
        raise e

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            stratify=df["scientific_name"],
        )
    except ValueError:
        print("Warning: Stratified split failed. Falling back to random split.")
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
        )
    return train_df, val_df


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load labels
    print(f"Loading labels from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    if args.filter_incomplete:
        print("Filtering out incomplete species names (sp. or Unknown)...")
        df = df[~df["scientific_name"].str.contains(r" sp|Unknown", regex=True)]

    if args.min_samples > 0:
        print(f"Filtering out classes with fewer than {args.min_samples} samples...")
        value_counts = df["scientific_name"].value_counts()
        to_keep = value_counts[value_counts >= args.min_samples].index
        df = df[df["scientific_name"].isin(to_keep)]
        print(f"Retained {len(to_keep)} classes after filtering.")

    if args.drop_missing_images:
        df = maybe_filter_valid_images(df, args.base_path)

    if args.debug:
        print("Debug mode enabled. Limiting dataset to 200 samples.")
        df = df.iloc[:200].copy()

    # Classes
    classes = sorted(df["scientific_name"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    print(f"Found {len(classes)} unique classes.")

    # Split
    train_df, val_df = stratified_split(df, args.val_ratio, args.seed)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Model
    model = BioCLIPClassifier(
        num_classes=len(classes),
        freeze_backbone=args.freeze_backbone,
        use_logit_scale=True,
    ).to(device)

    # Zero-Shot Initialization
    if args.init_zero_shot:
        print("Initializing classifier head with Zero-Shot text embeddings...")
        try:
             # We assume standard BioCLIP/OpenCLIP model name for tokenizer
             tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
             
             # Create simple prompts or just use class names
             # For better results, we could use "a photo of " + cls, but simple names work for BioCLIP
             text_tokens = tokenizer(classes).to(device)
             
             with torch.no_grad():
                 # Encode text
                 text_features = model.backbone.encode_text(text_tokens)
                 # Normalize
                 text_features /= text_features.norm(dim=-1, keepdim=True)
                 
                 # Assign to classifier weights
                 # model.classifier is nn.Linear(embed_dim, num_classes)
                 # Weight shape is [out_features, in_features] -> [num_classes, embed_dim]
                 # text_features shape is [num_classes, embed_dim]
                 if model.classifier.weight.shape == text_features.shape:
                     model.classifier.weight.data.copy_(text_features.float())
                     model.classifier.bias.data.fill_(0.0)
                     print("Successfully initialized classifier weights.")
                 else:
                     print(f"Error: Shape mismatch. Classifier: {model.classifier.weight.shape}, Text: {text_features.shape}")
        except Exception as e:
            print(f"Error during zero-shot initialization: {e}")

    preprocess_train, preprocess_val = model.get_transforms()

    # Optional warmup: train head only for N epochs
    if args.freeze_warmup_epochs > 0:
        print(f"Freeze warmup enabled: backbone frozen for first {args.freeze_warmup_epochs} epochs.")
        model.set_backbone_trainable(False)

    total_p, trainable_p = count_params(model)
    print(f"Model params: total={total_p:,} | trainable={trainable_p:,}")

    # DataLoaders
    train_dataset = AntDataset(train_df, preprocess_train, class_to_idx, args.base_path)
    val_dataset = AntDataset(val_df, preprocess_val, class_to_idx, args.base_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # Loss
    criterion = make_criterion(train_df, classes, class_to_idx, device, args)

    # Optimizer
    optimizer = build_optimizer(model, args)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    patience_left = args.patience

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Warmup transition
        if args.freeze_warmup_epochs > 0 and (epoch == args.freeze_warmup_epochs):
            if not args.freeze_backbone:
                print("Warmup complete -> Unfreezing backbone.")
                model.set_backbone_trainable(True)
                optimizer = build_optimizer(model, args)
                total_p, trainable_p = count_params(model)
                print(f"Model params after unfreeze: total={total_p:,} | trainable={trainable_p:,}")
            else:
                print("Backbone stays frozen (because --freeze-backbone was set).")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item()) * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({"loss": float(loss.item())})

        epoch_loss = train_loss / max(train_total, 1)
        epoch_acc = 100.0 * train_correct / max(train_total, 1)
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += float(loss.item()) * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save / Early stop
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            if args.patience > 0:
                patience_left = args.patience

            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "bioclip_finetuned.pt")
            print(f"Saving best model to {save_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": class_to_idx,
                    "best_acc": best_acc,
                    "classes": classes,
                },
                save_path,
            )
        else:
            if args.patience > 0:
                patience_left -= 1
                print(f"No improvement. Early-stop patience left: {patience_left}")
                if patience_left <= 0:
                    print("Early stopping triggered.")
                    break

    # Save class list
    os.makedirs(args.output_dir, exist_ok=True)
    save_path_classes = os.path.join(args.output_dir, "classes.txt")
    with open(save_path_classes, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Saved class list to {save_path_classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioCLIP Classifier")
    parser.add_argument("--csv-file", default="labels_synced.csv", help="Path to CSV with labels")
    parser.add_argument("--base-path", default=".", help="Base path for images")
    parser.add_argument("--output-dir", default="models", help="Directory to save models")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Learning rate for BioCLIP backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classifier head/logit_scale")

    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze BioCLIP backbone forever")
    parser.add_argument(
        "--freeze-warmup-epochs",
        type=int,
        default=2,
        help="Train head-only for N epochs, then unfreeze backbone (ignored if --freeze-backbone).",
    )

    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer samples")

    parser.add_argument("--filter-incomplete", action="store_true", help="Remove species with 'sp' or 'Unknown' in name")
    parser.add_argument("--min-samples", type=int, default=0, help="Minimum number of samples per class")
    parser.add_argument("--drop-missing-images", action="store_true", help="Drop rows with missing image files before training")

    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/shuffle")
    parser.add_argument("--init-zero-shot", action="store_true", help="Initialize classifier head with zero-shot text embeddings")

    # Long-tail helpers
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss (helps rare classes).",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Weighting strength: 1/freq^power. 0.5 = 1/sqrt(freq), 1.0 = 1/freq.",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs (0 disables).",
    )

    args = parser.parse_args()
    train(args)
