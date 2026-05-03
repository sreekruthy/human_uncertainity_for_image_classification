"""
Goal: build a model that predicts the FULL soft-label distribution p(y|x),
not just the modal class. This file provides the data pipeline for that task.

Data split (project spec §4.3):
    CIFAR-10H provides 10,000 images with soft labels.
    Split: 6,000 train / 2,000 val / 2,000 test  (fixed seed=42)

Training images:
    The 50,000 CIFAR-10 training images carry only hard labels. They may be
    used for backbone pretraining (Member 2/3), but MUST NOT be given soft
    targets — there are none. This file provides a separate hard-label loader
    for that pretraining phase.

What every loader returns: (img_tensor, soft_label, hard_label)
    img_tensor  : (3, 32, 32) normalised float32
    soft_label  : (10,) float32 — human annotator probability distribution
    hard_label  : int — CIFAR-10 ground-truth (majority vote) class index

Required project deliverables produced here (§4 "Initial sanity checks" +
"Required visualisations at the data stage"):
    1. Entropy histogram across the full dataset
    2. Per-class average entropy bar plot
    3. Annotator confusion-style matrix
    4. Example grid: low-entropy vs high-entropy images + distribution bars
    All saved to ./figures/ by default.

CHANGES vs original Peterson-replication dataset.py:
    - Replaced 10-fold CV loaders with the project-required 6k/2k/2k split
      (kept get_kfold_loaders() as an OPTIONAL helper for bonus experiments)
    - Added all four required EDA visualisations (generate_eda_figures)
    - CIFAR10HardDataset still uses pure one-hot (no leakage) — unchanged
    - download_cifar10h() still returns ndarray directly — unchanged
    - run_sanity_checks() still validates shape, sums, entropy, alignment
    - Added get_cifar10_pretrain_loader() for backbone pretraining phase
"""

import os
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import KFold

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
SEED        = 42   # fixed seed — document clearly in report (project §4.3)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ── Transforms ────────────────────────────────────────────────────────────────
# Allowed augmentations (project §5.4): random crop + horizontal flip only.
# DO NOT use augmentations that change class semantics.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Download CIFAR-10H ────────────────────────────────────────────────────────
def download_cifar10h(root="./data"):
    """
    Downloads cifar10h-probs.npy if not present and returns it as an ndarray.
    Shape: (10000, 10) — one probability distribution per CIFAR-10 test image.
    Each row sums to 1. Approximately 50 human annotators per image.
    """
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, "cifar10h-probs.npy")

    if not os.path.exists(file_path):
        url = (
            "https://github.com/jcpeterson/cifar-10h/raw/master/"
            "data/cifar10h-probs.npy"
        )
        print("[INFO] Downloading CIFAR-10H soft labels...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)
        print("[INFO] Download complete.")
    else:
        print("[INFO] CIFAR-10H already downloaded.")

    return np.load(file_path)   # shape (10000, 10)

# ── Entropy Utilities ─────────────────────────────────────────────────────────
def compute_entropy(soft_labels):
    """
    Shannon entropy H(p) = -sum_c p(c) * log2(p(c)) per image.

    Range:
        0           → perfect annotator agreement (one class gets all votes)
        log2(10) ≈ 3.32 → maximum disagreement (uniform over 10 classes)

    Args:
        soft_labels : ndarray (N, 10)
    Returns:
        entropy     : ndarray (N,)
    """
    p = np.clip(soft_labels, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

# ── Sanity Checks ─────────────────────────────────────────────────────────────
def run_sanity_checks(soft_labels, root="./data"):
    """
    Five required checks (project §4 "Initial sanity checks"):
      1. Shape is (10000, 10)
      2. Every soft-label vector sums to 1
      3. Entropy statistics (mean / std / min / max)
      4. Low (<0.5) and high (>2.0) disagreement image counts
      5. Soft-argmax vs CIFAR-10 hard-label alignment (expect >85%)

    Returns entropy array (N,) for downstream use.
    """
    print("\n" + "=" * 55)
    print("SANITY CHECKS — CIFAR-10H")
    print("=" * 55)

    # 1. Shape
    assert soft_labels.shape == (10000, 10), \
        f"Shape mismatch: expected (10000,10), got {soft_labels.shape}"
    print(f"[OK] Shape: {soft_labels.shape}")

    # 2. Sums to 1
    sums = soft_labels.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), \
        "Some soft labels do not sum to 1!"
    print(f"[OK] All rows sum to 1  "
          f"(min={sums.min():.6f}, max={sums.max():.6f})")

    # 3. Entropy statistics
    entropy = compute_entropy(soft_labels)
    print(f"[OK] Entropy stats:")
    print(f"       Mean  : {entropy.mean():.4f}")
    print(f"       Std   : {entropy.std():.4f}")
    print(f"       Min   : {entropy.min():.4f}")
    print(f"       Max   : {entropy.max():.4f}")
    print(f"       Max possible (uniform over 10) : {np.log2(10):.4f}")

    # 4. Disagreement counts
    low_dis  = (entropy < 0.5).sum()
    high_dis = (entropy > 2.0).sum()
    print(f"[OK] Low  disagreement (H < 0.5) : {low_dis:5d} images")
    print(f"[OK] High disagreement (H > 2.0) : {high_dis:5d} images")

    # 5. Alignment with CIFAR-10 majority-vote labels
    base       = datasets.CIFAR10(root=root, train=False,
                                   download=True, transform=test_transform)
    hard_labels = np.array(base.targets)
    soft_argmax = np.argmax(soft_labels, axis=1)
    agreement   = (soft_argmax == hard_labels).mean()
    print(f"[OK] Soft-argmax / hard-label agreement : {agreement * 100:.1f}%")
    assert agreement > 0.85, \
        f"Alignment too low ({agreement:.2f}) — data may be misaligned!"

    print("=" * 55 + "\n")
    return entropy

# ── Dataset Classes ───────────────────────────────────────────────────────────

class CIFAR10SoftDataset(Dataset):
    """
    Wraps CIFAR-10 test images with CIFAR-10H soft labels.

    Returns: (img_tensor, soft_label, hard_label)
        img_tensor  : (3, 32, 32) normalised float32
        soft_label  : (10,) float32 — human annotator distribution, sums to 1
        hard_label  : int — ground-truth CIFAR-10 class index

    This is the PRIMARY dataset for training the disagreement-prediction model.
    The training target is soft_label, NOT hard_label.
    hard_label is kept for top-1 accuracy bookkeeping only.
    """
    def __init__(self, cifar10_dataset, soft_labels, transform=None):
        self.data      = cifar10_dataset   # torchvision CIFAR10 object
        self.soft      = soft_labels       # ndarray (10000, 10)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, hard_label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        soft_label = torch.tensor(self.soft[idx], dtype=torch.float32)
        return img, soft_label, int(hard_label)


class CIFAR10HardDataset(Dataset):
    """
    Wraps CIFAR-10 test images with pure one-hot hard labels.

    Used only for the HARD-LABEL baseline condition.
    Deliberately does NOT read from cifar10h-probs.npy to avoid leaking
    soft information into the hard-label control.

    Returns: (img_tensor, one_hot_label, hard_label)
        img_tensor  : (3, 32, 32) normalised float32
        one_hot_label: (10,) float32 — one-hot, sums to 1
        hard_label  : int — CIFAR-10 ground-truth class index
    """
    def __init__(self, cifar10_dataset, transform=None):
        self.data      = cifar10_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, hard_label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        one_hot        = torch.zeros(NUM_CLASSES)
        one_hot[hard_label] = 1.0
        return img, one_hot, int(hard_label)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY PUBLIC API — Project §4.3 required 6k / 2k / 2k split
# ═══════════════════════════════════════════════════════════════════════════════

def get_split_loaders(
        root            = "./data",
        batch_size      = 128,
        use_soft_labels = True,
        seed            = SEED,
):
    """
    Returns (train_loader, val_loader, test_loader) for the project split.

    Split sizes (project spec §4.3 — Recommended split for CIFAR-10H):
        Train : 6,000 images
        Val   : 2,000 images
        Test  : 2,000 images

    Both soft and hard conditions split the SAME 10,000 CIFAR-10H images so
    the comparison remains fair.

    IMPORTANT: use_soft_labels=True  → target is human distribution (training)
               use_soft_labels=False → target is one-hot hard label (baseline)
    In both cases hard_label (integer) is always the third return value.

    Args:
        root            : data directory
        batch_size      : mini-batch size
        use_soft_labels : True for main model, False for hard-label baseline
        seed            : random seed for the split (must be fixed & reported)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    soft_labels = download_cifar10h(root)

    # Both conditions use the CIFAR-10 *test* set (the 10k that CIFAR-10H covers)
    cifar_base_train = datasets.CIFAR10(root=root, train=False,
                                         download=True, transform=None)
    cifar_base_eval  = datasets.CIFAR10(root=root, train=False,
                                         download=True, transform=None)

    if use_soft_labels:
        full_train_ds = CIFAR10SoftDataset(cifar_base_train, soft_labels,
                                            transform=train_transform)
        full_eval_ds  = CIFAR10SoftDataset(cifar_base_eval,  soft_labels,
                                            transform=test_transform)
    else:
        full_train_ds = CIFAR10HardDataset(cifar_base_train,
                                            transform=train_transform)
        full_eval_ds  = CIFAR10HardDataset(cifar_base_eval,
                                            transform=test_transform)

    # Deterministic split: 6k train / 2k val / 2k test
    rng = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_train_ds,
        [6000, 2000, 2000],
        generator=rng
    )

    # Eval split uses test_transform — override the subset indices on the
    # eval dataset (same indices as the training-transform dataset)
    val_indices  = val_ds.indices
    test_indices = test_ds.indices
    val_ds  = Subset(full_eval_ds, val_indices)
    test_ds = Subset(full_eval_ds, test_indices)

    label_type = "soft" if use_soft_labels else "hard"
    print(f"[INFO] Split ({label_type}) | "
          f"Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# BACKBONE PRETRAINING LOADER — CIFAR-10 50k hard labels (project §4.3 note)
# ═══════════════════════════════════════════════════════════════════════════════

def get_cifar10_pretrain_loader(root="./data", batch_size=128):
    """
    Returns (train_loader, val_loader) on the CIFAR-10 *training* set (50k).

    IMPORTANT: these loaders return (img, hard_label_int) — NOT (img, soft, hard).
    They are used ONLY for backbone pretraining on hard labels (project §5.2).
    Never pass these directly to the soft-label fine-tuning loop.

    The CIFAR-10 test set (10k) is reserved entirely for CIFAR-10H experiments.
    """
    train_ds = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=train_transform)
    val_ds   = datasets.CIFAR10(root=root, train=False,
                                 download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
    print(f"[INFO] CIFAR-10 pretrain split | Train={len(train_ds)} | Val={len(val_ds)}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# FULL DATASET (for Member 4 evaluation — all 10k images)
# ═══════════════════════════════════════════════════════════════════════════════

def get_full_soft_dataset(root="./data"):
    """
    Returns a CIFAR10SoftDataset covering all 10,000 CIFAR-10H images with
    test_transform (no augmentation). Used for Member 4's metric evaluation.

    Returns: soft_dataset (CIFAR10SoftDataset), soft_labels ndarray (10000,10)
    """
    soft_labels = download_cifar10h(root)
    base = datasets.CIFAR10(root=root, train=False,
                              download=True, transform=test_transform)
    return CIFAR10SoftDataset(base, soft_labels), soft_labels


def get_entropy_stats(root="./data"):
    """Returns entropy array (10000,) for all CIFAR-10H images. For plots."""
    soft_labels = download_cifar10h(root)
    return compute_entropy(soft_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL — K-Fold loaders (paper replication / bonus ablations only)
# ═══════════════════════════════════════════════════════════════════════════════

def get_kfold_loaders(
        root="./data", k=10, batch_size=128,
        use_soft_labels=True, seed=SEED
):
    """
    Optional: 10-fold CV loaders for replicating Peterson et al. §5.1.
    NOT required for the primary project deliverables.
    Use get_split_loaders() for all graded experiments.
    """
    soft_labels = download_cifar10h(root)
    base_train  = datasets.CIFAR10(root=root, train=False,
                                    download=True, transform=None)
    base_val    = datasets.CIFAR10(root=root, train=False,
                                    download=True, transform=None)
    kf      = KFold(n_splits=k, shuffle=True, random_state=seed)
    loaders = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(10000))):
        if use_soft_labels:
            tr_ds = Subset(CIFAR10SoftDataset(base_train, soft_labels,
                                               transform=train_transform),
                           train_idx.tolist())
            vl_ds = Subset(CIFAR10SoftDataset(base_val, soft_labels,
                                               transform=test_transform),
                           val_idx.tolist())
        else:
            tr_ds = Subset(CIFAR10HardDataset(base_train, transform=train_transform),
                           train_idx.tolist())
            vl_ds = Subset(CIFAR10HardDataset(base_val, transform=test_transform),
                           val_idx.tolist())
        loaders.append((
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True),
        ))
        label_type = "soft" if use_soft_labels else "hard"
        print(f"[INFO] Fold {fold_idx+1:02d}/{k} | {label_type} | "
              f"Train={len(tr_ds)} | Val={len(vl_ds)}")
    return loaders


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED EDA VISUALISATIONS (project §4 "Required visualisations at the data
# stage") — produces all four mandatory figures.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_eda_figures(root="./data", fig_dir="./figures", n_examples=5):
    """
    Generates all four required data-stage visualisations.

    Outputs (saved to fig_dir/):
        1. entropy_histogram.png       — histogram of true entropy across dataset
        2. per_class_entropy.png       — per-class average entropy bar chart
        3. annotator_confusion.png     — confusion-style matrix from soft labels
        4. example_grid.png            — low-entropy vs high-entropy example images

    Args:
        root       : data directory
        fig_dir    : output directory for figures
        n_examples : number of example images per entropy group (default 5)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not found — skipping figure generation. "
              "Install with: pip install matplotlib")
        return

    os.makedirs(fig_dir, exist_ok=True)

    soft_labels = download_cifar10h(root)
    entropy     = compute_entropy(soft_labels)
    hard_labels = np.array(
        datasets.CIFAR10(root=root, train=False,
                          download=True, transform=test_transform).targets
    )

    # ── Figure 1: Entropy Histogram ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropy, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(np.log2(10), color='crimson', linestyle='--',
               label=f'Max entropy = {np.log2(10):.2f}')
    ax.axvline(entropy.mean(), color='orange', linestyle='--',
               label=f'Mean = {entropy.mean():.2f}')
    ax.set_xlabel('Shannon Entropy H(p)  [bits]')
    ax.set_ylabel('Number of images')
    ax.set_title('Distribution of annotator entropy across CIFAR-10H')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'entropy_histogram.png'), dpi=150)
    plt.close(fig)
    print(f"[FIG] Saved entropy_histogram.png")

    # ── Figure 2: Per-class Average Entropy ───────────────────────────────────
    per_class_entropy = [
        entropy[hard_labels == c].mean() for c in range(NUM_CLASSES)
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(CIFAR10_CLASSES, per_class_entropy,
                  color='steelblue', edgecolor='white')
    ax.set_ylabel('Mean Shannon Entropy  [bits]')
    ax.set_title('Per-class average annotator entropy')
    ax.set_ylim(0, np.log2(10) * 1.05)
    ax.axhline(np.log2(10), color='crimson', linestyle='--', alpha=0.5,
               label='Max entropy')
    # Annotate bar values
    for bar, val in zip(bars, per_class_entropy):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'per_class_entropy.png'), dpi=150)
    plt.close(fig)
    print(f"[FIG] Saved per_class_entropy.png")

    # ── Figure 3: Annotator Confusion Matrix ──────────────────────────────────
    # confusion[i, j] = mean probability annotators assigned class j to images
    # whose hard label is class i.
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        mask = hard_labels == c
        if mask.sum() > 0:
            confusion[c] = soft_labels[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
    ax.set_xlabel('Assigned class (annotator vote)')
    ax.set_ylabel('True class (hard label)')
    ax.set_title('Annotator confusion matrix\n'
                 '(mean soft-label probability per true class)')
    # Annotate cells
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = confusion[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'annotator_confusion.png'), dpi=150)
    plt.close(fig)
    print(f"[FIG] Saved annotator_confusion.png")

    # ── Figure 4: Example Grid — Low vs High Entropy ──────────────────────────
    sorted_idx  = np.argsort(entropy)
    low_idx     = sorted_idx[:n_examples]          # lowest entropy
    high_idx    = sorted_idx[-n_examples:][::-1]   # highest entropy

    # Load raw (un-normalised) images for display
    raw_base = datasets.CIFAR10(root=root, train=False,
                                 download=True, transform=transforms.ToTensor())

    n_cols = n_examples
    n_rows = 4  # top-row images + dist bars, bottom-row images + dist bars
    fig   = plt.figure(figsize=(n_cols * 2.5, 9))
    spec  = gridspec.GridSpec(4, n_cols, figure=fig,
                               hspace=0.5, wspace=0.3)

    def plot_example(ax_img, ax_bar, img_idx, label):
        img, _ = raw_base[img_idx]
        img_np  = img.permute(1, 2, 0).numpy()
        ax_img.imshow(img_np)
        ax_img.set_title(
            f"H={entropy[img_idx]:.2f}\n"
            f"({CIFAR10_CLASSES[hard_labels[img_idx]]})",
            fontsize=8
        )
        ax_img.axis('off')
        probs = soft_labels[img_idx]
        ax_bar.bar(range(NUM_CLASSES), probs,
                   color=['steelblue' if i != hard_labels[img_idx]
                          else 'coral' for i in range(NUM_CLASSES)])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(range(NUM_CLASSES))
        ax_bar.set_xticklabels(
            [c[0].upper() for c in CIFAR10_CLASSES],
            fontsize=6
        )
        ax_bar.tick_params(axis='y', labelsize=6)

    for col, img_idx in enumerate(low_idx):
        plot_example(
            fig.add_subplot(spec[0, col]),
            fig.add_subplot(spec[1, col]),
            img_idx, 'low'
        )
    for col, img_idx in enumerate(high_idx):
        plot_example(
            fig.add_subplot(spec[2, col]),
            fig.add_subplot(spec[3, col]),
            img_idx, 'high'
        )

    fig.suptitle(
        f"Top {n_examples} lowest-entropy images (rows 1–2) vs\n"
        f"top {n_examples} highest-entropy images (rows 3–4)\n"
        "Orange bar = hard-label class",
        fontsize=10, y=1.01
    )
    fig.savefig(os.path.join(fig_dir, 'example_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIG] Saved example_grid.png")

    print(f"\n[INFO] All 4 required EDA figures saved to {fig_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("DATASET SELF-TEST")
    print("=" * 60)

    # 1. Download + sanity checks
    soft_labels = download_cifar10h()
    entropy     = run_sanity_checks(soft_labels)

    # 2. Primary split (project spec)
    print("--- Project split: 6k / 2k / 2k (soft labels) ---")
    train_loader, val_loader, test_loader = get_split_loaders(
        batch_size=64, use_soft_labels=True
    )
    imgs, soft, hard = next(iter(train_loader))
    print(f"  Train batch  — images: {tuple(imgs.shape)}, "
          f"soft: {tuple(soft.shape)}, hard: {hard[:4].tolist()}")
    print(f"  soft[0] sum  = {soft[0].sum():.4f}  (must be 1.0)")
    assert abs(soft[0].sum().item() - 1.0) < 1e-4, "Soft label does not sum to 1!"

    # 3. Hard-label baseline split
    print("--- Project split: 6k / 2k / 2k (hard labels) ---")
    train_h, val_h, test_h = get_split_loaders(
        batch_size=64, use_soft_labels=False
    )
    imgs, one_hot, hard = next(iter(train_h))
    print(f"  Hard batch   — one_hot: {tuple(one_hot.shape)}, "
          f"sum={one_hot[0].sum():.1f}")
    assert abs(one_hot[0].sum().item() - 1.0) < 1e-4, "One-hot does not sum to 1!"

    # 4. Pretrain loader
    print("--- CIFAR-10 pretrain loader (50k) ---")
    pt_train, pt_val = get_cifar10_pretrain_loader(batch_size=64)
    imgs, labels = next(iter(pt_train))
    print(f"  Pretrain batch — images: {tuple(imgs.shape)}, "
          f"labels dtype: {labels.dtype}, labels[:4]: {labels[:4].tolist()}")

    # 5. EDA figures
    print("\n--- Generating required EDA figures ---")
    generate_eda_figures(fig_dir="./figures")

    print("\n[PASS] All self-tests passed.")