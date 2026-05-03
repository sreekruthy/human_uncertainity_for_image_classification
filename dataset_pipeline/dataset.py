import os
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 10

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ── Transforms ────────────────────────────────────────────────────────────────
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
    """
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, "cifar10h-probs.npy")

    if not os.path.exists(file_path):
        url = (
            "https://github.com/jcpeterson/cifar-10h/raw/master/"
            "data/cifar10h-probs.npy"
        )
        print("[INFO] Downloading CIFAR-10H...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)
        print("[INFO] Download complete.")
    else:
        print("[INFO] CIFAR-10H already downloaded.")

    return np.load(file_path)

# ── Entropy ───────────────────────────────────────────────────────────────────
def compute_entropy(soft_labels):
    """
    Shannon entropy H(p) = -sum(p * log2(p)) per image.
    Range: 0 (full agreement) → log2(10) ≈ 3.32 (maximum disagreement).
    """
    p = np.clip(soft_labels, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

# ── Sanity Checks ─────────────────────────────────────────────────────────────
def run_sanity_checks(soft_labels, root="./data"):
    """
    Runs five required checks (from project specification):
      1. Shape is (10000, 10)
      2. Every soft label sums to 1
      3. Entropy statistics (mean/std/min/max)
      4. Low (<0.5) and high (>2.0) disagreement counts
      5. Soft argmax vs CIFAR-10 hard label alignment (>85%)
    Returns entropy array for downstream use.
    """
    print("\n" + "=" * 55)
    print("SANITY CHECKS — CIFAR-10H")
    print("=" * 55)

    # 1. Shape
    assert soft_labels.shape == (10000, 10), \
        f"Shape mismatch! Expected (10000,10), got {soft_labels.shape}"
    print(f"[OK] Shape: {soft_labels.shape}")

    # 2. Sums to 1
    sums = soft_labels.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), \
        "Some soft labels do not sum to 1!"
    print(f"[OK] All soft labels sum to 1  "
          f"(min={sums.min():.6f}, max={sums.max():.6f})")

    # 3. Entropy stats
    entropy = compute_entropy(soft_labels)
    print(f"[OK] Entropy stats:")
    print(f"       Mean : {entropy.mean():.4f}")
    print(f"       Std  : {entropy.std():.4f}")
    print(f"       Min  : {entropy.min():.4f}")
    print(f"       Max  : {entropy.max():.4f}")
    print(f"       Max possible (uniform) : {np.log2(10):.4f}")

    # 4. Disagreement counts
    low_dis  = (entropy < 0.5).sum()
    high_dis = (entropy > 2.0).sum()
    print(f"[OK] Low  disagreement (entropy < 0.5) : {low_dis:5d} images")
    print(f"[OK] High disagreement (entropy > 2.0) : {high_dis:5d} images")

    # 5. Alignment with CIFAR-10 hard labels
    base = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
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
    Returns (img, soft_label, hard_label) using CIFAR-10H soft labels.

    Used for the SOFT-LABEL training condition (paper's main result).
      img        : normalised 32×32 tensor
      soft_label : 10-dim human annotator probability distribution
      hard_label : original CIFAR-10 ground-truth integer label
    """
    def __init__(self, cifar10_dataset, soft_labels, transform=None):
        self.data      = cifar10_dataset
        self.soft      = soft_labels          # shape (10000, 10)
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
    Returns (img, one_hot_label, hard_label) using CIFAR-10 hard labels ONLY.

    Used for the HARD-LABEL baseline condition (control in paper).
      img        : normalised 32×32 tensor
      soft_label : one-hot 10-dim vector (no soft information)
      hard_label : original CIFAR-10 ground-truth integer label
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
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[hard_label] = 1.0
        return img, one_hot, int(hard_label)


# ── Public API — K-Fold Loaders (Paper Protocol) ──────────────────────────────
def get_kfold_loaders(
        root="./data",
        k=10,
        batch_size=128,
        use_soft_labels=True,
        seed=42
):
    """
    Returns list of (train_loader, val_loader) tuples for k-fold CV.

    Follows Peterson et al. §5.1 exactly:
      - 10-fold CV on 10,000 CIFAR-10H test images
      - 9,000 train / 1,000 val per fold
      - Fixed random_state=42 for reproducibility
      - Both soft and hard conditions train on the SAME 10k image pool
        (the fair apples-to-apples comparison the paper makes)

    Args:
        root            : data directory
        k               : number of folds (paper uses 10)
        batch_size      : mini-batch size
        use_soft_labels : True → CIFAR10SoftDataset, False → CIFAR10HardDataset
        seed            : KFold random state

    Returns:
        List of k (train_loader, val_loader) tuples.
    """
    soft_labels = download_cifar10h(root)

    # Both conditions use CIFAR-10 *test* set (10k images) —
    # the same pool that CIFAR-10H annotations cover.
    base_train = datasets.CIFAR10(
        root=root, train=False, download=True, transform=None
    )
    base_val = datasets.CIFAR10(
        root=root, train=False, download=True, transform=None
    )

    kf      = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = np.arange(10000)
    loaders = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_idx = train_idx.tolist()
        val_idx   = val_idx.tolist()

        if use_soft_labels:
            train_ds = Subset(
                CIFAR10SoftDataset(base_train, soft_labels,
                                   transform=train_transform),
                train_idx
            )
            val_ds = Subset(
                CIFAR10SoftDataset(base_val, soft_labels,
                                   transform=test_transform),
                val_idx
            )
        else:
            # Hard-label baseline: one-hot from CIFAR-10 targets, no soft file
            train_ds = Subset(
                CIFAR10HardDataset(base_train, transform=train_transform),
                train_idx
            )
            val_ds = Subset(
                CIFAR10HardDataset(base_val, transform=test_transform),
                val_idx
            )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,  num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,   batch_size=batch_size, shuffle=False, num_workers=2,
            pin_memory=True
        )

        loaders.append((train_loader, val_loader))
        label_type = "soft" if use_soft_labels else "hard"
        print(
            f"[INFO] Fold {fold_idx + 1:02d}/{k} | "
            f"{label_type} labels | "
            f"Train={len(train_ds):5d} | Val={len(val_ds):4d}"
        )

    return loaders


def get_single_fold_loaders(fold_idx, root="./data", batch_size=128,
                            use_soft_labels=True, seed=42):
    """
    Convenience wrapper: returns (train_loader, val_loader) for one fold.
    Called by training loop as:
        train_loader, val_loader = get_single_fold_loaders(fold_idx)
    """
    loaders = get_kfold_loaders(root, k=10, batch_size=batch_size,
                                 use_soft_labels=use_soft_labels, seed=seed)
    return loaders[fold_idx]


# ── Public API — Full Datasets (for Member 4 evaluation) ─────────────────────
def get_datasets(root="./data", run_checks=True):
    """
    Returns (soft_dataset, hard_dataset) on all 10,000 CIFAR-10H images.
    Useful for Member 4's evaluation and entropy-based analysis.
    """
    soft_labels = download_cifar10h(root)
    if run_checks:
        run_sanity_checks(soft_labels, root)

    base = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    soft_ds = CIFAR10SoftDataset(base, soft_labels)
    hard_ds = CIFAR10HardDataset(base)
    return soft_ds, hard_ds


# ── Public API — Standard CIFAR-10 Test Loader (for Member 4) ────────────────
def get_test_loader(root="./data", batch_size=128):
    """Standard CIFAR-10 test set with hard labels (for accuracy evaluation)."""
    test_ds = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )


# ── Public API — Entropy Stats (for Member 4 plots) ──────────────────────────
def get_entropy_stats(root="./data"):
    """Returns entropy array for all 10k images. Used for Figure plots."""
    soft_labels = download_cifar10h(root)
    return compute_entropy(soft_labels)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Sanity checks
    soft_labels = download_cifar10h()
    entropy     = run_sanity_checks(soft_labels)

    # 2. Smoke-test: build 2-fold loaders (quick, not the full 10)
    print("--- Soft label 2-fold smoke test ---")
    soft_loaders = get_kfold_loaders(k=2, batch_size=64, use_soft_labels=True)
    imgs, soft, hard = next(iter(soft_loaders[0][0]))
    print(f"  images : {tuple(imgs.shape)}")
    print(f"  soft   : {tuple(soft.shape)}  sum={soft[0].sum():.4f}")
    print(f"  hard   : {hard[:4].tolist()}")

    print("--- Hard label 2-fold smoke test ---")
    hard_loaders = get_kfold_loaders(k=2, batch_size=64, use_soft_labels=False)
    imgs, one_hot, hard = next(iter(hard_loaders[0][0]))
    print(f"  images : {tuple(imgs.shape)}")
    print(f"  one_hot: {tuple(one_hot.shape)}  sum={one_hot[0].sum():.1f}")
    print(f"  hard   : {hard[:4].tolist()}")

    print("\nAll checks passed.")