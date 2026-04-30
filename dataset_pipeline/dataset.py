import os
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold

# ── Normalization constants ───────────────────────────────────────────────────
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
def download_cifar10h(save_path="cifar10h-probs.npy"):
    if os.path.exists(save_path):
        print("CIFAR-10H already downloaded.")
        return save_path
    url = "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy"
    print("Downloading CIFAR-10H...")
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    print("Done.")
    return save_path

# ── Entropy Computation ───────────────────────────────────────────────────────
def compute_entropy(soft_labels):
    """
    Shannon entropy H(p) = -sum(p * log2(p)) for each image.
    Range: 0 (perfect agreement) to log2(10)=3.32 (maximum disagreement).
    """
    p = np.clip(soft_labels, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

# ── Sanity Checks ─────────────────────────────────────────────────────────────
def run_sanity_checks(soft_labels):
    """
    Required sanity checks from project specification:
    1. Verify shape is (10000, 10)
    2. Verify every soft label sums to 1
    3. Compute and report entropy statistics
    4. Count low and high disagreement images
    5. Verify alignment with CIFAR-10 hard labels
    """
    print("\n" + "="*50)
    print("SANITY CHECKS")
    print("="*50)

    # Check 1: Shape
    assert soft_labels.shape == (10000, 10), \
        f"Shape mismatch! Expected (10000,10), got {soft_labels.shape}"
    print(f"[OK] Shape: {soft_labels.shape}")

    # Check 2: Sums to 1
    sums = soft_labels.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), \
        "Some soft labels do not sum to 1!"
    print(f"[OK] All soft labels sum to 1 "
          f"(min={sums.min():.6f}, max={sums.max():.6f})")

    # Check 3: Entropy statistics
    entropy = compute_entropy(soft_labels)
    print(f"[OK] Entropy stats:")
    print(f"     Mean  : {entropy.mean():.4f}")
    print(f"     Std   : {entropy.std():.4f}")
    print(f"     Min   : {entropy.min():.4f}")
    print(f"     Max   : {entropy.max():.4f}")
    print(f"     Max possible (uniform): {np.log2(10):.4f}")

    # Check 4: Low vs high disagreement counts
    low  = (entropy < 0.5).sum()
    high = (entropy > 2.0).sum()
    print(f"[OK] Low disagreement  (entropy < 0.5): {low} images")
    print(f"[OK] High disagreement (entropy > 2.0): {high} images")

    # Check 5: Alignment — argmax of soft labels should match
    # CIFAR-10 hard labels for majority of images
    # This verifies CIFAR-10H is correctly aligned with CIFAR-10 images
    base = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )
    hard_labels = np.array(base.targets)
    soft_argmax = np.argmax(soft_labels, axis=1)
    agreement   = (soft_argmax == hard_labels).mean()
    print(f"[OK] Soft/Hard label alignment: "
          f"{agreement*100:.1f}% agreement")
    assert agreement > 0.85, \
        f"Alignment too low: {agreement:.2f} — data may be misaligned!"

    print("="*50 + "\n")
    return entropy

# ── Dataset Classes ───────────────────────────────────────────────────────────
class CIFAR10SoftDataset(Dataset):
    """
    PyTorch Dataset that returns (img, soft_label, hard_label).
    - img        : normalised 32x32 tensor
    - soft_label : 10-dim human annotator distribution from CIFAR-10H
    - hard_label : argmax of soft distribution (majority vote label)
    """
    def _init_(self, cifar10_dataset, soft_labels):
        self.data = cifar10_dataset
        self.soft = soft_labels

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        img, _     = self.data[idx]
        soft_label = torch.tensor(self.soft[idx], dtype=torch.float32)
        hard_label = int(np.argmax(self.soft[idx]))
        return img, soft_label, hard_label


class CIFAR10HardDataset(Dataset):
    """
    PyTorch Dataset that returns (img, soft_label, hard_label).
    - img        : normalised 32x32 tensor
    - soft_label : 10-dim human annotator distribution from CIFAR-10H
    - hard_label : original CIFAR-10 ground truth label (one-hot baseline)
    """
    def _init_(self, cifar10_dataset, soft_labels):
        self.data = cifar10_dataset
        self.soft = soft_labels

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        img, hard_label = self.data[idx]
        soft_label      = torch.tensor(self.soft[idx], dtype=torch.float32)
        return img, soft_label, hard_label


# ── Fixed Split (Project Specification) ──────────────────────────────────────
def get_split_indices(n=10000, train=6000, val=2000, test=2000, seed=42):
    """
    Returns fixed train/val/test indices.
    Split: 6000 train / 2000 val / 2000 test
    Fixed random seed=42 for full reproducibility.
    """
    assert train + val + test == n, "Split sizes must sum to 10000"
    rng       = np.random.default_rng(seed)
    indices   = rng.permutation(n)
    train_idx = indices[:train]
    val_idx   = indices[train:train+val]
    test_idx  = indices[train+val:]
    return train_idx, val_idx, test_idx


# ── Public API — Split Loaders ────────────────────────────────────────────────
def get_split_loaders(batch_size=128):
    """
    Returns (train_loader, val_loader, test_loader) using the
    6000/2000/2000 split specified in the project document.
    Fixed seed=42 for reproducibility.
    """
    soft_labels = np.load(download_cifar10h())

    base_train = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=train_transform
    )
    base_test = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )

    train_idx, val_idx, test_idx = get_split_indices()

    train_ds = Subset(CIFAR10SoftDataset(base_train, soft_labels), train_idx)
    val_ds   = Subset(CIFAR10SoftDataset(base_test,  soft_labels), val_idx)
    test_ds  = Subset(CIFAR10SoftDataset(base_test,  soft_labels), test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ── Public API — K-Fold Loaders (Paper Specification) ────────────────────────
def get_kfold_loaders(fold_idx, batch_size=128):
    """
    Returns (train_loader, val_loader) for the given fold (0-9).
    10-fold CV: 9000 train / 1000 val per fold.
    Exactly as described in the paper (Peterson et al., ICCV 2019).
    Fixed random_state=42 for reproducibility.
    Used by Member 3 training pipeline.
    """
    soft_labels = np.load(download_cifar10h())

    base_train = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=train_transform
    )
    base_val = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )

    soft_ds_train = CIFAR10SoftDataset(base_train, soft_labels)
    soft_ds_val   = CIFAR10SoftDataset(base_val,   soft_labels)

    kf     = KFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(kf.split(range(10000)))
    train_idx, val_idx = splits[fold_idx]

    train_loader = DataLoader(
        Subset(soft_ds_train, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Subset(soft_ds_val, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


# ── Public API — Full Dataset ─────────────────────────────────────────────────
def get_datasets(run_checks=True):
    """
    Returns (soft_dataset, hard_dataset) on the full 10k CIFAR-10H images.
    Runs sanity checks if run_checks=True.
    """
    soft_labels = np.load(download_cifar10h())

    if run_checks:
        run_sanity_checks(soft_labels)

    base_test = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )

    soft_ds = CIFAR10SoftDataset(base_test, soft_labels)
    hard_ds = CIFAR10HardDataset(base_test, soft_labels)
    return soft_ds, hard_ds


# ── Public API — Test Loader ──────────────────────────────────────────────────
def get_test_loader(batch_size=128):
    """Standard CIFAR-10 test set loader (hard labels only)."""
    test_ds = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )
    return DataLoader(test_ds, batch_size=batch_size,
                      shuffle=False, num_workers=0)


# ── Public API — Entropy Stats ────────────────────────────────────────────────
def get_entropy_stats():
    """
    Returns entropy array for all 10k images.
    Useful for plots and analysis in Member 4.
    """
    soft_labels = np.load(download_cifar10h())
    return compute_entropy(soft_labels)