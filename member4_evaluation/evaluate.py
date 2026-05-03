# =========================================================
# member4_evaluation/evaluate.py
# =========================================================

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# PROJECT IMPORTS
# =========================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from member1_dataset.dataset import get_split_loaders
from member2_model.models import get_model


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULT_DIR = "./evaluation_results"

PLOT_DIR = os.path.join(
    RESULT_DIR,
    "plots"
)

os.makedirs(RESULT_DIR, exist_ok=True)

os.makedirs(PLOT_DIR, exist_ok=True)

PRECISION_K = 3

CIFAR10_CLASSES = [

    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",

    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("=" * 60)

print("EVALUATION PIPELINE")

print("=" * 60)


# =========================================================
# HELPERS
# =========================================================

def jsd_loss(p, q):

    m = 0.5 * (p + q)

    jsd = 0.5 * (

        F.kl_div(
            (p + 1e-12).log(),
            m,
            reduction='batchmean'
        )

        +

        F.kl_div(
            (q + 1e-12).log(),
            m,
            reduction='batchmean'
        )
    )

    return jsd


def precision_at_k(
    probs,
    soft_targets,
    k=3
):

    top_model = probs.topk(k, dim=1).indices

    top_human = soft_targets.topk(k, dim=1).indices

    matches = 0.0

    for i in range(probs.size(0)):

        model_set = set(
            top_model[i].tolist()
        )

        human_set = set(
            top_human[i].tolist()
        )

        matches += (
            len(model_set & human_set) / k
        )

    return matches / probs.size(0)


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader
):

    model.eval()

    total = 0
    correct = 0

    ce_loss_total = 0.0
    kl_total = 0.0
    jsd_total = 0.0
    cos_total = 0.0
    p_at_k_total = 0.0

    n_batches = 0

    pred_entropies = []
    target_entropies = []

    all_preds = []
    all_labels = []

    ce_fn = torch.nn.CrossEntropyLoss()

    for images, soft_targets, hard_labels in loader:

        images = images.to(DEVICE)

        soft_targets = soft_targets.to(DEVICE)

        hard_labels = hard_labels.to(DEVICE)

        logits = model(images)

        log_probs = F.log_softmax(
            logits,
            dim=1
        )

        probs = F.softmax(
            logits,
            dim=1
        )

        preds = probs.argmax(dim=1)

        # -------------------------------------
        # ACCURACY
        # -------------------------------------

        correct += (
            preds == hard_labels
        ).sum().item()

        total += images.size(0)

        # -------------------------------------
        # CROSS ENTROPY
        # -------------------------------------

        ce = ce_fn(
            logits,
            hard_labels
        )

        ce_loss_total += ce.item()

        # -------------------------------------
        # KL DIVERGENCE
        # -------------------------------------

        kl = F.kl_div(
            log_probs,
            soft_targets,
            reduction='batchmean'
        )

        kl_total += kl.item()

        # -------------------------------------
        # JSD
        # -------------------------------------

        jsd = jsd_loss(
            probs,
            soft_targets
        )

        jsd_total += jsd.item()

        # -------------------------------------
        # COSINE SIMILARITY
        # -------------------------------------

        cos_sim = F.cosine_similarity(
            probs,
            soft_targets,
            dim=1
        ).mean()

        cos_total += cos_sim.item()

        # -------------------------------------
        # PRECISION@K
        # -------------------------------------

        p_at_k = precision_at_k(
            probs,
            soft_targets,
            k=PRECISION_K
        )

        p_at_k_total += p_at_k

        # -------------------------------------
        # ENTROPY
        # -------------------------------------

        pred_entropy = -(
            probs *
            log_probs
        ).sum(dim=1)

        safe_log_targets = (
            soft_targets
            .clamp(min=1e-12)
            .log()
        )

        target_entropy = -(
            soft_targets *
            safe_log_targets
        ).sum(dim=1)

        pred_entropies.extend(
            pred_entropy.cpu().numpy()
        )

        target_entropies.extend(
            target_entropy.cpu().numpy()
        )

        # -------------------------------------
        # STORE PREDICTIONS
        # -------------------------------------

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            hard_labels.cpu().numpy()
        )

        n_batches += 1

    # =====================================================
    # CORRELATIONS
    # =====================================================

    pearson_r, _ = pearsonr(
        pred_entropies,
        target_entropies
    )

    spearman_r, _ = spearmanr(
        pred_entropies,
        target_entropies
    )

    return {

        "accuracy":
            correct / total,

        "cross_entropy":
            ce_loss_total / n_batches,

        "kl_divergence":
            kl_total / n_batches,

        "jsd":
            jsd_total / n_batches,

        "cosine_similarity":
            cos_total / n_batches,

        "pearson_r":
            float(pearson_r),

        "spearman_r":
            float(spearman_r),

        f"precision_at_{PRECISION_K}":
            p_at_k_total / n_batches,

        "predictions":
            all_preds,

        "labels":
            all_labels
    }


# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    labels,
    preds,
    name
):

    cm = confusion_matrix(
        labels,
        preds
    )

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=CIFAR10_CLASSES
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(name)

    save_path = os.path.join(
        PLOT_DIR,
        f"{name}_confusion_matrix.png"
    )

    plt.savefig(save_path)

    plt.close()


# =========================================================
# MODEL NAME INFERENCE
# =========================================================

def infer_model_name(
    ckpt_name
):

    if "lightcnn" in ckpt_name:
        return "lightcnn"

    elif "resnet20" in ckpt_name:
        return "resnet20"

    elif "resnet56" in ckpt_name:
        return "resnet56"

    elif "110" in ckpt_name:
        return "resnet_basic_110"

    elif "vgg16" in ckpt_name:
        return "vgg16_bn"

    else:
        return None


# =========================================================
# MAIN
# =========================================================

def main():

    train_loader, val_loader, test_loader = (
        get_split_loaders(batch_size=64)
    )

    checkpoint_paths = glob.glob(
        "results/checkpoints/*.pt"
    )

    if len(checkpoint_paths) == 0:

        print(
            "[ERROR] No checkpoints found."
        )

        return

    final_results = []

    for ckpt_path in checkpoint_paths:

        print("\n" + "=" * 60)

        ckpt_name = os.path.basename(
            ckpt_path
        ).replace(".pt", "")

        print(
            f"Evaluating: {ckpt_name}"
        )

        model_name = infer_model_name(
            ckpt_name
        )

        if model_name is None:

            print(
                f"[WARNING] Unknown model: "
                f"{ckpt_name}"
            )

            continue

        model = get_model(
            model_name,
            device=DEVICE
        )

        checkpoint = torch.load(
            ckpt_path,
            map_location=DEVICE
        )

        if "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

        results = evaluate_model(
            model,
            test_loader
        )

        # =================================================
        # PRINT RESULTS
        # =================================================

        print(
            f"Accuracy            : "
            f"{results['accuracy']:.4f}"
        )

        print(
            f"Cross Entropy       : "
            f"{results['cross_entropy']:.4f}"
        )

        print(
            f"KL Divergence       : "
            f"{results['kl_divergence']:.4f}"
        )

        print(
            f"JSD                 : "
            f"{results['jsd']:.4f}"
        )

        print(
            f"Cosine Similarity   : "
            f"{results['cosine_similarity']:.4f}"
        )

        print(
            f"Pearson r           : "
            f"{results['pearson_r']:.4f}"
        )

        print(
            f"Spearman r          : "
            f"{results['spearman_r']:.4f}"
        )

        print(
            f"Precision@3         : "
            f"{results['precision_at_3']:.4f}"
        )

        # =================================================
        # SAVE CONFUSION MATRIX
        # =================================================

        save_confusion_matrix(

            results["labels"],

            results["predictions"],

            ckpt_name
        )

        # =================================================
        # STORE FINAL RESULTS
        # =================================================

        final_results.append({

            "model":
                ckpt_name,

            "accuracy":
                results["accuracy"],

            "cross_entropy":
                results["cross_entropy"],

            "kl_divergence":
                results["kl_divergence"],

            "jsd":
                results["jsd"],

            "cosine_similarity":
                results["cosine_similarity"],

            "pearson_r":
                results["pearson_r"],

            "spearman_r":
                results["spearman_r"],

            "precision_at_3":
                results["precision_at_3"]
        })

    # =====================================================
    # SAVE CSV + JSON
    # =====================================================

    df = pd.DataFrame(final_results)

    csv_path = os.path.join(
        RESULT_DIR,
        "evaluation_summary.csv"
    )

    json_path = os.path.join(
        RESULT_DIR,
        "evaluation_summary.json"
    )

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:

        json.dump(
            final_results,
            f,
            indent=4
        )

    print("\n" + "=" * 60)

    print("FINAL RESULTS")

    print("=" * 60)

    print(df)

    print(
        f"\nSaved CSV  : {csv_path}"
    )

    print(
        f"Saved JSON : {json_path}"
    )

    print(
        f"Plots dir  : {PLOT_DIR}"
    )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()