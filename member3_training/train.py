# =========================================================
# member3_training/train.py
# =========================================================

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler

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
# CONSTANTS
# =========================================================

NUM_EPOCHS  = 150
BATCH_SIZE  = 32
BASE_LR     = 0.001
PATIENCE    = 20
NUM_CLASSES = 10
SEED        = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# =========================================================
# SEED
# =========================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================================================
# LOSSES
# =========================================================

class SoftKLLoss(nn.Module):

    def forward(self, logits, soft_targets):

        log_probs = F.log_softmax(logits, dim=1)

        return F.kl_div(
            log_probs,
            soft_targets,
            reduction='batchmean'
        )


class JSDLoss(nn.Module):

    def forward(self, logits, soft_targets):

        probs = F.softmax(logits, dim=1)

        m = 0.5 * (probs + soft_targets)

        jsd = 0.5 * (
            F.kl_div(
                (probs + 1e-8).log(),
                m,
                reduction='batchmean'
            )
            +
            F.kl_div(
                (soft_targets + 1e-8).log(),
                m,
                reduction='batchmean'
            )
        )

        return jsd


class EntropyPenaltyLoss(nn.Module):

    def __init__(self, alpha=0.1):

        super().__init__()

        self.alpha = alpha

    def forward(self, logits, soft_targets):

        log_probs = F.log_softmax(logits, dim=1)

        probs = F.softmax(logits, dim=1)

        kl = F.kl_div(
            log_probs,
            soft_targets,
            reduction='batchmean'
        )

        pred_entropy = -(
            probs * log_probs
        ).sum(dim=1).mean()

        target_entropy = -(
            soft_targets *
            (soft_targets + 1e-8).log()
        ).sum(dim=1).mean()

        entropy_penalty = torch.abs(
            pred_entropy - target_entropy
        )

        return kl + self.alpha * entropy_penalty


def get_loss_function(name):

    if name == "kl":
        return SoftKLLoss()

    elif name == "jsd":
        return JSDLoss()

    elif name == "entropy_penalty":
        return EntropyPenaltyLoss()

    else:
        raise ValueError(f"Unknown loss: {name}")


# =========================================================
# METRICS
# =========================================================

@torch.no_grad()
def compute_metrics(logits, soft_targets, hard_labels):

    probs = F.softmax(logits, dim=1)

    preds = probs.argmax(dim=1)

    acc = (
        preds == hard_labels
    ).float().mean().item()

    pred_entropy = -(
        probs *
        (probs + 1e-8).log()
    ).sum(dim=1)

    target_entropy = -(
        soft_targets *
        (soft_targets + 1e-8).log()
    ).sum(dim=1)

    entropy_r = np.corrcoef(
        pred_entropy.cpu().numpy(),
        target_entropy.cpu().numpy()
    )[0, 1]

    return acc, entropy_r


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device
):

    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for imgs, soft_labels, hard_labels in loader:

        imgs = imgs.to(device)

        soft_labels = soft_labels.to(device)

        hard_labels = hard_labels.to(device)

        optimizer.zero_grad()

        with autocast(
            enabled=device.type == 'cuda'
        ):

            logits = model(imgs)

            loss = loss_fn(
                logits,
                soft_labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        acc, _ = compute_metrics(
            logits,
            soft_labels,
            hard_labels
        )

        running_loss += loss.item()

        running_acc += acc

    return (
        running_loss / len(loader),
        running_acc / len(loader)
    )


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    device
):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_er  = 0.0

    for imgs, soft_labels, hard_labels in loader:

        imgs = imgs.to(device)

        soft_labels = soft_labels.to(device)

        hard_labels = hard_labels.to(device)

        logits = model(imgs)

        loss = loss_fn(
            logits,
            soft_labels
        )

        acc, er = compute_metrics(
            logits,
            soft_labels,
            hard_labels
        )

        running_loss += loss.item()

        running_acc += acc

        running_er += er

    return (
        running_loss / len(loader),
        running_acc / len(loader),
        running_er / len(loader)
    )


# =========================================================
# MAIN TRAINING PIPELINE
# =========================================================

def run_training(

    model_name,

    head_type,

    loss_name,

    init_strategy,

    num_epochs,

    batch_size,

    learning_rate,

    output_dir,

    data_root,

    pretrain_ckpt=None
):

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(
        output_dir,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    run_id = (
        f"{model_name}_"
        f"{head_type}_"
        f"{loss_name}_"
        f"soft"
    )

    print("\n" + "=" * 65)

    print(f"  TRAINING RUN: {run_id}")

    print(f"  Device        : {DEVICE}")

    print(f"  Init strategy : {init_strategy}")

    print(f"  Epochs (max)  : "
          f"{num_epochs}  |  Patience: {PATIENCE}")

    print("=" * 65)

    train_loader, val_loader, _ = get_split_loaders(
        batch_size=batch_size
    )

    model = get_model(
        model_name,
        device=DEVICE
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"  Parameters: {n_params:,}")

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    scaler = GradScaler(
        enabled=DEVICE.type == 'cuda'
    )

    loss_fn = get_loss_function(
        loss_name
    )

    start_epoch = 1

    if (
        pretrain_ckpt is not None
        and
        os.path.exists(pretrain_ckpt)
    ):

        print(
            f"[INFO] Loading checkpoint: "
            f"{pretrain_ckpt}"
        )

        checkpoint = torch.load(
            pretrain_ckpt,
            map_location=DEVICE
        )

        if 'state_dict' in checkpoint:

            model.load_state_dict(
                checkpoint['state_dict']
            )

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(
                    checkpoint['optimizer']
                )

            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(
                    checkpoint['scheduler']
                )

            if 'epoch' in checkpoint:
                start_epoch = (
                    checkpoint['epoch'] + 1
                )

        else:
            model.load_state_dict(
                checkpoint
            )

        print(
            f"[INFO] Resuming from epoch "
            f"{start_epoch}"
        )

    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    best_entropy_r = 0.0

    no_improve = 0

    ckpt_path = os.path.join(
        checkpoint_dir,
        f"{run_id}_best.pt"
    )

    best_acc_ckpt = os.path.join(
        checkpoint_dir,
        f"{run_id}_best_acc.pt"
    )

    train_losses    = []
    val_losses      = []
    val_accs        = []
    val_entropy_rs  = []

    for epoch in range(
        start_epoch,
        num_epochs + 1
    ):

        tr_loss, tr_acc = train_one_epoch(

            model,

            train_loader,

            optimizer,

            loss_fn,

            scaler,

            DEVICE
        )

        vl_loss, vl_acc, vl_er = validate(

            model,

            val_loader,

            loss_fn,

            DEVICE
        )

        scheduler.step()

        train_losses.append(tr_loss)

        val_losses.append(vl_loss)

        val_accs.append(vl_acc)

        val_entropy_rs.append(vl_er)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch {epoch:3d}/{num_epochs} | "
            f"tr_loss={tr_loss:.4f} | "
            f"vl_loss={vl_loss:.4f} | "
            f"vl_acc={vl_acc:.3f} | "
            f"entropy_r={vl_er:.3f} | "
            f"lr={current_lr:.5f}"
        )

        # -----------------------------------------
        # BEST LOSS
        # -----------------------------------------

        if vl_loss < best_val_loss:

            best_val_loss  = vl_loss
            best_entropy_r = vl_er

            no_improve = 0

            torch.save({

                'epoch'         : epoch,

                'run_id'        : run_id,

                'model_name'    : model_name,

                'head_type'     : head_type,

                'loss_name'     : loss_name,

                'init_strategy' : init_strategy,

                'state_dict'    : model.state_dict(),

                'optimizer'     : optimizer.state_dict(),

                'scheduler'     : scheduler.state_dict(),

                'val_loss'      : vl_loss,

                'val_acc'       : vl_acc,

                'val_entropy_r' : vl_er,

            }, ckpt_path)

        else:

            no_improve += 1

        # -----------------------------------------
        # BEST ACCURACY
        # -----------------------------------------

        if vl_acc > best_val_acc:

            best_val_acc = vl_acc

            torch.save({

                'epoch'      : epoch,

                'state_dict' : model.state_dict(),

                'val_acc'    : vl_acc,

            }, best_acc_ckpt)

        # -----------------------------------------
        # EARLY STOPPING
        # -----------------------------------------

        if no_improve >= PATIENCE:

            print(
                f"\n  [Early stop] "
                f"No val improvement "
                f"for {PATIENCE} epochs "
                f"(epoch {epoch}). Stopping."
            )

            break

    # =====================================================
    # SAVE PLOTS
    # =====================================================

    plot_dir = os.path.join(
        output_dir,
        "plots"
    )

    os.makedirs(plot_dir, exist_ok=True)

    # LOSS CURVE

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label='train')

    plt.plot(val_losses, label='val')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title(run_id)

    plt.legend()

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{run_id}_loss.png"
        )
    )

    plt.close()

    # ACCURACY CURVE

    plt.figure(figsize=(8, 5))

    plt.plot(val_accs)

    plt.xlabel('Epoch')

    plt.ylabel('Validation Accuracy')

    plt.title(run_id)

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{run_id}_accuracy.png"
        )
    )

    plt.close()

    # ENTROPY CURVE

    plt.figure(figsize=(8, 5))

    plt.plot(val_entropy_rs)

    plt.xlabel('Epoch')

    plt.ylabel('Entropy Correlation')

    plt.title(run_id)

    plt.savefig(
        os.path.join(
            plot_dir,
            f"{run_id}_entropy.png"
        )
    )

    plt.close()

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    results = {

        'run_id': run_id,

        'best_val_loss': best_val_loss,

        'best_val_acc': best_val_acc,

        'best_entropy_r': best_entropy_r,

        'epochs_completed': epoch,

        'train_losses': train_losses,

        'val_losses': val_losses,

        'val_accs': val_accs,

        'val_entropy_rs': val_entropy_rs
    }

    json_path = os.path.join(
        output_dir,
        f"{run_id}_results.json"
    )

    with open(json_path, 'w') as f:

        json.dump(results, f, indent=4)

    print(
        f"\n[INFO] Saved results → "
        f"{json_path}"
    )

    print(
        f"\n  Best val loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"  Best val acc  : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"  Best entropy_r: "
        f"{best_entropy_r:.4f}"
    )

    return results


# =========================================================
# RUN ALL LOSSES
# =========================================================

def run_all_losses(args):

    losses = [
        "kl",
        "jsd",
        "entropy_penalty"
    ]

    all_results = {}

    for loss_name in losses:

        results = run_training(

            model_name=args.model,

            head_type=args.head,

            loss_name=loss_name,

            init_strategy=args.init,

            num_epochs=args.epochs,

            batch_size=args.batch,

            learning_rate=args.lr,

            output_dir=args.outdir,

            data_root=args.data,

            pretrain_ckpt=args.ckpt
        )

        all_results[loss_name] = results

    summary_path = os.path.join(
        args.outdir,
        f"{args.model}_loss_summary.json"
    )

    with open(summary_path, 'w') as f:

        json.dump(all_results, f, indent=4)

    print(
        f"\n[INFO] Saved summary → "
        f"{summary_path}"
    )


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=
        'CIFAR10H Soft Label Training'
    )

    p.add_argument(
        '--model',
        type=str,
        default='resnet_basic_110'
    )

    p.add_argument(
        '--head',
        type=str,
        default='linear'
    )

    p.add_argument(
        '--loss',
        type=str,
        default='kl'
    )

    p.add_argument(
        '--init',
        type=str,
        default='random'
    )

    p.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS
    )

    p.add_argument(
        '--batch',
        type=int,
        default=BATCH_SIZE
    )

    p.add_argument(
        '--lr',
        type=float,
        default=BASE_LR
    )

    p.add_argument(
        '--outdir',
        type=str,
        default='./results'
    )

    p.add_argument(
        '--data',
        type=str,
        default='./data'
    )

    p.add_argument(
        '--ckpt',
        type=str,
        default=None
    )

    return p.parse_args()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    if args.loss == "all":

        run_all_losses(args)

    else:

        run_training(

            model_name=args.model,

            head_type=args.head,

            loss_name=args.loss,

            init_strategy=args.init,

            num_epochs=args.epochs,

            batch_size=args.batch,

            learning_rate=args.lr,

            output_dir=args.outdir,

            data_root=args.data,

            pretrain_ckpt=args.ckpt
        )