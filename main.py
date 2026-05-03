import argparse
import torch

from member3_training.train import run_kfold_experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="lightcnn"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=2
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("CIFAR10H PROJECT")
    print("=" * 60)

    print("Training SOFT LABEL model")

    run_kfold_experiment(
        model_name=args.model,
        k=args.folds,
        use_soft=True,
        epochs=args.epochs
    )

    print("Training HARD LABEL baseline")

    run_kfold_experiment(
        model_name=args.model,
        k=args.folds,
        use_soft=False,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()