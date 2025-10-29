import argparse
import torch

from src.models import FashionMNISTModel, ResNet18_Tiny
from src.continual import incremental_learning
from datasets import IncrementalFashionMNIST, IncrementalTinyImageNet


def main():
    parser = argparse.ArgumentParser(description="Train incremental model and save checkpoints")
    parser.add_argument("--increment", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--method", type=str, default="normal", choices=["normal", "replay"]) 
    parser.add_argument("--memory_size", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="experiments/fashion_mnist_incremental")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    parser.add_argument("--no_comprehensive_eval", action="store_true", help="Skip comprehensive evaluation after training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST()
        model = FashionMNISTModel()
        num_classes = 10
        
    elif args.dataset == "tiny_imagenet":
        data_manager = IncrementalTinyImageNet()
        model = ResNet18_Tiny()
        num_classes = 200
    else:
        raise ValueError("Invalid dataset")
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    incremental_learning(
        model,
        data_manager,
        epochs=args.epochs,
        device=device,
        num_classes=num_classes,
        increment=args.increment,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=args.batch_size,
        val_loader=None,
        method=args.method,
        memory_size=args.memory_size,
        save_dir=args.save_dir,
        run_comprehensive_eval=not args.no_comprehensive_eval,
    )


if __name__ == "__main__":
    main()


# Exampole command:
# python run_experiment.py --increment 2 --epochs 1 --batch_size 64 --method replay --memory_size 5000 --save_dir experiments/fashion_mnist_incremental --dataset fashion_mnist