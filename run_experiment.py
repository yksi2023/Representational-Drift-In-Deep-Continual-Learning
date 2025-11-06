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
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--method", type=str, default="normal", choices=["normal", "replay"]) 
    parser.add_argument("--memory_size", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="experiments/fashion_mnist_incremental")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    parser.add_argument("--no_comprehensive_eval", action="store_true", help="Skip comprehensive evaluation after training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum improvement in val loss to reset patience")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio for FashionMNIST")
    parser.add_argument("--no_validation", action="store_true", help="Disable validation/early stopping")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP) on CUDA")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile (PyTorch 2.x)")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format for CNNs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Backend optimizations
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'

        except Exception:
            pass
        try:
            # PyTorch 2.x matmul precision hint
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST(val_ratio=args.val_ratio)
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
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer =  torch.optim.Adam(model.parameters(), lr=0.001)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # minimize validation loss
        factor=0.5,           # reduce LR by half
        patience=3,           # wait 3 epochs before reducing
        min_lr=1e-4          # minimum learning rate
    )

    incremental_learning(
        model,
        data_manager,
        epochs=args.epochs,
        device=device,
        num_classes=num_classes,
        increment=args.increment,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=args.batch_size,
        method=args.method,
        memory_size=args.memory_size,
        save_dir=args.save_dir,
        run_comprehensive_eval=not args.no_comprehensive_eval,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        use_validation=not args.no_validation,
        use_amp=args.amp,
        compile_model=args.compile,
        channels_last=args.channels_last,
    )


if __name__ == "__main__":
    main()


# Exampole command:
# python run_experiment.py --increment 2 --epochs 1 --batch_size 64 --method replay --memory_size 5000 --save_dir experiments/fashion_mnist_incremental --dataset fashion_mnist