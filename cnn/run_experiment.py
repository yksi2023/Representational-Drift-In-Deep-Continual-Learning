import argparse
import torch
import json
import os

from src.models import MODEL_CHOICES, MODEL_DEFAULTS, build_model
from src.continual import incremental_learning
from datasets import DATASET_CHOICES, build_dataset


def main():
    parser = argparse.ArgumentParser(description="Train incremental model and save checkpoints")
    parser.add_argument("--increment", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--method", type=str, default="normal", choices=["normal", "replay", "ewc", "gpm", "lwf"])
    parser.add_argument("--memory_size", type=int, default=5000,
                        help="Total replay buffer budget (used only when --memory_per_class is unset).")
    parser.add_argument("--memory_per_class", type=int, default=None,
                        help="iCaRL-style per-class exemplar count. If set, overrides --memory_size; total buffer grows linearly with classes seen.")
    parser.add_argument("--first_task_only_memory", action="store_true", help="Only keep first task data in memory, do not add subsequent task data")
    parser.add_argument("--save_dir", type=str, default="experiments/fashion_mnist_incremental")
    parser.add_argument("--dataset", type=str, default="tiny_imagenet", choices=DATASET_CHOICES)
    parser.add_argument("--model", type=str, default=None, choices=MODEL_CHOICES,
                        help="Model architecture. Defaults per dataset (see MODEL_DEFAULTS in src/models.py).")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override total classes (mainly for CIFAR-100 subsetting). Defaults per dataset.")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Override input image size. Defaults per dataset.")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="For resnet18_pretrained / bit_r50x1: disable pretrained weights.")
    parser.add_argument("--no_comprehensive_eval", action="store_true", help="Skip comprehensive evaluation after training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum improvement in val loss to reset patience")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio for FashionMNIST")
    parser.add_argument("--no_validation", action="store_true", help="Disable validation/early stopping")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP) on CUDA")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile (PyTorch 2.x)")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format for CNNs")
    # Freezing options (applicable to resnet18_pretrained and bit_r50x1)
    parser.add_argument("--freeze_layers", type=str, default="", help="Comma-separated layer names (resnet18_pretrained only)")
    parser.add_argument("--freeze_until", type=str, default=None,
                        help="Freeze all layers up to and including this one. Names differ per model.")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC regularization strength (only used when method=ewc)")
    parser.add_argument("--lwf_lambda", type=float, default=1.0, help="LwF distillation strength (only used when method=lwf)")
    parser.add_argument("--lwf_temperature", type=float, default=2.0, help="LwF distillation temperature (only used when method=lwf)")
    parser.add_argument("--learning_mode", type=str, default="til", choices=["til", "cil"], help="Learning mode: 'til' (task-incremental, masked output) or 'cil' (class-incremental, full output)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "none"],
                        help="LR schedule per task. 'plateau'=ReduceLROnPlateau (val-driven), "
                             "'cosine'=CosineAnnealingLR over epochs (good for from-scratch CIFAR), "
                             "'none'=fixed LR.")
    args = parser.parse_args()

    # Save experiment configuration
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    print(f"Experiment configuration saved to {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Backend optimizations
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            # New API (PyTorch 2.9+): explicitly opt into TF32 for matmul & conv.
            # Don't call torch.set_float32_matmul_precision() -- in 2.9 it
            # internally flips the legacy flags and self-triggers a deprecation
            # warning from aten/.../Context.cpp:80.
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
        except Exception:
            pass

    # Resolve defaults from the dataset's registry entry
    defaults = MODEL_DEFAULTS[args.dataset]
    model_name = args.model or defaults["model"]
    num_classes = args.num_classes if args.num_classes is not None else defaults["num_classes"]
    img_size = args.img_size if args.img_size is not None else defaults["img_size"]

    # Persist resolved choices so analyze_drift.py can reconstruct the model
    args.model = model_name
    args.num_classes = num_classes
    args.img_size = img_size
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    data_manager = build_dataset(
        args.dataset,
        num_classes=num_classes,
        img_size=img_size,
        val_ratio=args.val_ratio,
    )
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_layers=args.freeze_layers,
        freeze_until=args.freeze_until,
    )
    print(f"Dataset: {args.dataset} (num_classes={num_classes}, img_size={img_size})")
    print(f"Model:   {model_name}")
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    # Split parameters: no weight decay on norm layers / biases.
    # Critical for GN+WS+Zero-gamma networks: if WD is applied to gn.gamma,
    # the zero-initialized gn2.gamma is held near zero and residual blocks
    # stay dead, capping first-task accuracy well below baseline.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": decay_params,    "weight_decay": 5e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.lr, momentum=args.momentum, nesterov=True,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": decay_params,    "weight_decay": 1e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=0.001,
        )

    # Scheduler configuration (will be created per task)
    if args.scheduler == "plateau":
        scheduler_config = {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 3,
            'min_lr': 1e-4,
        }
    elif args.scheduler == "cosine":
        scheduler_config = {
            'type': 'CosineAnnealingLR',
            'T_max': args.epochs,   # one cosine cycle per task
            'eta_min': 1e-4,
        }
    else:
        scheduler_config = None

    incremental_learning(
        model,
        data_manager,
        epochs=args.epochs,
        device=device,
        num_classes=num_classes,
        increment=args.increment,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        batch_size=args.batch_size,
        method=args.method,
        memory_size=args.memory_size,
        memory_per_class=args.memory_per_class,
        first_task_only_memory=args.first_task_only_memory,
        save_dir=args.save_dir,
        run_comprehensive_eval=not args.no_comprehensive_eval,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        use_validation=not args.no_validation,
        use_amp=args.amp,
        compile_model=args.compile,
        channels_last=args.channels_last,
        ewc_lambda=args.ewc_lambda,
        lwf_lambda=args.lwf_lambda,
        lwf_temperature=args.lwf_temperature,
        learning_mode=args.learning_mode,
    )


if __name__ == "__main__":
    main()


# Exampole command:
# python run_experiment.py --increment 2 --epochs 1 --batch_size 64 --method replay --memory_size 5000 --save_dir experiments/fashion_mnist_incremental --dataset fashion_mnist