import argparse
import json
import os
import torch

from src.models import CognitiveRNN
from src.continual import sequential_learning
from datasets import get_default_config, get_task_generator, DEFAULT_TASKS


def main():
    parser = argparse.ArgumentParser(description="Sequential learning experiment for cognitive RNNs")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--sigma_rec", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="softplus", choices=["softplus", "tanh", "relu", "retanh"])
    parser.add_argument("--w_rec_init", type=str, default="diag", choices=["diag", "randortho", "randgauss"])
    parser.add_argument("--num_iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--method", type=str, default="normal", choices=["normal", "ewc", "replay", "lwf", "hypernet"])
    # EWC
    parser.add_argument("--ewc_lambda", type=float, default=100.0, help="EWC regularization strength")
    parser.add_argument("--fisher_samples", type=int, default=200, help="Samples for Fisher estimation")
    # Replay
    parser.add_argument("--memory_per_task", type=int, default=50, help="Trials stored per past task for replay")
    parser.add_argument("--replay_num_tasks", type=int, default=1, help="Past tasks to sample per replay step (default=1 for speed)")
    # LwF
    parser.add_argument("--lwf_lambda", type=float, default=1.0, help="LwF distillation loss weight")
    parser.add_argument("--lwf_temperature", type=float, default=2.0, help="LwF softmax temperature")
    # HyperNet
    parser.add_argument("--hnet_emb_dim", type=int, default=64, help="Task embedding dimension")
    parser.add_argument("--hnet_chunk_emb_dim", type=int, default=64, help="Chunk embedding dimension")
    parser.add_argument("--hnet_hidden", type=int, default=128, help="Hypernetwork hidden layer width")
    parser.add_argument("--hnet_chunks", type=int, default=10, help="Number of weight chunks")
    parser.add_argument("--hnet_beta", type=float, default=0.01, help="Output regularisation strength")
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        help=f"Task names to learn sequentially. Default: 18 tasks (excludes dmcgo/dmcnogo). Available: {DEFAULT_TASKS}")
    parser.add_argument("--early_stop_patience", type=int, default=500, help="Early stop if no improvement for this many iters (0=disable)")
    parser.add_argument("--early_stop_delta", type=float, default=1e-3, help="Min loss improvement to reset patience")
    parser.add_argument("--train_pool_size", type=int, default=200, help="Number of pre-generated training batches per task")
    parser.add_argument("--train_seed", type=int, default=12345, help="Base seed for training set generation (must differ from test seed=42)")
    parser.add_argument("--save_dir", type=str, default="experiments/rnn_drift")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (requires PyTorch 2.x)")
    args = parser.parse_args()

    # Save experiment configuration
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    print(f"Experiment configuration saved to {config_path}")

    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    from src.utils import set_seed
    set_seed(args.seed)

    # Task config
    config = get_default_config()

    # Build model
    model = CognitiveRNN(
        input_size=config['n_input'],
        hidden_size=args.hidden_size,
        output_size=config['n_output'],
        dt=config['dt'],
        tau=config['dt'] / config['alpha'],
        sigma_rec=args.sigma_rec,
        activation=args.activation,
        w_rec_init=args.w_rec_init,
    )
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Build task sequence from --tasks argument
    tasks = [(name, get_task_generator(name)) for name in args.tasks]
    print(f"Task sequence ({len(tasks)}): {[t[0] for t in tasks]}")

    # Run sequential learning via the orchestrator
    sequential_learning(
        model=model,
        tasks=tasks,
        config=config,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        method=args.method,
        save_dir=args.save_dir,
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        train_pool_size=args.train_pool_size,
        train_seed=args.train_seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        memory_per_task=args.memory_per_task,
        replay_num_tasks=args.replay_num_tasks,
        lwf_lambda=args.lwf_lambda,
        lwf_temperature=args.lwf_temperature,
        hnet_emb_dim=args.hnet_emb_dim,
        hnet_chunk_emb_dim=args.hnet_chunk_emb_dim,
        hnet_hidden=args.hnet_hidden,
        hnet_chunks=args.hnet_chunks,
        hnet_beta=args.hnet_beta,
    )


if __name__ == "__main__":
    main()


# Example commands:
# Baseline (no CL):
#   python run_experiment.py --method normal --save_dir experiments/rnn_normal
# EWC:
#   python run_experiment.py --method ewc --ewc_lambda 100 --save_dir experiments/rnn_ewc
# Replay:
#   python run_experiment.py --method replay --memory_per_task 50 --save_dir experiments/rnn_replay
# LwF:
#   python run_experiment.py --method lwf --lwf_lambda 1.0 --save_dir experiments/rnn_lwf
# HyperNet:
#   python run_experiment.py --method hypernet --hnet_beta 0.01 --save_dir experiments/rnn_hypernet
# Subset of tasks:
#   python run_experiment.py --tasks delaygo delayanti dm1 dm2 --method normal --save_dir experiments/rnn_subset
