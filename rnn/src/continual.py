from src.methods import get_method


def sequential_learning(
    model,
    tasks,
    config,
    num_iterations=2000,
    batch_size=64,
    lr=0.001,
    device='cpu',
    method='normal',
    save_dir='./experiments/rnn_drift',
    ewc_lambda=100.0,
    fisher_samples=200,
    train_pool_size=200,
    train_seed=12345,
    memory_per_task=50,
    replay_num_tasks=1,
    lwf_lambda=1.0,
    lwf_temperature=2.0,
    hnet_emb_dim=64,
    hnet_chunk_emb_dim=64,
    hnet_hidden=128,
    hnet_chunks=10,
    hnet_beta=0.01,
):
    """
    Train the model sequentially on cognitive tasks.

    Args:
        model: CognitiveRNN instance.
        tasks: List of (task_name, task_generator_fn) tuples.
        config: Task configuration dictionary.
        num_iterations: Training iterations per task.
        batch_size: Batch size for training.
        lr: Learning rate.
        device: 'cpu' or 'cuda'.
        method: Continual learning method ('normal', 'ewc').
        save_dir: Directory to save checkpoints and results.
        ewc_lambda: EWC regularization strength (only used when method='ewc').
        fisher_samples: Number of samples for Fisher estimation (only used when method='ewc').
        train_pool_size: Number of pre-generated training batches per task.
        train_seed: Base seed for training set generation (distinct from test seed=42).
        memory_per_task: Trials stored per past task (only used when method='replay').
        replay_num_tasks: Past tasks to sample per replay step (only used when method='replay').
        lwf_lambda: Distillation loss weight (only used when method='lwf').
        lwf_temperature: Softmax temperature for distillation (only used when method='lwf').
        hnet_emb_dim: Task embedding dimension (only used when method='hypernet').
        hnet_chunk_emb_dim: Chunk embedding dimension (only used when method='hypernet').
        hnet_hidden: Hypernetwork hidden layer width (only used when method='hypernet').
        hnet_chunks: Number of weight chunks (only used when method='hypernet').
        hnet_beta: Output regularisation strength (only used when method='hypernet').
    """
    common_kwargs = {
        'model': model,
        'tasks': tasks,
        'config': config,
        'num_iterations': num_iterations,
        'batch_size': batch_size,
        'lr': lr,
        'device': device,
        'save_dir': save_dir,
        'train_pool_size': train_pool_size,
        'train_seed': train_seed,
    }

    method_kwargs = {}
    m = method.lower()
    if m == 'ewc':
        method_kwargs = {
            'ewc_lambda': ewc_lambda,
            'fisher_samples': fisher_samples,
        }
    elif m == 'replay':
        method_kwargs = {
            'memory_per_task': memory_per_task,
            'replay_num_tasks': replay_num_tasks,
        }
    elif m == 'lwf':
        method_kwargs = {
            'lwf_lambda': lwf_lambda,
            'lwf_temperature': lwf_temperature,
        }
    elif m == 'hypernet':
        method_kwargs = {
            'hnet_emb_dim': hnet_emb_dim,
            'hnet_chunk_emb_dim': hnet_chunk_emb_dim,
            'hnet_hidden': hnet_hidden,
            'hnet_chunks': hnet_chunks,
            'hnet_beta': hnet_beta,
        }

    method_cls = get_method(method)
    learner = method_cls(**common_kwargs, **method_kwargs)

    return learner.run()
