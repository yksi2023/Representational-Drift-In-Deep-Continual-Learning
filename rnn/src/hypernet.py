"""
Chunked Hypernetwork for continual learning (von Oswald et al., 2020).

The hypernetwork generates all weights of a target network conditioned on a
task embedding.  To keep the hypernetwork itself compact, parameters are
produced in *chunks*: a shared MLP maps (task_emb, chunk_emb) → chunk of
target weights.  The full target weight vector is the concatenation of all
chunks.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: collect target-network parameter metadata
# ---------------------------------------------------------------------------

def get_target_param_info(target_model: nn.Module) -> Tuple[int, List[Tuple[str, torch.Size]]]:
    """Return (total_num_params, [(name, shape), ...]) for the target network."""
    info = []
    total = 0
    for name, p in target_model.named_parameters():
        info.append((name, p.shape))
        total += p.numel()
    return total, info


# ---------------------------------------------------------------------------
# Chunked Hypernetwork
# ---------------------------------------------------------------------------

class ChunkedHyperNetwork(nn.Module):
    """
    Generates the full weight vector of a target network from a task embedding.

    Architecture (per chunk):
        input  = concat(task_emb, chunk_emb)          dim = emb_dim + chunk_emb_dim
        hidden = Linear → ReLU → Linear               dim → hyper_hidden → chunk_size

    The chunk embeddings are *learnable* and shared across tasks.
    Task embeddings are stored externally (one per task, created on demand).

    Args:
        target_num_params: total number of target-network parameters.
        num_chunks: how many chunks to split the weight vector into.
        emb_dim: dimensionality of task embeddings.
        chunk_emb_dim: dimensionality of chunk embeddings.
        hyper_hidden: hidden-layer width of the shared chunk generator.
    """

    def __init__(
        self,
        target_num_params: int,
        num_chunks: int = 10,
        emb_dim: int = 64,
        chunk_emb_dim: int = 64,
        hyper_hidden: int = 128,
    ):
        super().__init__()
        self.target_num_params = target_num_params
        self.num_chunks = num_chunks
        self.emb_dim = emb_dim
        self.chunk_emb_dim = chunk_emb_dim

        # Chunk size (last chunk may be smaller)
        self.chunk_size = math.ceil(target_num_params / num_chunks)
        # Pad so all chunks have equal size; we trim when returning
        self.padded_total = self.chunk_size * num_chunks

        # Learnable chunk embeddings
        self.chunk_embs = nn.Parameter(torch.randn(num_chunks, chunk_emb_dim) * 0.01)

        # Shared chunk generator MLP
        in_dim = emb_dim + chunk_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hyper_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden, self.chunk_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Small init for final layer to start near zero
        final = list(self.mlp.children())[-1]
        nn.init.normal_(final.weight, std=0.01)

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_emb: (emb_dim,) task embedding vector.

        Returns:
            weights: (target_num_params,) flattened target-network weights.
        """
        chunks = []
        for c in range(self.num_chunks):
            inp = torch.cat([task_emb, self.chunk_embs[c]], dim=-1)  # (emb_dim + chunk_emb_dim)
            chunk = self.mlp(inp)                                     # (chunk_size,)
            chunks.append(chunk)
        full = torch.cat(chunks, dim=-1)                              # (padded_total,)
        return full[:self.target_num_params]

    def generate_and_load(self, task_emb: torch.Tensor, target_model: nn.Module, param_info):
        """Generate weights and load them into the target model (in-place)."""
        flat_weights = self.forward(task_emb)
        offset = 0
        for name, shape in param_info:
            numel = 1
            for s in shape:
                numel *= s
            param_data = flat_weights[offset:offset + numel].view(shape)
            # Navigate to the parameter and set its data
            parts = name.split('.')
            mod = target_model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            getattr(mod, parts[-1]).data.copy_(param_data)
            offset += numel
