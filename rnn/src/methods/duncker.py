"""
Duncker et al. (NeurIPS 2020) — Orthogonal Gradient Projection for Continual Learning.

"Organizing recurrent network dynamics by task-computation to enable continual learning."

Augmented-matrix formulation:
  Recurrent layer:  a = W_aug @ z,  W_aug = [W_hh | W_ih | b],  z = [h; x; 1]
  Readout layer:    y = W_out_aug @ h_aug,  W_out_aug = [W_out | c],  h_aug = [h; 1]

After task k, collect M samples z_t = [h_{t-1}; x_t; 1] (all timesteps, noise off),
build covariances Σ_z and Σ_wz = W_aug Σ_z W_aug^T, construct soft projections:
  P = V diag(α / (α + λ_i)) V^T

Projected gradient update:
  G_CL = P_wz @ G_aug @ P_z     (recurrent layer, double-sided)
  G_out_CL = P_wy @ G_out_aug @ P_h   (readout layer, double-sided)

Adapted from the original TensorFlow implementation (seqMultiTaskRNN) to PyTorch.
"""

import torch
import numpy as np
from src.methods.base import BaseMethod
from src.train import compute_loss


def build_projection(cov, alpha):
    """
    Build a soft null-space projection from a covariance matrix.

    Equivalent to (I + (1/α') Σ)^{-1} with appropriate α' scaling.
    Implemented via eigendecomposition (no explicit inverse):
        P = V diag(α / (α + λ_i)) V^T

    Large eigenvalues → factor ≈ 0 (direction blocked).
    Small eigenvalues → factor ≈ 1 (direction open).

    Args:
        cov: symmetric PSD covariance matrix, shape (d, d)
        alpha: soft threshold (smaller = stricter protection)

    Returns:
        P: projection matrix, shape (d, d)
    """
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    scaling = alpha / (alpha + evals)
    return (evecs * scaling[np.newaxis, :]) @ evecs.T


class DunckerMethod(BaseMethod):
    """
    Orthogonal Gradient Projection (Duncker et al., NeurIPS 2020).

    Augmented-matrix formulation with double-sided projection.

    Recurrent layer:
        W_aug = [W_hh | W_ih | b]   shape (n_h, n_h + n_x + 1)
        z     = [h; x; 1]           shape (n_h + n_x + 1,)
        a     = W_aug @ z           shape (n_h,)
        G_CL  = P_wz @ G_aug @ P_z

    Readout layer:
        W_out_aug = [W_out | c]     shape (n_y, n_h + 1)
        h_aug     = [h; 1]          shape (n_h + 1,)
        y         = W_out_aug @ h_aug
        G_out_CL  = P_wy @ G_out_aug @ P_h
    """

    def __init__(self, duncker_alpha=0.001, duncker_samples=1024, **kwargs):
        super().__init__(**kwargs)
        self.duncker_alpha = duncker_alpha  # projection regularization (NOT Euler alpha=dt/tau)
        self.duncker_samples = duncker_samples

        # Projection matrices (None until first task is completed)
        # Recurrent layer projections
        self.P_z = None    # input-side, shape (n_h + n_x + 1, n_h + n_x + 1)
        self.P_wz = None   # output-side (preactivation), shape (n_h, n_h)
        # Readout layer projections
        self.P_h = None    # input-side, shape (n_h + 1, n_h + 1)
        self.P_wy = None   # output-side, shape (n_y, n_y)

        # Running average covariances (numpy, computed on CPU)
        self.cov_z = None       # (n_h + n_x + 1, n_h + n_x + 1)
        self.cov_h_aug = None   # (n_h + 1, n_h + 1)
        self.cov_wz = None      # (n_h, n_h)
        self.cov_wy = None      # (n_y, n_y)
        self.tasks_completed = 0

    def train_step(self, optimizer, trial):
        """
        Training step: backward → clip → Adam step → project the REALIZED update.

        IMPORTANT: We project the actual parameter update (Δθ) AFTER optimizer.step(),
        not the raw gradient before it. With Adam, the applied update is
            Δθ = lr * m_hat / (sqrt(v_hat) + eps)
        whose per-coordinate rescaling re-mixes a pre-projected gradient OUT of the
        protected subspace, nullifying the projection. The original Duncker optimizer
        (opt_tools.AdamOptimizer_withProjection) projects the computed update, not the
        gradient — we replicate that here via snapshot/step/project-delta.
        """
        self.model.train()
        optimizer.zero_grad()

        x, y, mask = trial.to_tensor(device=self.device)
        outputs = self.model(x, return_all_states=False)
        loss = compute_loss(outputs, y, mask, loss_type=trial.config.get('loss_type', 'cross_entropy'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        if self.P_z is not None:
            # Snapshot projected params, let Adam compute its full update, then
            # project the realized delta back into the protected subspace.
            model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            rnn_cell = model.rnn_cell
            readout = model.readout
            snapshot = (
                rnn_cell.weight_hh.data.clone(),
                rnn_cell.weight_ih.data.clone(),
                rnn_cell.bias.data.clone(),
                readout.weight.data.clone(),
                readout.bias.data.clone(),
            )
            optimizer.step()
            self._project_update(model, snapshot)
        else:
            optimizer.step()

        return loss.item()

    def _project_update(self, model, snapshot):
        """
        Double-sided projection of the REALIZED Adam update Δθ = θ_new - θ_old.

        Recurrent layer:
            D_aug = [ΔW_hh | ΔW_ih | Δb]      shape (n_h, n_h + n_x + 1)
            D_CL  = P_wz @ D_aug @ P_z          shape (n_h, n_h + n_x + 1)
            θ_new = θ_old + D_CL  (split back into W_hh, W_ih, b)

        Readout layer:
            D_out_aug = [ΔW_out | Δc]          shape (n_y, n_h + 1)
            D_out_CL  = P_wy @ D_out_aug @ P_h
            θ_new = θ_old + D_out_CL

        PyTorch stores weight_ih as (n_h, n_x) and weight_hh as (n_h, n_h),
        which is ALREADY W_aug's native layout (output_dim, input_dim).
        No transposition needed for assembly.
        """
        rnn_cell = model.rnn_cell
        readout = model.readout
        n_h = rnn_cell.hidden_size
        n_x = rnn_cell.input_size
        hh0, ih0, b0, wout0, c0 = snapshot

        with torch.no_grad():
            # --- Recurrent layer: realized deltas ---
            d_hh = rnn_cell.weight_hh.data - hh0            # (n_h, n_h)
            d_ih = rnn_cell.weight_ih.data - ih0            # (n_h, n_x)
            d_b = (rnn_cell.bias.data - b0).unsqueeze(1)    # (n_h, 1)

            D_aug = torch.cat([d_hh, d_ih, d_b], dim=1)     # (n_h, n_h + n_x + 1)
            D_CL = self.P_wz @ D_aug @ self.P_z             # (n_h, n_h + n_x + 1)

            rnn_cell.weight_hh.data.copy_(hh0 + D_CL[:, :n_h])
            rnn_cell.weight_ih.data.copy_(ih0 + D_CL[:, n_h:n_h + n_x])
            rnn_cell.bias.data.copy_(b0 + D_CL[:, -1])

            # --- Readout layer: realized deltas ---
            d_wout = readout.weight.data - wout0            # (n_y, n_h)
            d_c = (readout.bias.data - c0).unsqueeze(1)     # (n_y, 1)

            D_out_aug = torch.cat([d_wout, d_c], dim=1)     # (n_y, n_h + 1)
            D_out_CL = self.P_wy @ D_out_aug @ self.P_h     # (n_y, n_h + 1)

            readout.weight.data.copy_(wout0 + D_out_CL[:, :n_h])
            readout.bias.data.copy_(c0 + D_out_CL[:, -1])

    def after_task(self, task_idx, task_name, task_gen_fn):
        """
        After training on task k (before task k+1 begins):
        1. Collect M samples of z_t = [h_{t-1}; x_t; 1] from old-task rollouts (noise OFF)
        2. Compute covariances Σ_z and Σ_wz = W_aug Σ_z W_aug^T
        3. Incrementally update running-average covariances across tasks
        4. Rebuild projection matrices from accumulated covariances

        Timing guarantee: projections from tasks 1..k are applied when learning task k+1.
        Current task activity is added AFTER training, so no leakage.
        """
        print(f"  [Duncker] Computing activity covariances for '{task_name}' "
              f"(alpha={self.duncker_alpha}, samples={self.duncker_samples})...")

        self.model.eval()  # noise OFF (sigma only active during training)
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        rnn_cell = model.rnn_cell
        n_h = rnn_cell.hidden_size
        n_x = rnn_cell.input_size
        n_y = model.readout.out_features

        # ----------------------------------------------------------
        # Collect z_t = [h_{t-1}; x_t; 1] and h_aug_t = [h_t; 1]
        # from deterministic rollouts on the just-completed task.
        # ----------------------------------------------------------
        all_z = []      # will hold (n_h + n_x + 1, ?) chunks
        all_h_aug = []  # will hold (n_h + 1, ?) chunks

        with torch.no_grad():
            remaining = self.duncker_samples
            pool = self.fixed_train_sets[task_name]
            idx = 0
            while remaining > 0:
                trial = pool[idx % len(pool)]
                x_t, _, _ = trial.to_tensor(device=self.device)
                # x_t: (T, B, n_x)
                _, states = self.model(x_t, return_all_states=True)
                # states[t] is h_t after consuming x_t.
                # Recurrent preactivation at time t uses h_{t-1}, so align x_t with h_prev[t].
                h0 = torch.zeros(1, states.size(1), n_h, device=self.device)
                h_prev = torch.cat([h0, states[:-1]], dim=0)

                T, B, _ = x_t.shape
                take = min(B, remaining)

                # Flatten time and batch → M columns
                x_flat = x_t[:, :take, :].reshape(-1, n_x)       # (T*take, n_x)
                h_prev_flat = h_prev[:, :take, :].reshape(-1, n_h)  # (T*take, n_h)
                h_readout_flat = states[:, :take, :].reshape(-1, n_h)  # (T*take, n_h)
                ones = torch.ones(T * take, 1, device=self.device)

                # z_s = [h_{t-1}; x_t; 1], stored as rows then transposed
                z_block = torch.cat([h_prev_flat, x_flat, ones], dim=1).t()  # (n_h+n_x+1, T*take)
                # readout uses h_t, not h_{t-1}
                h_aug_block = torch.cat([h_readout_flat, ones], dim=1).t()      # (n_h+1, T*take)

                all_z.append(z_block.cpu())
                all_h_aug.append(h_aug_block.cpu())

                remaining -= take
                idx += 1

        # Z_old: (n_h + n_x + 1, M),  H_old: (n_h + 1, M)
        Z_old = torch.cat(all_z, dim=1).numpy()
        H_old = torch.cat(all_h_aug, dim=1).numpy()
        M = Z_old.shape[1]

        # ----------------------------------------------------------
        # Covariances for this task
        # ----------------------------------------------------------
        # Input-side covariance for recurrent layer
        Sigma_z_task = Z_old @ Z_old.T / (M - 1)  # (n_h + n_x + 1, n_h + n_x + 1)

        # Input-side covariance for readout layer
        Sigma_h_task = H_old @ H_old.T / (M - 1)  # (n_h + 1, n_h + 1)

        W_hh = rnn_cell.weight_hh.data.cpu().numpy()          # (n_h, n_h)
        W_ih = rnn_cell.weight_ih.data.cpu().numpy()          # (n_h, n_x)
        b = rnn_cell.bias.data.cpu().numpy().reshape(-1, 1)   # (n_h, 1)
        W_aug = np.concatenate([W_hh, W_ih, b], axis=1)       # (n_h, n_h + n_x + 1)
        Sigma_wz_task = W_aug @ Sigma_z_task @ W_aug.T        # (n_h, n_h)

        W_out = model.readout.weight.data.cpu().numpy()       # (n_y, n_h)
        c = model.readout.bias.data.cpu().numpy().reshape(-1, 1)  # (n_y, 1)
        W_out_aug = np.concatenate([W_out, c], axis=1)        # (n_y, n_h + 1)
        Sigma_wy_task = W_out_aug @ Sigma_h_task @ W_out_aug.T  # (n_y, n_y)

        # ----------------------------------------------------------
        # Running average across all completed tasks
        # ----------------------------------------------------------
        k = self.tasks_completed
        if k == 0:
            self.cov_z = Sigma_z_task
            self.cov_h_aug = Sigma_h_task
            self.cov_wz = Sigma_wz_task
            self.cov_wy = Sigma_wy_task
        else:
            self.cov_z = k / (k + 1) * self.cov_z + Sigma_z_task / (k + 1)
            self.cov_h_aug = k / (k + 1) * self.cov_h_aug + Sigma_h_task / (k + 1)
            self.cov_wz = k / (k + 1) * self.cov_wz + Sigma_wz_task / (k + 1)
            self.cov_wy = k / (k + 1) * self.cov_wy + Sigma_wy_task / (k + 1)

        self.tasks_completed += 1

        # ----------------------------------------------------------
        # Build projection matrices from accumulated covariances
        # ----------------------------------------------------------
        alpha = self.duncker_alpha

        # --- Recurrent layer ---
        # P_z: input-side projection, shape (n_h + n_x + 1, n_h + n_x + 1)
        P_z_np = build_projection(self.cov_z, alpha)

        # P_wz: output-side (preactivation) projection, shape (n_h, n_h)
        P_wz_np = build_projection(self.cov_wz, alpha)

        # --- Readout layer ---
        # P_h: input-side projection, shape (n_h + 1, n_h + 1)
        P_h_np = build_projection(self.cov_h_aug, alpha)

        # P_wy: output-side projection, shape (n_y, n_y)
        P_wy_np = build_projection(self.cov_wy, alpha)

        # ----------------------------------------------------------
        # Convert to torch tensors on model device
        # ----------------------------------------------------------
        param_dtype = next(model.parameters()).dtype
        self.P_z = torch.tensor(P_z_np, dtype=param_dtype, device=self.device)
        self.P_wz = torch.tensor(P_wz_np, dtype=param_dtype, device=self.device)
        self.P_h = torch.tensor(P_h_np, dtype=param_dtype, device=self.device)
        self.P_wy = torch.tensor(P_wy_np, dtype=param_dtype, device=self.device)

        # Diagnostics: how strongly does each projection block its subspace?
        # blocked_frac = fraction of the trace removed by the projection
        #   = 1 - tr(P Σ P) / tr(Σ)  (≈1 means strong protection, ≈0 means projection ≈ identity)
        def _blocked_frac(P_np, cov_np):
            tr_full = np.trace(cov_np)
            if tr_full <= 0:
                return 0.0
            tr_kept = np.trace(P_np @ cov_np @ P_np.T)
            return float(1.0 - tr_kept / tr_full)

        bf_z = _blocked_frac(P_z_np, self.cov_z)
        bf_wz = _blocked_frac(P_wz_np, self.cov_wz)
        bf_h = _blocked_frac(P_h_np, self.cov_h_aug)
        bf_wy = _blocked_frac(P_wy_np, self.cov_wy)

        print(f"  [Duncker] Projections updated | tasks_completed={self.tasks_completed}")
        print(f"    P_z:  {self.P_z.shape}  |  P_wz: {self.P_wz.shape}")
        print(f"    P_h:  {self.P_h.shape}  |  P_wy: {self.P_wy.shape}")
        print(f"    W_aug: ({n_h}, {n_h + n_x + 1})  |  Z_old: ({n_h + n_x + 1}, {M})")
        print(f"    W_out_aug: ({n_y}, {n_h + 1})  |  H_old: ({n_h + 1}, {M})")
        print(f"    alpha={alpha:g} | blocked frac: P_z={bf_z:.3f} P_wz={bf_wz:.3f} "
              f"P_h={bf_h:.3f} P_wy={bf_wy:.3f}")
        print(f"    (blocked frac near 1.0 = strong protection; near 0.0 = projection ≈ identity, raise/lower alpha)")
