"""
V-JEPA2 Trainer
================
Custom trainer for V-JEPA2 that implements the JEPA training loop:
forward target, forward context, loss, backward, optimizer step, EMA update.
"""

import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import tqdm.auto

import src.trainer.base as base
import src.trainer.stats as stats

logger = logging.getLogger(__name__)


class VJepa2Trainer(base.Trainer):
    """
    Trainer for V-JEPA2 that implements the full JEPA training loop.
    
    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader yielding (video_clips, masks_enc, masks_pred).
    encoder : torch.nn.Module
        The ViT encoder.
    predictor : torch.nn.Module
        The predictor network.
    target_encoder : torch.nn.Module
        EMA copy of the encoder.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : object
        Learning rate scheduler (step-based).
    wd_scheduler : object
        Weight decay scheduler.
    momentum_scheduler : iterator
        Iterator yielding EMA momentum values.
    device : torch.device
        Device for computation.
    stats : stats.TrainerStats
        Stats collector.
    loss_exp : float
        Exponent for the loss (1 = L1, 2 = L2).
    reg_coeff : float
        Regularization coefficient.
    clip_grad : float
        Max gradient norm for clipping.
    num_epochs : int
        Number of training epochs.
    mixed_precision : bool
        Whether to use mixed precision training.
    dtype : torch.dtype
        Data type for mixed precision.
    mask_collator : object
        The mask collator for generating masks.
    batch_size : int
        Batch size.
    num_clips : int
        Number of clips per sample.
    """

    def __init__(self,
                 loader: data.DataLoader,
                 encoder: nn.Module,
                 predictor: nn.Module,
                 target_encoder: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 wd_scheduler,
                 momentum_scheduler,
                 device: torch.device,
                 stats: stats.TrainerStats = stats.NOOPTrainerStats(),
                 loss_exp: float = 1.0,
                 reg_coeff: float = 0.0,
                 clip_grad: float = 10.0,
                 num_epochs: int = 2,
                 mixed_precision: bool = True,
                 dtype: torch.dtype = torch.bfloat16,
                 scaler=None,
                 batch_size: int = 2,
                 num_clips: int = 1,
                 max_runtime_seconds: Optional[float] = None,
                 max_steps: Optional[int] = None):
        # Pass encoder as the model to the base class
        super().__init__(model=encoder, loader=loader, device=device, stats=stats)
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.loss_exp = loss_exp
        self.reg_coeff = reg_coeff
        self.clip_grad = clip_grad
        self.num_epochs = num_epochs
        self.mixed_precision = mixed_precision
        self.dtype = dtype
        self.scaler = scaler
        self.batch_size = batch_size
        self.num_clips = num_clips
        self.max_runtime_seconds = (
            max_runtime_seconds if max_runtime_seconds and max_runtime_seconds > 0 else None
        )
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        self.global_step = 0
        self.stop_reason = "completed_epochs"

    def _should_stop(self, train_start_time: float) -> bool:
        if self.max_steps is not None and self.global_step >= self.max_steps:
            self.stop_reason = "max_steps"
            return True
        if self.max_runtime_seconds is not None:
            elapsed = time.perf_counter() - train_start_time
            if elapsed >= self.max_runtime_seconds:
                self.stop_reason = "max_runtime"
                return True
        return False

    def process_batch(self, i: int, batch: Any) -> Any:
        """Move batch data to device."""
        # Record data loading phase end / data transfer start
        if hasattr(self.stats, 'start_phase'):
            self.stats.start_phase("data_transfer")
        
        udata, masks_enc, masks_pred = batch
        
        clips = torch.cat(
            [u.to(self.device, non_blocking=True) for u in udata[0]], dim=0
        )
        
        from src.models.vjepa2.jepa.src.utils.tensors import repeat_interleave_batch
        _masks_enc, _masks_pred = [], []
        for _me, _mp in zip(masks_enc, masks_pred):
            _me = _me.to(self.device, non_blocking=True)
            _mp = _mp.to(self.device, non_blocking=True)
            _me = repeat_interleave_batch(_me, self.batch_size, repeat=self.num_clips)
            _mp = repeat_interleave_batch(_mp, self.batch_size, repeat=self.num_clips)
            _masks_enc.append(_me)
            _masks_pred.append(_mp)

        # Trim all masks to the common minimum token count per type
        # (different mask generators can produce different K values)
        if len(_masks_enc) > 1:
            min_keep_enc = min(m.shape[1] for m in _masks_enc)
            _masks_enc = [m[:, :min_keep_enc] for m in _masks_enc]
        if len(_masks_pred) > 1:
            min_keep_pred = min(m.shape[1] for m in _masks_pred)
            _masks_pred = [m[:, :min_keep_pred] for m in _masks_pred]
        
        if hasattr(self.stats, 'stop_phase'):
            self.stats.stop_phase("data_transfer")
        
        return clips, _masks_enc, _masks_pred

    def forward(self, i: int, batch: Any, model_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Forward pass: target encoder + context encoder + predictor + loss."""
        clips, masks_enc, masks_pred = batch
        
        from src.models.vjepa2.jepa.src.masks.utils import apply_masks
        
        # Forward target (no grad) — keep full h for predictor input
        with torch.no_grad():
            h = self.target_encoder(clips)
            h = F.layer_norm(h, (h.size(-1),))
            # Pre-masked targets (list) used only for loss computation
            h_masked = apply_masks(h, masks_pred, concat=False)
        
        # Forward context encoder (all masks concatenated internally)
        z = self.encoder(clips, masks_enc)
        
        # Predictor must be called per mask pair — its internal logic
        # assumes masks_ctxt and masks_tgt are aligned 1:1.
        B = clips.shape[0]
        z_chunks = z.split(B, dim=0)          # one chunk per encoder mask
        z_preds = []
        for idx, (z_i, mc, mt) in enumerate(zip(z_chunks, masks_enc, masks_pred)):
            z_pred_i = self.predictor(z_i, h, [mc], [mt], mask_index=idx)
            z_preds.append(z_pred_i)
        
        # JEPA loss
        loss = 0.0
        for zi, hi in zip(z_preds, h_masked):
            loss += torch.mean(torch.abs(zi - hi) ** self.loss_exp) / self.loss_exp
        loss /= len(masks_pred)
        
        # Regularization
        if self.reg_coeff > 0:
            pstd_z = sum(
                [torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z_preds]
            ) / len(z_preds)
            reg_loss = torch.mean(F.relu(1.0 - pstd_z))
            loss = loss + self.reg_coeff * reg_loss
        
        return loss

    def backward(self, i: int, loss: torch.Tensor) -> None:
        """Backward pass with optional mixed precision."""
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        # Gradient clipping
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_grad)

    def optimizer_step(self, i: int) -> None:
        """Optimizer step + EMA update with phase instrumentation."""
        if self.mixed_precision and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        self.scheduler.step()
        self.wd_scheduler.step()
        
        # EMA update — timed separately
        if hasattr(self.stats, 'start_phase'):
            self.stats.start_phase("ema_update")
        
        m = next(self.momentum_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
        
        if hasattr(self.stats, 'stop_phase'):
            self.stats.stop_phase("ema_update")

    def train(self, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Override the base train method to support multi-epoch training.
        V-JEPA trains for multiple epochs with the same loader.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        self.encoder.train()
        self.predictor.train()
        self.target_encoder.train()  # In eval conceptually, but .train() for BatchNorm etc.
        
        # Disable grad for target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.stats.start_train()
        train_start_time = time.perf_counter()
        completed_epochs = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            if hasattr(self.stats, "set_epoch"):
                self.stats.set_epoch(epoch + 1)
            loader_iter = iter(self.loader)
            pbar = tqdm.auto.tqdm(
                enumerate(loader_iter),
                total=len(self.loader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            for i, batch in pbar:
                self.stats.start_step()
                loss, info = self.step(i, batch, model_kwargs)
                self.stats.stop_step()
                self.stats.log_loss(loss)
                self.stats.log_step()
                self.global_step += 1
                
                if info:
                    pbar.write(info)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if self._should_stop(train_start_time):
                    logger.info(
                        "Stopping V-JEPA2 training after %d step(s) because %s was reached.",
                        self.global_step,
                        self.stop_reason,
                    )
                    break
            
            if self.stop_reason != "completed_epochs":
                break
            logger.info(f"Epoch {epoch+1} complete")
            completed_epochs = epoch + 1
        
        self.stats.stop_train()
        self.stats.log_stats()
        wall_time_s = time.perf_counter() - train_start_time
        logger.info(
            "TRAINING_SUMMARY global_steps=%d completed_epochs=%d stop_reason=%s wall_time_s=%.3f",
            self.global_step,
            completed_epochs,
            self.stop_reason,
            wall_time_s,
        )
