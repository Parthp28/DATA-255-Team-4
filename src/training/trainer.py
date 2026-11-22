"""Shared Trainer (Execution Plan 9.2, recipe D10).

Used UNCHANGED by both the ResNet (Kaushik) and DeiT-S (Khushi) tracks -- only
the model, train renderings, loss, and lambda differ. That sameness is what makes
the CNN-vs-ViT comparison fair (Section 5.3).

Recipe (D10, frozen): AdamW + cosine schedule, 1 warmup epoch, max 10 epochs,
patience 3 on val Macro F1 over OWN renderings, 128 images/step, fp16 AMP,
grad clip 1.0, no augmentation, ImageNet normalization.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..common.config import ExperimentConfig
from ..common.schema import PREDICTION_COLUMNS, CLASS_ORDER
from . import consistency_ref


@dataclass
class FitResult:
    best_val_f1: float
    best_epoch: int
    epochs_run: int
    checkpoint_path: str


def _model_input(batch: dict, key: str, device) -> torch.Tensor:
    return batch[key].to(device, non_blocking=True)


class Trainer:
    def __init__(self, model: nn.Module, config: ExperimentConfig, device: torch.device,
                 output_dir: str, consistency_fn=consistency_ref.combined_loss):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.output_dir = output_dir
        self.loss_fn = consistency_fn
        self.input_key = "seq" if config.is_numeric else "image"
        # AMP only helps on CUDA; on CPU we run fp32 so the stack is testable anywhere.
        self.use_amp = bool(config.amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ---- optimizer + schedule (D10) ----
    def _build_optim(self, steps_per_epoch: int):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        warmup_steps = max(1, steps_per_epoch)                       # 1 warmup epoch
        total_steps = max(warmup_steps + 1, self.cfg.epochs_max * steps_per_epoch)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[warmup_steps])
        return opt, sched

    def _batch_loss(self, batch: dict):
        labels = batch["label"].to(self.device)
        if self.cfg.pairing == "pair_distinct":
            a = _model_input(batch, "image_a", self.device)
            b = _model_input(batch, "image_b", self.device)
            logits_a, logits_b = self.model(a), self.model(b)
            return self.loss_fn(self.cfg.loss, self.cfg.consistency_lambda,
                                logits_a, labels, logits_b)
        x = _model_input(batch, self.input_key, self.device)
        return self.loss_fn(self.cfg.loss, self.cfg.consistency_lambda, self.model(x), labels)

    def fit(self, train_loader: DataLoader, val_eval_dataset, own_render_ids: list[str],
            checkpoint_path: str, max_steps: int | None = None) -> FitResult:
        opt, sched = self._build_optim(len(train_loader))
        best_f1, best_epoch, patience_left = -1.0, -1, self.cfg.patience
        step = 0
        for epoch in range(self.cfg.epochs_max):
            self.model.train()
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss, _ = self._batch_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                sched.step()
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
            # early stopping on val Macro F1 over own renderings
            f1 = self.evaluate_macro_f1(val_eval_dataset, own_render_ids)
            if f1 > best_f1:
                best_f1, best_epoch, patience_left = f1, epoch, self.cfg.patience
                self.save_checkpoint(checkpoint_path, epoch, best_f1)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
            if max_steps is not None and step >= max_steps:
                self.save_checkpoint(checkpoint_path, epoch, max(best_f1, f1))
                best_epoch = epoch if best_epoch < 0 else best_epoch
                break
        return FitResult(best_f1, best_epoch, epoch + 1, checkpoint_path)

    @torch.no_grad()
    def _infer(self, dataset, render_ids: list[str] | None):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=max(32, self.cfg.images_per_step),
                            shuffle=False)
        rows = []
        for batch in loader:
            key = "seq" if "seq" in batch else "image"
            x = batch[key].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x).float()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            rid = batch.get("render_id", None)
            for i in range(len(batch["label"])):
                r = rid[i] if rid is not None else "NUM"
                if render_ids is not None and r not in render_ids:
                    continue
                rows.append((batch["window_id"][i], r, int(batch["label"][i]),
                             logits[i].tolist(), probs[i].tolist(), int(preds[i])))
        return rows

    def evaluate_macro_f1(self, dataset, render_ids: list[str]) -> float:
        rows = self._infer(dataset, render_ids)
        if not rows:
            return 0.0
        y = [r[2] for r in rows]
        yhat = [r[5] for r in rows]
        return float(f1_score(y, yhat, average="macro",
                              labels=list(range(len(CLASS_ORDER))), zero_division=0))

    def predict(self, dataset, run_id: str, split: str, checkpoint: str) -> pd.DataFrame:
        rows = self._infer(dataset, render_ids=None)
        rec = []
        for wid, rid, label, logits, probs, pred in rows:
            rec.append([run_id, self.cfg.experiment_id, self.cfg.seed, split, wid, rid,
                        label, *logits, *probs, pred, checkpoint])
        return pd.DataFrame(rec, columns=PREDICTION_COLUMNS)

    def save_checkpoint(self, path: str, epoch: int, best_f1: float):
        torch.save({"model": self.model.state_dict(), "epoch": epoch,
                    "best_f1": best_f1, "config": self.cfg.to_dict()}, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        return ckpt
