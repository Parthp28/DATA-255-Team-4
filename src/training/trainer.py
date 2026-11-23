"""Shared Trainer (Execution Plan 9.2, recipe D10).

Used UNCHANGED by both the ResNet (Kaushik) and DeiT-S (Khushi) tracks -- only
the model, train renderings, loss, and lambda differ. That sameness is what makes
the CNN-vs-ViT comparison fair (Section 5.3).

Recipe (D10, frozen): AdamW + cosine schedule, 1 warmup epoch, max 10 epochs,
patience 3 on val Macro F1 over OWN renderings, 128 images/step, fp16 AMP,
grad clip 1.0, no augmentation, ImageNet normalization.
"""
from __future__ import annotations

import os
import random
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
from . import consistency_ref  # noqa: F401  (Kaushik's reference loss, kept for the T-C cross-review)
from .losses import paired_consistency_loss


@dataclass
class FitResult:
    best_val_f1: float
    best_epoch: int
    epochs_run: int
    checkpoint_path: str


def _model_input(batch: dict, key: str, device) -> torch.Tensor:
    return batch[key].to(device, non_blocking=True)


def canonical_loss(mode: str, lam: float, logits_a, labels, logits_b=None):
    """The trainer's three loss modes on Khushi's canonical D12 loss (src/training/losses.py).

    "ce" -> E1/E2/E5 cross-entropy; "paired_ce" -> 0.5[CE(a)+CE(b)], lambda 0 (E3b/E4b);
    "ce_plus_js" -> + lambda * JS(p_a, p_b) (E3/E4). Same (total, parts) contract as consistency_ref."""
    if mode == "ce":
        ce = nn.functional.cross_entropy(logits_a.float(), labels)
        return ce, {"ce": float(ce.detach())}
    if mode not in ("paired_ce", "ce_plus_js"):
        raise ValueError(f"unknown loss mode {mode!r}")
    out = paired_consistency_loss(logits_a, logits_b, labels, lam=float(lam) if mode == "ce_plus_js" else 0.0)
    return out.total, {"ce": float(out.ce.detach()), "js": float(out.js.detach())}


class Trainer:
    def __init__(self, model: nn.Module, config: ExperimentConfig, device: torch.device,
                 output_dir: str, consistency_fn=canonical_loss):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.output_dir = output_dir
        self.loss_fn = consistency_fn
        self.input_key = "seq" if config.is_numeric else "image"
        # AMP only helps on CUDA; on CPU we run fp32 so the stack is testable anywhere.
        self.use_amp = bool(config.amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[dict] = []        # one entry per epoch (train loss, val F1, lr)
        self.step_losses: list[float] = []   # training losses of the steps run by this fit() call

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
            # One forward over all 2 x 64 = 128 images (plan 5.1: same images per step as E1/E2), so
            # BatchNorm sees the same 128-image batch statistics and one running-stat update per step
            # as E1. Two separate 64-image forwards would change both for ResNet (LayerNorm DeiT is
            # unaffected either way).
            logits_a, logits_b = self.model(torch.cat([a, b], dim=0)).split([a.shape[0], b.shape[0]], dim=0)
            return self.loss_fn(self.cfg.loss, self.cfg.consistency_lambda,
                                logits_a, labels, logits_b)
        x = _model_input(batch, self.input_key, self.device)
        return self.loss_fn(self.cfg.loss, self.cfg.consistency_lambda, self.model(x), labels)

    # ---- checkpoints: best.pt (selection) and last.pt (resume, plan 14.5 / T-T6) ----
    def _rng_state(self) -> dict:
        st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
        if self.device.type == "cuda":
            st["cuda"] = torch.cuda.get_rng_state_all()
        return st

    def _set_rng_state(self, st: dict) -> None:
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])
        if "cuda" in st and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(st["cuda"])

    def save_last(self, path: str, opt, sched, state: dict) -> None:
        """Full training state after a finished epoch, written atomically."""
        tmp = path + ".tmp"
        torch.save({"model": self.model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": self.scaler.state_dict(), "rng": self._rng_state(), "history": list(self.history),
                    "config": self.cfg.to_dict(), **state}, tmp)
        os.replace(tmp, path)

    def load_last(self, path: str, opt, sched) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("config") != self.cfg.to_dict():
            raise ValueError(f"{path} was written by a different configuration; refusing to resume")
        self.model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self._set_rng_state(ckpt["rng"])
        self.history = list(ckpt.get("history", []))
        return {k: ckpt[k] for k in ("epoch", "step", "best_f1", "best_epoch", "patience_left")}

    def fit(self, train_loader: DataLoader, val_eval_dataset, own_render_ids: list[str],
            checkpoint_path: str, max_steps: int | None = None, resume: bool = True,
            stop_after_epochs: int | None = None) -> FitResult:
        """Train with early stopping. last.pt (model, optimizer, scheduler, scaler, RNG, counters) is written
        after every epoch next to checkpoint_path; with resume=True an existing last.pt is continued exactly.
        stop_after_epochs ends the call after that many epochs (simulated session death for smoke/tests)."""
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        last_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "last.pt")
        opt, sched = self._build_optim(len(train_loader))
        st = {"epoch": 0, "step": 0, "best_f1": -1.0, "best_epoch": -1, "patience_left": self.cfg.patience}
        if resume and os.path.exists(last_path):
            st = self.load_last(last_path, opt, sched)
        best_f1, best_epoch, patience_left, step = st["best_f1"], st["best_epoch"], st["patience_left"], st["step"]
        start = epochs_done = st["epoch"]
        self.step_losses = []
        for epoch in range(start, self.cfg.epochs_max):
            if patience_left <= 0 or (max_steps is not None and step >= max_steps):
                break
            self.model.train()
            losses = []
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss, _ = self._batch_loss(batch)
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}, step {step}")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                sched.step()
                step += 1
                losses.append(float(loss.detach()))
                if max_steps is not None and step >= max_steps:
                    break
            self.step_losses += losses
            # early stopping on val Macro F1 averaged over own renderings (plan 5.1)
            f1 = self.evaluate_macro_f1(val_eval_dataset, own_render_ids)
            if f1 > best_f1:
                best_f1, best_epoch, patience_left = f1, epoch, self.cfg.patience
                self.save_checkpoint(checkpoint_path, epoch, best_f1)
            else:
                patience_left -= 1
            if max_steps is not None and step >= max_steps and best_epoch < 0:
                self.save_checkpoint(checkpoint_path, epoch, max(best_f1, f1))
                best_epoch = epoch
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
                                 "val_f1": f1, "lr": float(sched.get_last_lr()[0])})
            epochs_done = epoch + 1
            self.save_last(last_path, opt, sched, {"epoch": epochs_done, "step": step, "best_f1": best_f1,
                                                   "best_epoch": best_epoch, "patience_left": patience_left})
            if stop_after_epochs is not None and epochs_done - start >= stop_after_epochs:
                break
        return FitResult(best_f1, best_epoch, epochs_done, checkpoint_path)

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
        """Validation Macro F1 averaged over the experiment's own renderings (plan 5.1: one Macro F1 per
        rendering, then the mean); identical to the pooled value when there is a single rendering."""
        rows = self._infer(dataset, render_ids)
        if not rows:
            return 0.0
        per_render: dict[str, tuple[list, list]] = {}
        for r in rows:
            y, yhat = per_render.setdefault(r[1], ([], []))
            y.append(r[2])
            yhat.append(r[5])
        labels = list(range(len(CLASS_ORDER)))
        return float(np.mean([f1_score(y, yhat, average="macro", labels=labels, zero_division=0)
                              for y, yhat in per_render.values()]))

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
