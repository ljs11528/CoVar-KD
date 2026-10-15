"""Losses, pooled metrics, and exactly restored virtual optimizer updates."""
from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def resized_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0]:
        raise ValueError("Expected BCHW logits and BHW targets")
    return F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=True)


def ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore: int = -1) -> torch.Tensor:
    if not bool((target != ignore).any()):
        raise ValueError("No valid pixels")
    return F.cross_entropy(resized_logits(logits, target), target.long(), ignore_index=ignore)


def kd_loss(student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor,
            mode: str = "shared", temperature: float = 2.0, ignore: int = -1) -> torch.Tensor:
    if student.ndim != 4 or teacher.shape != student.shape:
        raise ValueError("Teacher and student must have identical BCHW shapes")
    if mode not in ("shared", "teacher") or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature placement")
    valid = target != ignore
    valid = F.interpolate(valid[:, None].float(), student.shape[-2:], mode="nearest")[:, 0].bool()
    if not bool(valid.any()):
        raise ValueError("No valid native-grid pixels")
    student_t = temperature if mode == "shared" else 1.0
    q = F.softmax(teacher.detach() / temperature, dim=1)
    per_pixel = F.kl_div(F.log_softmax(student / student_t, dim=1), q, reduction="none").sum(1)
    return per_pixel[valid].mean()


class PooledMetrics:
    """Pool sufficient statistics before averaging classes, never minibatch means."""

    def __init__(self, classes: int, ignore: int = -1):
        if classes < 2:
            raise ValueError("At least two classes required")
        self.classes, self.ignore = classes, ignore
        self.ce = torch.zeros(classes, dtype=torch.float64)
        self.gt = torch.zeros(classes, dtype=torch.int64)
        self.soft_intersection = torch.zeros(classes, dtype=torch.float64)
        self.soft_prediction = torch.zeros(classes, dtype=torch.float64)
        self.confusion = torch.zeros((classes, classes), dtype=torch.int64)

    @torch.no_grad()
    def add(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        z = resized_logits(logits, target)
        if z.shape[1] != self.classes:
            raise ValueError("Incorrect class count")
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite monitoring logits")
        valid = target != self.ignore
        y = target[valid].long()
        if y.numel() == 0:
            return
        if bool(((y < 0) | (y >= self.classes)).any()):
            raise ValueError("Invalid class index")
        # CPU float64 pooling makes batching affect only the forward pass, not
        # the reduction precision. No full prediction tensor is retained.
        logp = F.log_softmax(z, dim=1).permute(0, 2, 3, 1)[valid].double().cpu()
        y = y.cpu()
        probs = logp.exp()
        nll = -logp.gather(1, y[:, None])[:, 0]
        self.ce += torch.bincount(y, weights=nll, minlength=self.classes)
        self.gt += torch.bincount(y, minlength=self.classes)
        self.soft_intersection += torch.bincount(y, weights=probs.gather(1, y[:, None])[:, 0], minlength=self.classes)
        self.soft_prediction += probs.sum(0)
        predicted = logp.argmax(1)
        self.confusion += torch.bincount(self.classes * y + predicted, minlength=self.classes ** 2).reshape(self.classes, self.classes)

    def result(self) -> dict[str, Any]:
        total = int(self.gt.sum())
        if not total:
            raise ValueError("No valid monitoring pixels")
        present = self.gt > 0
        soft_union = self.gt.double() + self.soft_prediction - self.soft_intersection
        hard_i = self.confusion.diag().double()
        hard_u = self.confusion.sum(0) + self.confusion.sum(1) - self.confusion.diag()
        hard_present = hard_u > 0
        soft = self.soft_intersection / soft_union.clamp_min(1e-30)
        hard = hard_i / hard_u.clamp_min(1)
        return {
            "pixel_ce": float(self.ce.sum() / total),
            "class_ce": float((self.ce[present] / self.gt[present]).mean()),
            "soft_miou": float(soft[present].mean()),
            "hard_miou": float(hard[hard_present].mean()),
            "class_ce_values": [float(self.ce[k] / self.gt[k]) if present[k] else None for k in range(self.classes)],
            "soft_iou": [float(soft[k]) if present[k] else None for k in range(self.classes)],
            "hard_iou": [float(hard[k]) if hard_present[k] else None for k in range(self.classes)],
            "valid_pixels": total,
            "gt_counts": self.gt.tolist(),
            "confusion": self.confusion.tolist(),
            "soft_intersection": self.soft_intersection.tolist(),
            "soft_prediction": self.soft_prediction.tolist(),
            "all_classes_present": bool(present.all()),
        }


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def tree_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng(seed: int):
    old = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(old)


@dataclass
class Snapshot:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    modes: dict[str, bool]
    rng: dict[str, Any]
    gradients: dict[str, torch.Tensor | None]

    @classmethod
    def capture(cls, model, optimizer):
        return cls(_clone_cpu(model.state_dict()), _clone_cpu(optimizer.state_dict()),
                   {name: child.training for name, child in model.named_modules()}, rng_state(),
                   {name: None if p.grad is None else _clone_cpu(p.grad) for name, p in model.named_parameters()})

    def restore(self, model, optimizer):
        model.load_state_dict(self.model, strict=True)
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        # Some optimizer parameters belong to inactive feature criteria. SGD
        # state loading casts its per-parameter tensors to the parameter device.
        for name, child in model.named_modules():
            child.training = self.modes[name]
        for name, p in model.named_parameters():
            saved = self.gradients[name]
            p.grad = None if saved is None else saved.to(device=p.device, dtype=p.dtype).clone()
        restore_rng(self.rng)

    def verify(self, model, optimizer):
        current = Snapshot.capture(model, optimizer)
        if not all(tree_equal(getattr(self, key), getattr(current, key))
                   for key in ("model", "optimizer", "modes", "rng", "gradients")):
            raise RuntimeError("Virtual update did not restore the complete state")


@contextmanager
def restored_state(model, optimizer):
    snapshot = Snapshot.capture(model, optimizer)
    try:
        yield snapshot
    finally:
        snapshot.restore(model, optimizer)
        snapshot.verify(model, optimizer)


@contextmanager
def virtual_outcome(model, optimizer):
    """Record numerical divergence while still restoring every state component."""
    outcome = {"status": "PASS", "reason": None}
    with restored_state(model, optimizer):
        try:
            yield outcome
        except FloatingPointError as error:
            outcome.update(status="DIVERGED", reason=str(error))


@contextmanager
def evaluation_mode(model):
    modes = {name: child.training for name, child in model.named_modules()}
    try:
        model.eval()
        yield
    finally:
        for name, child in model.named_modules():
            child.training = modes[name]


def gradient_norm(grads: Iterable[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))


def choose_coefficient(scores: dict[float, float], metric: str, previous: float = 1.0) -> float:
    if metric not in ("pixel_ce", "soft_miou") or not scores or any(not math.isfinite(x) for x in scores.values()):
        raise ValueError("Invalid selection scores")
    best = min(scores.values()) if metric == "pixel_ce" else max(scores.values())
    tolerance = 1e-6 * max(1.0, abs(best))
    tied = [c for c, score in scores.items() if abs(score - best) <= tolerance]
    return previous if previous in tied else min(tied)
