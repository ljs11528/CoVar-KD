import os
import argparse
import math
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

from PCOS import (
    get_max_confidence_and_residual_variance,
)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.down1(x)
        e2 = self.down2(self.pool(e1))
        e3 = self.down3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


class RandomSegDataset(Dataset):
    def __init__(self, length: int, num_classes: int, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        h, w = self.image_size
        img = torch.randn(3, h, w)
        mask = torch.randint(0, self.num_classes, (h, w), dtype=torch.long)
        return img, mask


@torch.no_grad()
def compute_covar_weights(
    teacher_prob: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute CoVar weight w(x) = exp(-alpha * g_j * residual_variance).

    g_j * residual_variance is exactly the scaled_residual_variance from get_max_confidence_and_residual_variance.
    """
    max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(
        teacher_prob, valid_mask, num_classes, epsilon
    )
    weight = torch.exp(-alpha * scaled_residual_variance)
    weight = torch.where(valid_mask, weight, torch.zeros_like(weight))
    return weight


def supervised_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return criterion(logits, target)


def distill_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    ignore_mask: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Temperature-softened KL with CoVar weights per pixel."""
    # teacher/student probs at temperature
    t_prob = torch.softmax(teacher_logits / temperature, dim=1)
    s_log_prob = torch.log_softmax(student_logits / temperature, dim=1)
    # per-pixel KL = sum_k p_T * (log p_T - log p_S)
    kl = (t_prob * (torch.log(t_prob + 1e-8) - s_log_prob)).sum(dim=1)  # [N,H,W]
    valid_mask = (ignore_mask != ignore_index)
    loss = (kl * weights * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    return (temperature ** 2) * loss


@torch.no_grad()
def _fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    mask = (target != ignore_index)
    pred = pred[mask].view(-1)
    target = target[mask].view(-1)
    k = (target >= 0) & (target < num_classes)
    idx = target[k] * num_classes + pred[k]
    hist = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


def train_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    sup_weight: float,
    distill_weight: float,
    use_covar: bool,
    covar_alpha: float,
    temperature: float,
):
    teacher.eval()
    student.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        ignore_mask = masks

        # Normalize inputs for SegFormer teacher (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        pixel_values = (imgs - mean) / std

        with torch.no_grad():
            # SegFormer returns logits possibly at lower resolution; upsample to match student/target size
            t_out = teacher(pixel_values=pixel_values)
            t_logits = t_out.logits
            if t_logits.shape[-2:] != imgs.shape[-2:]:
                t_logits = torch.nn.functional.interpolate(
                    t_logits, size=imgs.shape[-2:], mode="bilinear", align_corners=False
                )
            t_prob = torch.softmax(t_logits / temperature, dim=1)
            valid_mask = (ignore_mask != 255)
            if use_covar:
                weights = compute_covar_weights(t_prob, valid_mask, num_classes, alpha=covar_alpha)
            else:
                weights = torch.where(valid_mask, torch.ones_like(ignore_mask, dtype=torch.float), torch.zeros_like(ignore_mask, dtype=torch.float))

        s_logits = student(imgs)
        loss_sup = supervised_loss(s_logits, masks)
        loss_dis = distill_kd_loss(s_logits, t_logits, weights, ignore_mask, temperature=temperature)
        loss = sup_weight * loss_sup + distill_weight * loss_dis

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> float:
    model.eval()
    correct = 0
    total = 0
    hist = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            valid = (masks != 255)
            correct += (preds[valid] == masks[valid]).sum().item()
            total += valid.sum().item()
            hist += _fast_hist(preds, masks, num_classes=num_classes)
    iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist)).clamp_min(1)
    miou = iou.mean().item()
    return correct / max(total, 1), miou


def build_dataloaders(batch_size: int, num_classes: int, length: int = 64):
    # Placeholder random dataset; replace with real dataset (e.g., Cityscapes) for meaningful training/eval.
    train_set = RandomSegDataset(length=length, num_classes=num_classes)
    val_set = RandomSegDataset(length=length // 4, num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def maybe_load_checkpoint(model: nn.Module, ckpt: Optional[str]) -> None:
    if ckpt is None:
        return
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)


def build_teacher(model_id: str, device: torch.device, num_classes: int) -> nn.Module:
    teacher = SegformerForSemanticSegmentation.from_pretrained(model_id)
    if getattr(teacher.config, "num_labels", num_classes) != num_classes:
        # If mismatch, warn but continue; downstream interpolation still works but labels may differ
        print(
            f"[warn] teacher num_labels={teacher.config.num_labels} mismatches --num-classes={num_classes}; "
            "consider aligning or fine-tuning the head."
        )
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def main():
    parser = argparse.ArgumentParser(description="Covar-guided distillation for semantic segmentation")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature T")
    parser.add_argument("--covar-alpha", type=float, default=1.0, help="Scaling for CoVar weight exp(-alpha * g_j * var)")
    parser.add_argument("--no-covar", action="store_true", help="Disable covar weighting (baseline distillation)")
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        help="Hugging Face model id for the SegFormer teacher",
    )
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--student-ckpt", type=str, default=None)
    parser.add_argument("--save", type=str, default="student_final.pth")
    parser.add_argument("--demo-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_teacher(args.teacher_model_id, device, args.num_classes)
    student = SimpleUNet(in_channels=3, num_classes=args.num_classes).to(device)

    maybe_load_checkpoint(teacher, args.teacher_ckpt)
    maybe_load_checkpoint(student, args.student_ckpt)

    # Freeze teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False

    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size, num_classes=args.num_classes, length=args.demo_length
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            teacher,
            student,
            train_loader,
            optimizer,
            device,
            num_classes=args.num_classes,
            sup_weight=args.sup_weight,
            distill_weight=args.distill_weight,
            use_covar=not args.no_covar,
            covar_alpha=args.covar_alpha,
            temperature=args.temperature,
        )
        acc, miou = evaluate(student, val_loader, device, args.num_classes)
        print(
            f"Epoch {epoch+1}/{args.epochs} - loss: {loss:.4f} - val pixel acc: {acc:.4f} - val mIoU: {miou:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), args.save)
            print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
