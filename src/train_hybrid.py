"""Treinamento do pipeline híbrido: U-Net (Limpador) + ShapeClassifier.

Lê dataset_hybrid/ (gerado por dataset_generator_hybrid.py), treina as duas
redes, guarda os melhores checkpoints pela validação e, ao final, avalia o
pipeline completo (limpeza -> classificação -> ajuste geométrico) medindo a
distância em pixels entre a forma gerada pela IA e o alvo ideal.

Uso:
  python src/train_hybrid.py
  python src/train_hybrid.py --epochs 30 --data dataset_hybrid
  python src/train_hybrid.py --eval-only
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hybrid_vectorizer import (
    SHAPE_NAMES,
    HybridShapeCorrector,
    ShapeClassifier,
    UNetExpert,
    to_network_input,
)

CLEANER_FILE = "cleaner_model.pt"
CLASSIFIER_FILE = "classifier_model.pt"


# ==============================================================================
# DADOS
# ==============================================================================

def read_labels(root: Path) -> list[tuple[str, int]]:
    labels_file = root / "labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"{labels_file} nao encontrado. Rode dataset_generator_hybrid.py antes.")
    samples = []
    for line in labels_file.read_text().splitlines():
        if line.strip():
            name, raw_label = line.strip().split(",")
            label = int(raw_label)
            if not 0 <= label < len(SHAPE_NAMES):
                raise ValueError(f"Label {label} fora do intervalo 0..{len(SHAPE_NAMES) - 1}: {name}")
            samples.append((name, label))
    return samples


def balanced_subset(
    samples: list[tuple[str, int]],
    limit: int,
    seed: int,
) -> list[tuple[str, int]]:
    if not limit or limit >= len(samples):
        return samples.copy()
    rng = random.Random(seed)
    groups = {label: [] for label in range(len(SHAPE_NAMES))}
    for sample in samples:
        groups[sample[1]].append(sample)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for label in range(len(SHAPE_NAMES)):
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop())
    rng.shuffle(selected)
    return selected


def stratified_split(
    samples: list[tuple[str, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    train, val = [], []
    for label in range(len(SHAPE_NAMES)):
        group = [sample for sample in samples if sample[1] == label]
        rng.shuffle(group)
        if not group:
            continue
        n_val = min(len(group), max(1, int(round(len(group) * val_split))))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    if not train:
        raise ValueError("O conjunto de treino ficou vazio; aumente --max-samples.")
    return train, val


def augment_sketch(suja: np.ndarray) -> np.ndarray:
    """Variações só na entrada, aproximando o domínio do canvas (traço binário e uniforme)."""
    r = random.random()
    if r < 0.3:
        suja = np.where(suja < random.uniform(110, 200), 0, 255).astype(np.uint8)
    elif r < 0.5:
        ink = (255.0 - suja) * random.uniform(0.6, 1.4)
        suja = (255.0 - np.clip(ink, 0, 255)).astype(np.uint8)
    if random.random() < 0.2:
        suja = cv2.erode(suja, np.ones((3, 3), np.uint8))
    return suja


class HybridDataset(Dataset):
    """Pares (suja, limpa) já normalizados como na inferência: traço=1, fundo=0, WORK_SIZE."""

    def __init__(self, root: Path, samples: list[tuple[str, int]], augment: bool) -> None:
        self.root = root
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, label = self.samples[idx]
        suja = cv2.imread(str(self.root / "sujas" / name), cv2.IMREAD_GRAYSCALE)
        limpa = cv2.imread(str(self.root / "limpas" / name), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            suja = augment_sketch(suja)

        x, y = to_network_input(suja), to_network_input(limpa)

        if self.augment:
            # Rotações de 90° e espelhamento não mudam a classe da forma.
            k = random.randint(0, 3)
            x, y = np.rot90(x, k), np.rot90(y, k)
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]

        x = torch.from_numpy(np.ascontiguousarray(x))[None]
        y = torch.from_numpy(np.ascontiguousarray(y))[None]
        return x, y, label


# ==============================================================================
# PERDAS E MÉTRICAS
# ==============================================================================

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice suave: compensa o desbalanceamento (~1% dos pixels são traço)."""
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def hard_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).float()
    tgt = (target > 0.5).float()
    inter = (pred * tgt).sum(dim=(1, 2, 3))
    return ((2 * inter + 1) / (pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1)).sum()


def chamfer_px(pred: np.ndarray, target: np.ndarray) -> float:
    """Distância média simétrica (px) entre os traços de duas imagens (fundo branco)."""
    a, b = pred < 128, target < 128
    if not a.any() or not b.any():
        return float("inf")
    dist_to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
    return 0.5 * float(dist_to_b[a].mean() + dist_to_a[b].mean())


# ==============================================================================
# TREINO
# ==============================================================================

def classifier_input(x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mistura por amostra: alvo limpo, rascunho e saída binarizada da U-Net.

    Na inferência o classificador recebe a máscara da U-Net; treinar com as três
    fontes o torna robusto a imperfeições da limpeza.
    """
    pred = (torch.sigmoid(logits.detach()) > 0.4).float()
    choice = torch.randint(0, 3, (x.shape[0], 1, 1, 1), device=x.device)
    return torch.where(choice == 0, y, torch.where(choice == 1, x, pred))


def run_training(args: argparse.Namespace, train_set: Dataset, val_set: Dataset, out_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    cleaner = UNetExpert().to(device)
    classifier = ShapeClassifier().to(device)

    steps = args.epochs * len(train_loader)
    opt_cleaner = torch.optim.AdamW(cleaner.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_classifier = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_cleaner = torch.optim.lr_scheduler.OneCycleLR(opt_cleaner, max_lr=args.lr, total_steps=steps)
    sched_classifier = torch.optim.lr_scheduler.OneCycleLR(opt_classifier, max_lr=args.lr, total_steps=steps)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_dice, best_acc = -1.0, -1.0

    print(f"Treinando em {device} | treino={len(train_set)} val={len(val_set)} | {args.epochs} epocas")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        cleaner.train()
        classifier.train()
        sum_clean = sum_cls = 0.0

        for x, y, label in train_loader:
            x, y, label = x.to(device), y.to(device), label.to(device)

            logits = cleaner(x)
            loss_clean = bce(logits, y) + dice_loss(logits, y)
            opt_cleaner.zero_grad(set_to_none=True)
            loss_clean.backward()
            opt_cleaner.step()
            sched_cleaner.step()

            loss_cls = ce(classifier(classifier_input(x, y, logits)), label)
            opt_classifier.zero_grad(set_to_none=True)
            loss_cls.backward()
            opt_classifier.step()
            sched_classifier.step()

            sum_clean += loss_clean.item()
            sum_cls += loss_cls.item()

        # Validação espelhando a inferência: classificador recebe a saída da U-Net.
        cleaner.eval()
        classifier.eval()
        dice_total, correct = 0.0, 0
        with torch.inference_mode():
            for x, y, label in val_loader:
                x, y, label = x.to(device), y.to(device), label.to(device)
                logits = cleaner(x)
                dice_total += hard_dice(logits, y).item()
                pred_mask = (torch.sigmoid(logits) > 0.4).float()
                correct += (classifier(pred_mask).argmax(dim=1) == label).sum().item()
        val_dice = dice_total / len(val_set)
        val_acc = correct / len(val_set)

        tags = []
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(cleaner.state_dict(), out_dir / CLEANER_FILE)
            tags.append("cleaner*")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), out_dir / CLASSIFIER_FILE)
            tags.append("classifier*")

        n = len(train_loader)
        print(
            f"Epoca [{epoch:02d}/{args.epochs}] {time.time() - start:5.0f}s | "
            f"loss cleaner {sum_clean / n:.4f} | loss classifier {sum_cls / n:.4f} | "
            f"val dice {val_dice:.3f} | val acc {val_acc:.3f} {' '.join(tags)}"
        )

    print(f"Melhores: dice={best_dice:.3f} acc={best_acc:.3f} -> {out_dir / CLEANER_FILE}, {out_dir / CLASSIFIER_FILE}")


# ==============================================================================
# AVALIAÇÃO PONTA A PONTA
# ==============================================================================

def evaluate_pipeline(root: Path, samples: list[tuple[str, int]], out_dir: Path) -> None:
    """Avalia fitting puro (A), Otsu (B) e pipeline completo (C)."""
    corrector = HybridShapeCorrector(out_dir / CLEANER_FILE, out_dir / CLASSIFIER_FILE,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
    metrics = {
        stage: {k: [] for k in range(len(SHAPE_NAMES))}
        for stage in ("A_ALVO", "B_OTSU", "C_REDE")
    }
    confusion = np.zeros((len(SHAPE_NAMES), len(SHAPE_NAMES)), dtype=int)
    preview_rows: list[np.ndarray] = []
    shown = {k: 0 for k in range(len(SHAPE_NAMES))}

    for name, label in samples:
        suja = cv2.imread(str(root / "sujas" / name), cv2.IMREAD_GRAYSCALE)
        limpa = cv2.imread(str(root / "limpas" / name), cv2.IMREAD_GRAYSCALE)
        fit_target = corrector.regularize(limpa, label)
        _, foreground = cv2.threshold(suja, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fit_otsu = corrector.regularize(255 - foreground, label)
        corrected, shape_name = corrector.process(suja)
        predicted = SHAPE_NAMES.index(shape_name)

        confusion[label, predicted] += 1
        metrics["A_ALVO"][label].append(chamfer_px(fit_target, limpa))
        metrics["B_OTSU"][label].append(chamfer_px(fit_otsu, limpa))
        metrics["C_REDE"][label].append(chamfer_px(corrected, limpa))

        if shown[label] < 2:
            shown[label] += 1
            separator = np.full((256, 4), 128, np.uint8)
            row = np.hstack([suja, separator, fit_target, separator, fit_otsu,
                             separator, corrected, separator, limpa])
            cv2.putText(row, f"real={SHAPE_NAMES[label]} ia={shape_name}", (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
            preview_rows.append(np.pad(row, 3, constant_values=200))

    total = confusion.sum()
    print(f"\nAvaliacao em {total} amostras de validacao")
    print(f"  Acuracia de classificacao: {np.trace(confusion) / total:.3f}")
    for stage, per_class in metrics.items():
        print(f"  Chamfer {stage} (px):")
        non_empty = [np.asarray(values) for values in per_class.values() if values]
        for k, values in per_class.items():
            if not values:
                continue
            v = np.asarray(values)
            print(f"    {SHAPE_NAMES[k]:<11} media={v.mean():.2f}  p90={np.percentile(v, 90):.2f}  "
                  f"<=1.5px: {100 * np.mean(v <= 1.5):5.1f}%")
        all_d = np.concatenate(non_empty)
        print(f"    {'GERAL':<11} media={all_d.mean():.2f}  p90={np.percentile(all_d, 90):.2f}  "
              f"<=1.5px: {100 * np.mean(all_d <= 1.5):5.1f}%")

    print("  Matriz de confusao (linhas=real, colunas=IA):")
    for k, row in enumerate(confusion):
        print(f"    {SHAPE_NAMES[k]:<11} {row}")

    preview_path = out_dir / "eval_preview.png"
    cv2.imwrite(str(preview_path), np.vstack(preview_rows))
    print(f"  Preview (suja | A | B | C | alvo): {preview_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino do pipeline artistico v2 (U-Net + classificador).")
    parser.add_argument("--data", default="dataset_artist_v2", help="Diretorio do dataset")
    parser.add_argument("--out-dir", default="models", help="Onde salvar os checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Limita o dataset (0 = todos), util para testes")
    parser.add_argument("--eval-only", action="store_true", help="So avalia os checkpoints existentes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root, out_dir = Path(args.data), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = balanced_subset(read_labels(root), args.max_samples, args.seed)
    train_samples, val_samples = stratified_split(samples, args.val_split, args.seed)

    if not args.eval_only:
        run_training(
            args,
            HybridDataset(root, train_samples, augment=True),
            HybridDataset(root, val_samples, augment=False),
            out_dir,
        )
    evaluate_pipeline(root, val_samples, out_dir)


if __name__ == "__main__":
    main()
