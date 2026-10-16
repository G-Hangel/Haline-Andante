import argparse
import json
import sys
import numpy as np
from pathlib import Path

import torch

# O console do Windows usa cp1252 por padrão e aborta com UnicodeEncodeError ao
# imprimir acentos e símbolos; reconfigurar evita descascar o texto do relatório.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from hybrid_vectorizer import (
        HybridShapeCorrector, SHAPE_NAMES, ShapeClassifier, UNetExpert,
        VectorPrimitive, WORK_SIZE, snap_primitive,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback para execução como pacote
    from src.hybrid_vectorizer import (
        HybridShapeCorrector, SHAPE_NAMES, ShapeClassifier, UNetExpert,
        VectorPrimitive, WORK_SIZE, snap_primitive,
    )


def load_model(
    cleaner_checkpoint: str | None = None,
    classifier_checkpoint: str | None = None,
) -> HybridShapeCorrector:
    """Carrega o pipeline híbrido, opcionalmente com checkpoints treinados."""
    return HybridShapeCorrector(
        cleaner_checkpoint=cleaner_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
    )


def layer_parameter_stats(model: torch.nn.Module):
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        count = sum(param.numel() for param in params)
        rows.append((name, count, sum(param.numel() for param in params if param.requires_grad)))
    return rows


def summarize_network(name: str, model: torch.nn.Module) -> dict:
    """Resumo de parâmetros de uma sub-rede individual."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": name,
        "class_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "layers": [
            {
                "name": layer_name,
                "parameters": int(parameter_count),
                "trainable": int(trainable_count),
            }
            for layer_name, parameter_count, trainable_count in layer_parameter_stats(model)
        ],
    }


def summarize_pipeline(corrector: HybridShapeCorrector) -> dict:
    """Resumo completo do pipeline híbrido (cleaner + classifier)."""
    networks = []
    networks.append(summarize_network("cleaner (UNetExpert)", corrector.cleaner))
    if corrector.classifier is not None:
        networks.append(summarize_network("classifier (ShapeClassifier)", corrector.classifier))

    total = sum(n["total_parameters"] for n in networks)
    trainable = sum(n["trainable_parameters"] for n in networks)

    return {
        "pipeline": "HybridShapeCorrector",
        "stages": ["clean_stroke (UNetExpert)", "classify_shape (ShapeClassifier/heurística)", "regularize (geométrico)"],
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "cleaner_has_checkpoint": HybridShapeCorrector._has_checkpoint(corrector.cleaner),
        "classifier_has_checkpoint": corrector.classifier is not None and HybridShapeCorrector._has_checkpoint(corrector.classifier),
        "networks": networks,
    }


def analyze_forward(corrector: HybridShapeCorrector, input_shape: tuple[int, ...]) -> dict:
    """Executa o forward pass de diagnóstico em cada estágio do pipeline."""
    # Gera imagem sintética (retângulo preto sobre fundo branco)
    h, w = input_shape[-2], input_shape[-1]
    import cv2
    sketch = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(sketch, (w // 5, h // 5), (4 * w // 5, 4 * h // 5), 0, 4)

    result = {"input_shape": [h, w]}

    # Estágio 1: Limpeza
    clean = corrector.clean_stroke(sketch)
    result["clean_output_shape"] = list(clean.shape)
    result["clean_min"] = int(clean.min())
    result["clean_max"] = int(clean.max())
    result["clean_stroke_pixels"] = int(np.count_nonzero(clean < 128))

    # Estágio 2: Classificação
    shape_id = corrector.classify_shape(clean)
    result["shape_id"] = shape_id
    result["shape_name"] = SHAPE_NAMES[shape_id]
    result["classifier_mode"] = "CNN" if corrector.classifier is not None else "heurística geométrica"

    # Classificação detalhada (logits se CNN disponível)
    if corrector.classifier is not None:
        from hybrid_vectorizer import to_network_input
        tensor = torch.from_numpy(to_network_input(clean))[None, None].to(corrector.device)
        with torch.inference_mode():
            logits = corrector.classifier(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        result["classifier_logits"] = [round(float(v), 4) for v in logits[0]]
        result["classifier_probs"] = {
            name: round(float(p), 4) for name, p in zip(SHAPE_NAMES, probs)
        }

    # Estágio 3: Regularização
    primitive = corrector.vectorize(clean, shape_id)
    regularized = corrector.regularize(clean, shape_id)
    result["regularized_output_shape"] = list(regularized.shape)
    result["regularized_stroke_pixels"] = int(np.count_nonzero(regularized < 128))
    result["primitive_points"] = 0 if primitive is None else len(primitive.points)
    result["primitive_closed"] = False if primitive is None else primitive.closed

    existing = VectorPrimitive(0, np.array([[40.0, 40.0], [100.0, 40.0]]), False)
    incoming = VectorPrimitive(0, np.array([[105.0, 45.0], [180.0, 100.0]]), False)
    snapped = snap_primitive(incoming, [existing])
    result["snap_endpoint_distance"] = round(float(np.linalg.norm(snapped.points[0] - existing.points[1])), 4)

    return result


def print_report(summary: dict) -> None:
    print("=== Analisador de Modelo — Pipeline Híbrido ===")
    print(f"Pipeline: {summary['pipeline']}")
    print(f"Estágios: {' → '.join(summary['stages'])}")
    print(f"Parâmetros totais: {summary['total_parameters']:,}")
    print(f"Parâmetros treináveis: {summary['trainable_parameters']:,}")
    print(f"Parâmetros congelados: {summary['frozen_parameters']:,}")
    print(f"Cleaner com checkpoint: {'Sim' if summary['cleaner_has_checkpoint'] else 'Não'}")
    print(f"Classifier com checkpoint: {'Sim' if summary['classifier_has_checkpoint'] else 'Não'}")
    print()
    for net in summary["networks"]:
        print(f"--- {net['name']} ({net['class_name']}) ---")
        print(f"  Parâmetros: {net['total_parameters']:,} ({net['trainable_parameters']:,} treináveis)")
        for layer in net["layers"]:
            print(f"    {layer['name']}: {layer['parameters']:,} params ({layer['trainable']:,} treináveis)")
    print()


def print_forward_report(forward: dict) -> None:
    print("=== Forward Pass de Diagnóstico ===")
    print(f"Input: {forward['input_shape']}")
    print()
    print(f"Estágio 1 — Limpeza:")
    print(f"  Output shape: {forward['clean_output_shape']}")
    print(f"  Pixels de traço: {forward['clean_stroke_pixels']}")
    print(f"  Intensidade: min={forward['clean_min']}, max={forward['clean_max']}")
    print()
    print(f"Estágio 2 — Classificação ({forward['classifier_mode']}):")
    print(f"  Forma detectada: {forward['shape_name']} (id={forward['shape_id']})")
    if "classifier_probs" in forward:
        print(f"  Logits: {forward['classifier_logits']}")
        print(f"  Probabilidades: {forward['classifier_probs']}")
    print()
    print(f"Estágio 3 — Regularização:")
    print(f"  Output shape: {forward['regularized_output_shape']}")
    print(f"  Pixels de traço: {forward['regularized_stroke_pixels']}")
    print(f"  Primitiva: {forward['primitive_points']} pontos, fechada={forward['primitive_closed']}")
    print(f"  Teste de snapping: distância final={forward['snap_endpoint_distance']} px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa a arquitetura e o pipeline do HybridShapeCorrector."
    )
    parser.add_argument("--cleaner-checkpoint", type=str, default=None,
                        help="Caminho do checkpoint .pt do cleaner (UNetExpert).")
    parser.add_argument("--classifier-checkpoint", type=str, default=None,
                        help="Caminho do checkpoint .pt do classifier (ShapeClassifier).")
    parser.add_argument("--input-shape", type=int, nargs=2, default=[256, 256],
                        help="Formato da entrada H W (default: 256 256).")
    parser.add_argument("--json", action="store_true",
                        help="Emite o relatório em JSON em vez de texto legível.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaner_ckpt = args.cleaner_checkpoint
    classifier_ckpt = args.classifier_checkpoint
    project_dir = Path(__file__).resolve().parent.parent

    if cleaner_ckpt is None:
        for candidate in (project_dir / "models" / "cleaner_model.pt", project_dir / "cleaner_model.pt", project_dir / "model_artist_v2" / "cleaner_model.pt"):
            if candidate.exists():
                cleaner_ckpt = str(candidate)
                break

    if classifier_ckpt is None:
        for candidate in (project_dir / "models" / "classifier_model.pt", project_dir / "classifier_model.pt", project_dir / "model_artist_v2" / "classifier_model.pt"):
            if candidate.exists():
                classifier_ckpt = str(candidate)
                break

    corrector = load_model(cleaner_ckpt, classifier_ckpt)
    summary = summarize_pipeline(corrector)

    if args.json:
        forward = analyze_forward(corrector, tuple(args.input_shape))
        payload = {**summary, "forward": forward}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print_report(summary)
    forward = analyze_forward(corrector, tuple(args.input_shape))
    print_forward_report(forward)


if __name__ == "__main__":
    main()
