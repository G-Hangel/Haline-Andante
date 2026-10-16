"""
Gerador de dataset sintético para o pipeline híbrido de vetorização.

Produz pares (suja, limpa) + rótulo de forma para treinar:
  - U-Net (Limpador):   sujas/  -> limpas/
  - ShapeClassifier:    sujas/  -> label (0..5)

Classes: reta, retângulo, círculo/elipse, curva, triângulo e polilinha.
Triângulos e polilinhas podem ter pequenas interrupções de caneta somente na
entrada suja; o alvo permanece conectado e geometricamente perfeito.

Princípio central (correção do bug de função identidade):
  1. Cada amostra nasce de uma GEOMETRIA IDEAL (reta, retângulo, elipse, curva).
  2. limpas/ é renderizada direto dessa geometria ideal: traço fino (2 px),
     espessura constante, binário puro {0, 255}, sem anti-aliasing.
  3. sujas/ é uma trajetória DERIVADA e degradada da geometria ideal:
       - oscilação espacial suave (baixa frequência) via B-Spline (splprep);
       - tremor de alta frequência na trajetória;
       - variação de pressão (espessura e intensidade variáveis ao longo do traço);
       - microrruído nas bordas (raio do "carimbo" da caneta oscila por ponto);
       - fechamento imperfeito em formas fechadas (gap/overshoot);
       - grafite em tons de cinza, granulação e textura de papel.
  Um validador rejeita pares em que a suja fique parecida demais com a limpa.

Saída:
  dataset_artist_v2/
    sujas/sample_XXXXX.png   (256x256, escala de cinza)
    limpas/sample_XXXXX.png  (256x256, binária: fundo 255, traço 0)
    labels.txt               (sample_XXXXX.png,LABEL)
    preview.png              (grade suja | limpa para inspeção visual)

Uso:
  python dataset_generator_hybrid.py
  python dataset_generator_hybrid.py --samples-per-class 1000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

IMG_SIZE = 256
MARGIN = 24                 # folga para o traço sujo (oscilação + espessura) não sair da imagem
CLEAN_THICKNESS = 2         # espessura fixa do alvo da U-Net
SUPERSAMPLE = 4             # renderização do traço sujo em 4x para bordas suaves
SUBPIXEL_SHIFT = 4          # cv2 shift=4 -> precisão de 1/16 px
MAX_ATTEMPTS = 20           # tentativas por amostra antes de abortar

# Limite máximo de IoU entre o traço sujo binarizado e o limpo.
# IoU alto significa que a suja ficou quase igual à limpa (o bug que estamos corrigindo).
MAX_STROKE_IOU = 0.75
MIN_INK_PIXELS = 40

CLASS_NAMES = {
    0: "LINHA_RETA",
    1: "RETANGULO",
    2: "CIRCULO",
    3: "CURVA",
    4: "TRIANGULO",
    5: "POLILINHA",
}


# ==============================================================================
# UTILITÁRIOS GEOMÉTRICOS
# ==============================================================================

def rotate(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotaciona pontos (N, 2) em torno da origem."""
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]])


def fit_to_canvas(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Escala (só se necessário) e posiciona a forma aleatoriamente dentro das margens."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = hi - lo
    avail = IMG_SIZE - 2 * MARGIN
    scale = min(1.0, avail / max(float(extent.max()), 1e-6))
    pts = (pts - lo) * scale
    extent = extent * scale
    offset = MARGIN + rng.uniform(0.0, 1.0, size=2) * (avail - extent)
    return pts + offset


def resample_path(pts: np.ndarray, closed: bool, spacing: float = 1.0) -> np.ndarray:
    """
    Reamostra uma polilinha com espaçamento uniforme por comprimento de arco.
    Para caminhos fechados o último ponto NÃO repete o primeiro.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts, seg = pts[keep], seg[seg > 1e-9]

    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(np.ceil(total / spacing)), 8)
    s_new = np.linspace(0.0, total, n, endpoint=not closed)
    return np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])


def arc_length_param(pts: np.ndarray) -> np.ndarray:
    """Parâmetro u em [0, 1] proporcional ao comprimento de arco."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / max(s[-1], 1e-9)


# ==============================================================================
# GERADORES DE FORMAS IDEAIS (coordenadas locais; posicionadas depois)
# Cada gerador retorna (pontos, fechado).
# ==============================================================================

def shape_line(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 0: segmento de reta com inclinação aleatória."""
    half = rng.uniform(90, 230) / 2
    pts = np.array([[-half, 0.0], [half, 0.0]])
    return rotate(pts, rng.uniform(0, np.pi)), False


def shape_rectangle(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 1: retângulo em traço contínuo (às vezes levemente/fortemente rotacionado)."""
    w, h = rng.uniform(60, 200), rng.uniform(45, 200)
    if rng.random() < 0.6:
        angle = np.deg2rad(rng.uniform(-6, 6))
    else:
        angle = rng.uniform(-np.pi / 4, np.pi / 4)

    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # Canto inicial e sentido do traço aleatórios (onde a caneta começa/termina)
    corners = np.roll(corners, rng.integers(4), axis=0)
    if rng.random() < 0.5:
        corners = corners[::-1]
    return rotate(corners, angle), True


def shape_ellipse(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 2: círculo ou elipse."""
    if rng.random() < 0.5:
        a = b = rng.uniform(30, 100)
    else:
        a = rng.uniform(45, 105)
        b = a * rng.uniform(0.45, 0.85)

    direction = rng.choice([-1.0, 1.0])
    t = rng.uniform(0, 2 * np.pi) + direction * np.linspace(0, 2 * np.pi, 360, endpoint=False)
    pts = np.column_stack([a * np.cos(t), b * np.sin(t)])
    return rotate(pts, rng.uniform(0, np.pi)), True


def shape_curve(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 3: curva aberta (arco, S ou ondulação livre)."""
    kind = rng.choice(["arco", "s", "livre"])
    t = np.linspace(0.0, 1.0, 300)

    if kind == "arco":
        radius = rng.uniform(45, 130)
        theta = t * np.deg2rad(rng.uniform(70, 200))
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    elif kind == "s":
        length = rng.uniform(130, 220)
        amp = rng.uniform(18, 45)
        cycles = rng.uniform(0.8, 1.25)
        pts = np.column_stack([(t - 0.5) * length, amp * np.sin(2 * np.pi * cycles * t)])

    else:
        # Ondulação livre: pontos de controle com sinais alternados garantem curvatura real
        n_ctrl = int(rng.integers(4, 7))
        length = rng.uniform(140, 220)
        cx = np.linspace(-length / 2, length / 2, n_ctrl)
        signs = np.where(np.arange(n_ctrl) % 2 == 0, 1.0, -1.0) * rng.choice([-1.0, 1.0])
        cy = signs * rng.uniform(15, 50, size=n_ctrl)
        tck, _ = splprep([cx, cy], s=0, k=3)
        pts = np.column_stack(splev(t, tck))

    return rotate(pts, rng.uniform(0, 2 * np.pi)), False


def shape_triangle(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 4: triângulo escaleno, isósceles ou retângulo, sempre não degenerado."""
    kind = rng.choice(["escaleno", "isosceles", "retangulo"], p=[0.5, 0.3, 0.2])
    width = rng.uniform(70, 190)
    height = rng.uniform(65, 190)

    if kind == "isosceles":
        apex_x = 0.0
    elif kind == "retangulo":
        apex_x = -width / 2
    else:
        apex_x = rng.uniform(-0.32 * width, 0.32 * width)

    corners = np.array([
        [-width / 2, height / 2],
        [width / 2, height / 2],
        [apex_x, -height / 2],
    ])
    corners = np.roll(corners, rng.integers(3), axis=0)
    if rng.random() < 0.5:
        corners = corners[::-1]
    return rotate(corners, rng.uniform(-np.pi, np.pi)), True


def shape_polyline(rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Classe 5: caminho aberto formado por 2 a 5 segmentos retos bem distintos."""
    for _ in range(30):
        n_segments = int(rng.integers(2, 6))
        lengths = rng.uniform(35, 75, n_segments)
        heading = rng.uniform(-np.pi, np.pi)
        points = [np.zeros(2)]
        for length in lengths:
            heading += rng.choice([-1.0, 1.0]) * rng.uniform(np.deg2rad(35), np.deg2rad(115))
            points.append(points[-1] + length * np.array([np.cos(heading), np.sin(heading)]))
        pts = np.asarray(points)
        if np.linalg.norm(pts[-1] - pts[0]) > 35:
            return rotate(pts - pts.mean(axis=0), rng.uniform(-np.pi, np.pi)), False
    raise RuntimeError("Nao foi possivel gerar uma polilinha valida.")


SHAPE_GENERATORS = {
    0: shape_line,
    1: shape_rectangle,
    2: shape_ellipse,
    3: shape_curve,
    4: shape_triangle,
    5: shape_polyline,
}


# ==============================================================================
# RENDERIZAÇÃO DO ALVO LIMPO
# ==============================================================================

def render_clean(path: np.ndarray, closed: bool) -> np.ndarray:
    """Traço ideal: espessura constante, LINE_8 (sem anti-aliasing), binário {0, 255}."""
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    pts = np.round(path * (1 << SUBPIXEL_SHIFT)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], closed, 0, CLEAN_THICKNESS, lineType=cv2.LINE_8, shift=SUBPIXEL_SHIFT)
    return np.where(img < 128, 0, 255).astype(np.uint8)


# ==============================================================================
# DEGRADAÇÃO: GERAÇÃO DO RASCUNHO SUJO
# ==============================================================================

def imperfect_closure(path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Converte um caminho fechado em um traço aberto, como a mão faz:
    às vezes passa do ponto inicial (overshoot), às vezes não chega (gap).
    """
    n = len(path)
    k = int(round(rng.uniform(-0.03, 0.07) * n))
    if k >= 0:
        return np.vstack([path, path[: k + 1]])
    return path[: n + k]


def bspline_oscillation(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Oscilação espacial SUAVE (baixa frequência) via B-Spline.

    Sorteia deslocamentos 2D em poucos nós ao longo do comprimento de arco e
    interpola com splprep/splev. Como o deslocamento é um campo suave somado
    à trajetória (e não um refit da forma), os cantos do retângulo continuam
    existindo, apenas "entortados" como num desenho à mão.
    """
    u = arc_length_param(traj)
    length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    n_knots = int(np.clip(length / 45.0, 4, 14))   # ~1 nó a cada 45 px -> baixa frequência
    # Traços curtos oscilam menos, senão uma reta curta vira um "S" e confunde o classificador
    amp = rng.uniform(1.0, 3.0) * float(np.clip(length / 160.0, 0.5, 1.0))
    u_knots = np.linspace(0.0, 1.0, n_knots)
    disp = rng.normal(0.0, amp, size=(n_knots, 2))

    tck, _ = splprep([disp[:, 0], disp[:, 1]], u=u_knots, s=0, k=3)
    dx, dy = splev(u, tck)
    return traj + np.column_stack([dx, dy])


def hand_tremor(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Microtremor de alta frequência (correlação de ~2-3 px ao longo do traço)."""
    noise = rng.normal(0.0, rng.uniform(0.3, 0.8), size=traj.shape)
    kernel = np.array([0.25, 0.5, 0.25])
    noise[:, 0] = np.convolve(noise[:, 0], kernel, mode="same")
    noise[:, 1] = np.convolve(noise[:, 1], kernel, mode="same")
    return traj + noise


def pressure_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Pressão da caneta em [0.1, 1] ao longo do traço:
    base + soma de senoides lentas + afinamento na entrada e na saída do traço.
    """
    t = np.linspace(0.0, 1.0, n)
    p = np.full(n, rng.uniform(0.45, 0.8))
    for _ in range(3):
        p += rng.uniform(0.05, 0.18) * np.sin(
            2 * np.pi * rng.uniform(0.5, 4.0) * t + rng.uniform(0, 2 * np.pi)
        )
    taper_in = np.clip(t / rng.uniform(0.03, 0.10), 0.0, 1.0)
    taper_out = np.clip((1.0 - t) / rng.uniform(0.05, 0.15), 0.0, 1.0)
    p *= 0.35 + 0.65 * np.minimum(taper_in, taper_out)
    return np.clip(p, 0.1, 1.0)


def render_dirty(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Renderiza o rascunho carimbando discos ao longo da trajetória:
      - raio depende da pressão (variação de espessura ~1 a ~7 px);
      - raio recebe ruído por ponto (bordas serrilhadas / tremidas);
      - opacidade depende da pressão (grafite mais claro com pouca pressão).
    Desenho em supersampling 4x e redução com INTER_AREA (bordas em tons de cinza).
    """
    n = len(traj)
    p = pressure_profile(n, rng)

    r_min, r_max = 0.6, rng.uniform(2.2, 3.6)
    radius = r_min + p * (r_max - r_min) + rng.normal(0.0, 0.3, n)
    radius = np.clip(radius, 0.5, None)
    opacity = np.clip(0.5 + 0.5 * p + rng.normal(0.0, 0.04, n), 0.3, 1.0)

    scale = SUPERSAMPLE * (1 << SUBPIXEL_SHIFT)
    centers = np.round(traj * scale).astype(np.int64)
    radii = np.round(radius * scale).astype(np.int64)
    values = np.round(opacity * 255).astype(np.int64)

    ink_hr = np.zeros((IMG_SIZE * SUPERSAMPLE, IMG_SIZE * SUPERSAMPLE), dtype=np.uint8)
    for (cx, cy), r, v in zip(centers, radii, values):
        cv2.circle(ink_hr, (int(cx), int(cy)), int(r), int(v), -1, cv2.LINE_8, shift=SUBPIXEL_SHIFT)

    ink = cv2.resize(ink_hr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Granulação do grafite: falhas sutis dentro do traço
    grain = cv2.GaussianBlur(rng.uniform(0.7, 1.0, (IMG_SIZE, IMG_SIZE)).astype(np.float32), (3, 3), 0.6)
    ink *= grain

    # Papel: levemente fora do branco puro, com ruído fino
    paper = rng.uniform(238, 255) + rng.normal(0.0, 2.5, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    dirty = np.clip(paper, 0, 255) * (1.0 - ink)
    return np.clip(dirty, 0, 255).astype(np.uint8)


def pen_lift_gaps(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove trechos curtos para simular uma forma construída com mais de um gesto."""
    if len(traj) < 40 or rng.random() >= 0.65:
        return traj
    keep = np.ones(len(traj), dtype=bool)
    for _ in range(int(rng.integers(1, 3))):
        center = int(rng.integers(len(traj) // 8, 7 * len(traj) // 8))
        half = int(rng.integers(1, 4))
        keep[max(0, center - half):min(len(traj), center + half + 1)] = False
    return traj[keep]


def make_dirty(
    path: np.ndarray,
    closed: bool,
    rng: np.random.Generator,
    allow_pen_lifts: bool = False,
) -> np.ndarray:
    """Pipeline completo: geometria ideal -> rascunho manual imperfeito."""
    traj = imperfect_closure(path, rng) if closed else path.copy()
    traj = bspline_oscillation(traj, rng)
    traj = hand_tremor(traj, rng)
    if allow_pen_lifts:
        traj = pen_lift_gaps(traj, rng)
    return render_dirty(traj, rng)


# ==============================================================================
# VALIDAÇÃO DO PAR
# ==============================================================================

def stroke_iou(clean: np.ndarray, dirty: np.ndarray) -> float:
    c = clean < 128
    d = dirty < 160
    union = np.count_nonzero(c | d)
    return np.count_nonzero(c & d) / union if union else 1.0


def is_valid_pair(clean: np.ndarray, dirty: np.ndarray, iou: float) -> bool:
    """Garante alvo binário, ambos com tinta e suja visivelmente diferente da limpa."""
    if not np.isin(clean, (0, 255)).all():
        return False
    if np.count_nonzero(clean == 0) < MIN_INK_PIXELS or np.count_nonzero(dirty < 160) < MIN_INK_PIXELS:
        return False
    if np.array_equal(clean, dirty):
        return False
    return iou < MAX_STROKE_IOU


def generate_sample(label: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Gera um par (limpa, suja) válido para a classe informada."""
    for _ in range(MAX_ATTEMPTS):
        pts, closed = SHAPE_GENERATORS[label](rng)
        pts = fit_to_canvas(pts, rng)
        path = resample_path(pts, closed)

        clean = render_clean(path, closed)
        dirty = make_dirty(path, closed, rng, allow_pen_lifts=label in (4, 5))

        iou = stroke_iou(clean, dirty)
        if is_valid_pair(clean, dirty, iou):
            return clean, dirty, iou

    raise RuntimeError(f"Nao foi possivel gerar um par valido para a classe {CLASS_NAMES[label]}.")


# ==============================================================================
# PREVIEW
# ==============================================================================

def save_preview(pairs: list[tuple[np.ndarray, np.ndarray, int]], out_path: Path, cols: int = 4) -> None:
    """Salva uma grade com células [suja | limpa] para inspeção visual rápida."""
    sep = 4
    cells = []
    for dirty, clean, label in pairs:
        cell = np.full((IMG_SIZE + 20, IMG_SIZE * 2 + sep), 255, dtype=np.uint8)
        cell[20:, :IMG_SIZE] = dirty
        cell[20:, IMG_SIZE:IMG_SIZE + sep] = 128
        cell[20:, IMG_SIZE + sep:] = clean
        cv2.putText(cell, f"{label}:{CLASS_NAMES[label]}  suja | limpa", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
        cells.append(np.pad(cell, sep, constant_values=200))

    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def generate_dataset(output_dir: str, samples_per_class: int, seed: int, preview: int) -> None:
    rng = np.random.default_rng(seed)

    base = Path(output_dir)
    dirty_dir, clean_dir = base / "sujas", base / "limpas"
    dirty_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Remove amostras antigas para não misturar com pares gerados pelo gerador defeituoso
    stale = list(dirty_dir.glob("sample_*.png")) + list(clean_dir.glob("sample_*.png"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"Removidas {len(stale)} imagens antigas de {base}/")

    # Classes equilibradas, em ordem embaralhada
    labels = np.repeat(np.arange(len(CLASS_NAMES)), samples_per_class)
    rng.shuffle(labels)
    total = len(labels)

    print(f"Gerando {total} amostras ({samples_per_class} por classe) em {base.resolve()}")

    ious: dict[int, list[float]] = {k: [] for k in CLASS_NAMES}
    preview_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    preview_per_class = int(np.ceil(preview / len(CLASS_NAMES))) if preview > 0 else 0
    step = max(total // 10, 1)

    with open(base / "labels.txt", "w", newline="\n") as f:
        for idx, label in enumerate(labels):
            label = int(label)
            clean, dirty, iou = generate_sample(label, rng)

            fname = f"sample_{idx:05d}.png"
            cv2.imwrite(str(dirty_dir / fname), dirty)
            cv2.imwrite(str(clean_dir / fname), clean)
            f.write(f"{fname},{label}\n")

            ious[label].append(iou)
            if sum(1 for *_, lb in preview_pairs if lb == label) < preview_per_class:
                preview_pairs.append((dirty, clean, label))

            if (idx + 1) % step == 0 or idx + 1 == total:
                print(f"  [{idx + 1:>6}/{total}] {100 * (idx + 1) / total:5.1f}%")

    if preview_pairs:
        preview_pairs.sort(key=lambda item: item[2])
        save_preview(preview_pairs, base / "preview.png")

    print("\nResumo (IoU suja x limpa; quanto menor, mais diferentes):")
    for label, name in CLASS_NAMES.items():
        vals = ious[label]
        print(f"  {label} {name:<11} n={len(vals):>5}  IoU medio={np.mean(vals):.3f}  max={np.max(vals):.3f}")
    print(f"\nArquivos: {dirty_dir}/  {clean_dir}/  {base / 'labels.txt'}")
    if preview_pairs:
        print(f"Preview:  {base / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerador de dataset sintetico para o pipeline artistico v2.")
    parser.add_argument("--output", default="dataset_artist_v2", help="Diretorio de saida (padrao: dataset_artist_v2)")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Amostras por classe (padrao: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    parser.add_argument("--preview", type=int, default=16, help="Numero de pares no preview.png (0 desativa)")
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class deve ser >= 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.output, args.samples_per_class, args.seed, args.preview)
