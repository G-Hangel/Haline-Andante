"""Pipeline híbrido de vetorização em três estágios.

1. UNetExpert (Limpador): rascunho ruidoso -> traço ideal fino, já endireitado.
2. ShapeClassifier: identifica reta, retângulo, círculo, curva, triângulo ou polilinha.
3. Regularização geométrica: produz uma primitiva vetorial perfeita na posição original.
4. Snapping contextual: conecta vértices e endpoints a formas já existentes.

As redes trabalham internamente em WORK_SIZE (mais leve em CPU); a geometria
final é ajustada e desenhada em IMAGE_SIZE com precisão subpixel.
Sem checkpoints, o pipeline continua utilizável via fallbacks OpenCV.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import splev, splprep

IMAGE_SIZE = (256, 256)
WORK_SIZE = 128
SHAPE_NAMES = ("LINHA_RETA", "RETANGULO", "CIRCULO", "CURVA", "TRIANGULO", "POLILINHA")

OUTPUT_THICKNESS = 2
MASK_THRESHOLD = 0.4
MIN_STROKE_PIXELS = 10
MAX_STROKE_FRACTION = 0.25
SNAP_DEGREES = 3.0
SNAP_RADIUS = 8.0
CIRCLE_AXIS_RATIO = 0.92
SUBPIXEL_SHIFT = 4


@dataclass
class VectorPrimitive:
    shape_id: int
    points: np.ndarray
    closed: bool


def to_network_input(gray: np.ndarray) -> np.ndarray:
    """uint8 (fundo branco, traço escuro) -> float32 WORK_SIZE com traço=1 e fundo=0.

    Única normalização usada no treino E na inferência.
    """
    if gray.shape != (WORK_SIZE, WORK_SIZE):
        gray = cv2.resize(gray, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    return 1.0 - gray.astype(np.float32) / 255.0


# ==============================================================================
# REDES
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv 3x3 -> BN -> ReLU) x2, com dilatação opcional."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetExpert(nn.Module):
    """U-Net de 4 níveis com gargalo dilatado.

    O campo receptivo cobre a imagem inteira: necessário para endireitar
    oscilações de baixa frequência (comprimento de onda de 45-90 px).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = nn.Sequential(DoubleConv(c4, c4, dilation=2), DoubleConv(c4, c4, dilation=4))
        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c3)
        self.up3 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d = self.dec3(torch.cat([self.up3(d), e3], dim=1))
        d = self.dec2(torch.cat([self.up2(d), e2], dim=1))
        d = self.dec1(torch.cat([self.up1(d), e1], dim=1))
        return self.out_conv(d)


class ShapeClassifier(nn.Module):
    """CNN de 5 blocos; pooling médio + máximo preserva evidência local (cantos)."""

    def __init__(self, num_classes: int = len(SHAPE_NAMES)) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96), block(96, 128))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(torch.cat([self.avg(f), self.max(f)], dim=1).flatten(1))


# ==============================================================================
# GEOMETRIA: AJUSTE ROBUSTO DE PRIMITIVAS
# ==============================================================================

def _rotate(pts: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]])


def _snap(angle: float, step_deg: float) -> float:
    step = np.deg2rad(step_deg)
    nearest = round(angle / step) * step
    return nearest if abs(angle - nearest) <= np.deg2rad(SNAP_DEGREES) else angle


def fit_line(pts: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reta robusta (Huber); extremos = projeções extremas (percentis contra respingos)."""
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    angle = _snap(float(np.arctan2(vy, vx)), 45.0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    origin = np.array([x0, y0], dtype=np.float64)
    proj = (pts - origin) @ direction
    t0, t1 = np.percentile(proj, [0.5, 99.5])
    return np.array([origin + t0 * direction, origin + t1 * direction]), False


def fit_ellipse(pts: np.ndarray) -> tuple[np.ndarray, bool]:
    """Elipse por mínimos quadrados (Direct), com 1 passada de rejeição de outliers."""

    def fit(p: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        (cx, cy), (w, h), ang = cv2.fitEllipseDirect(p.astype(np.float32))
        return np.array([cx, cy]), max(w / 2, 1.0), max(h / 2, 1.0), np.deg2rad(ang)

    center, a, b, ang = fit(pts)
    q = _rotate(pts - center, -ang)
    dist = np.abs(np.hypot(q[:, 0] / a, q[:, 1] / b) - 1.0) * min(a, b)
    inliers = pts[dist <= max(3.0, 3.0 * np.median(dist))]
    if len(inliers) >= 20:
        center, a, b, ang = fit(inliers)
        pts = inliers

    if min(a, b) / max(a, b) >= CIRCLE_AXIS_RATIO:
        a = b = float(np.median(np.hypot(*(pts - center).T)))

    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    return center + _rotate(np.column_stack([a * np.cos(t), b * np.sin(t)]), ang), True


def _rect_sides(q: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Atribui cada ponto ao lado mais próximo e reestima os lados pela mediana."""
    left, right, top, bottom = bounds
    dist = np.abs(np.column_stack([q[:, 0] - left, q[:, 0] - right, q[:, 1] - top, q[:, 1] - bottom]))
    side = dist.argmin(axis=1)
    new = bounds.copy()
    for i, axis in ((0, 0), (1, 0), (2, 1), (3, 1)):
        sel = side == i
        if sel.sum() >= 3:
            new[i] = np.median(q[sel, axis])
    return new, side


def _rect_angle_error(q: np.ndarray, side: np.ndarray, bounds: np.ndarray) -> float:
    """Resíduo de rotação estimado pela inclinação de cada lado (ignorando os cantos)."""
    left, right, top, bottom = bounds
    my, mx = 0.15 * (bottom - top), 0.15 * (right - left)
    deltas, weights = [], []
    for i in (0, 1):   # lados verticais: x = -tan(d) * y + c
        sel = (side == i) & (q[:, 1] > top + my) & (q[:, 1] < bottom - my)
        if sel.sum() >= 8:
            deltas.append(-np.arctan(np.polyfit(q[sel, 1], q[sel, 0], 1)[0]))
            weights.append(sel.sum())
    for i in (2, 3):   # lados horizontais: y = tan(d) * x + c
        sel = (side == i) & (q[:, 0] > left + mx) & (q[:, 0] < right - mx)
        if sel.sum() >= 8:
            deltas.append(np.arctan(np.polyfit(q[sel, 0], q[sel, 1], 1)[0]))
            weights.append(sel.sum())
    return float(np.average(deltas, weights=weights)) if deltas else 0.0


def fit_rectangle(pts: np.ndarray) -> tuple[np.ndarray, bool]:
    """Retângulo perfeito: orientação refinada pelos lados, posição dos lados por mediana.

    Robusto a overshoot/gap no fechamento, ao contrário de minAreaRect puro.
    """
    center = pts.mean(axis=0)
    theta = np.deg2rad(cv2.minAreaRect(pts.astype(np.float32))[2])

    def bounds_for(angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = _rotate(pts - center, -angle)
        bounds = np.concatenate([np.percentile(q[:, 0], [2, 98]), np.percentile(q[:, 1], [2, 98])])
        for _ in range(2):
            bounds, side = _rect_sides(q, bounds)
        return q, bounds, side

    for _ in range(4):
        q, bounds, side = bounds_for(theta)
        delta = _rect_angle_error(q, side, bounds)
        theta += delta
        if abs(delta) < 1e-4:
            break

    theta = _snap(theta, 90.0)
    _, (left, right, top, bottom), _ = bounds_for(theta)
    corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    return _rotate(corners, theta) + center, True


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    denom = float(delta @ delta)
    if denom < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ delta) / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * delta), axis=1)


def _line_intersection(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> Optional[np.ndarray]:
    cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(cross) < 1e-5:
        return None
    delta = point_b - point_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / cross
    return point_a + t * direction_a


def fit_triangle(pts: np.ndarray) -> tuple[np.ndarray, bool]:
    """Triângulo refinado por três retas robustas e suas interseções."""
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    corners = None
    for ratio in np.linspace(0.01, 0.12, 24):
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), ratio * perimeter, True).reshape(-1, 2)
        if len(approx) == 3:
            corners = approx.astype(np.float64)
            break
    if corners is None:
        _, enclosing = cv2.minEnclosingTriangle(pts.astype(np.float32))
        if enclosing is None:
            return np.empty((0, 2)), True
        corners = enclosing.reshape(3, 2).astype(np.float64)

    center = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))
    corners = corners[order]
    distances = np.column_stack([
        _point_segment_distance(pts, corners[i], corners[(i + 1) % 3]) for i in range(3)
    ])
    side = distances.argmin(axis=1)
    residual = distances[np.arange(len(pts)), side]
    cutoff = max(3.0, float(np.median(residual) + 3.0 * np.median(np.abs(residual - np.median(residual)))))

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(3):
        selected = pts[(side == i) & (residual <= cutoff)]
        if len(selected) < 4:
            selected = pts[side == i]
        if len(selected) < 2:
            return np.empty((0, 2)), True
        vx, vy, x0, y0 = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))

    refined = []
    for i in range(3):
        vertex = _line_intersection(*lines[(i - 1) % 3], *lines[i])
        if vertex is None:
            return np.empty((0, 2)), True
        refined.append(vertex)
    refined = np.asarray(refined)
    area = abs(cv2.contourArea(refined.astype(np.float32)))
    sides = np.linalg.norm(np.roll(refined, -1, axis=0) - refined, axis=1)
    if area < 150 or sides.min() < 12 or np.any((refined < -16) | (refined > 271)):
        return corners, True
    return refined, True


def _thin(mask: np.ndarray) -> np.ndarray:
    """Esqueletização Zhang-Suen vetorizada (dispensa opencv-contrib/scikit-image)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in range(2):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(p.astype(np.int32) for p in ring)
            a = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _longest_skeleton_path(skeleton: np.ndarray) -> np.ndarray:
    """Caminho geodésico mais longo do esqueleto (diâmetro via BFS duplo), ordenado."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.empty((0, 2))
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == biggest)
    index = -np.ones(skeleton.shape, dtype=np.int64)
    index[ys, xs] = np.arange(len(xs))

    h, w = skeleton.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors: list[list[int]] = [[] for _ in range(len(xs))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and index[yy, xx] >= 0:
                neighbors[i].append(int(index[yy, xx]))

    def bfs(start: int) -> tuple[int, np.ndarray]:
        parent = -np.ones(len(xs), dtype=np.int64)
        parent[start] = start
        queue, last = deque([start]), start
        while queue:
            last = queue.popleft()
            for nb in neighbors[last]:
                if parent[nb] < 0:
                    parent[nb] = last
                    queue.append(nb)
        return last, parent

    endpoints = [i for i, nb in enumerate(neighbors) if len(nb) == 1]
    far_a, _ = bfs(endpoints[0] if endpoints else 0)
    far_b, parent = bfs(far_a)

    path = [far_b]
    while path[-1] != far_a:
        path.append(int(parent[path[-1]]))
    return np.column_stack([xs[path], ys[path]]).astype(np.float64)


def fit_curve(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Curva aberta: linha central ordenada + B-spline suavizante."""
    path = _longest_skeleton_path(_thin(mask))
    if len(path) < 6:
        return path, False
    tck, _ = splprep([path[:, 0], path[:, 1]], s=len(path) * 1.5, k=3)
    return np.column_stack(splev(np.linspace(0, 1, 400), tck)), False


def _rdp_indices(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return np.arange(len(points))
    direction = points[-1] - points[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        offset = points - points[0]
        distances = np.abs(direction[0] * offset[:, 1] - direction[1] * offset[:, 0]) / norm
    index = int(np.argmax(distances))
    if distances[index] <= epsilon:
        return np.array([0, len(points) - 1])
    left = _rdp_indices(points[:index + 1], epsilon)
    right = _rdp_indices(points[index:], epsilon) + index
    return np.concatenate([left[:-1], right])


def fit_polyline(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Polilinha aberta: esqueleto ordenado, RDP e retas Huber por segmento."""
    connected = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    path = _longest_skeleton_path(_thin(connected))
    if len(path) < 12:
        return np.empty((0, 2)), False

    indices = np.array([0, len(path) - 1])
    for epsilon in (1.5, 2.0, 2.8, 3.8, 5.0):
        candidate = _rdp_indices(path, epsilon)
        if 3 <= len(candidate) <= 8:
            indices = candidate
            break
    if len(indices) < 3:
        return np.empty((0, 2)), False

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(indices[:-1], indices[1:]):
        segment = path[start:end + 1]
        if len(segment) < 3:
            return np.empty((0, 2)), False
        vx, vy, x0, y0 = cv2.fitLine(segment.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))

    first_point, first_dir = lines[0]
    last_point, last_dir = lines[-1]
    vertices = [first_point + ((path[0] - first_point) @ first_dir) * first_dir]
    for i in range(1, len(lines)):
        vertex = _line_intersection(*lines[i - 1], *lines[i])
        vertices.append(path[indices[i]] if vertex is None else vertex)
    vertices.append(last_point + ((path[-1] - last_point) @ last_dir) * last_dir)
    vertices = np.asarray(vertices)

    keep = np.concatenate([[True], np.linalg.norm(np.diff(vertices, axis=0), axis=1) >= 8.0])
    vertices = vertices[keep]
    if len(vertices) < 3:
        return np.empty((0, 2)), False
    return vertices, False


def _semantic_vertex_indices(primitive: VectorPrimitive) -> list[int]:
    if primitive.shape_id == 2:
        return []
    if primitive.shape_id == 3:
        return [0, len(primitive.points) - 1]
    return list(range(len(primitive.points)))


def _primitive_segments(primitive: VectorPrimitive):
    limit = len(primitive.points) if primitive.closed else len(primitive.points) - 1
    for i in range(max(limit, 0)):
        yield primitive.points[i], primitive.points[(i + 1) % len(primitive.points)]


def _project_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    denom = float(delta @ delta)
    if denom < 1e-9:
        return a.copy()
    t = np.clip(((point - a) @ delta) / denom, 0.0, 1.0)
    return a + t * delta


def snap_primitive(
    primitive: VectorPrimitive,
    scene: Sequence[VectorPrimitive],
    radius: float = SNAP_RADIUS,
) -> VectorPrimitive:
    """Conecta vértices do novo gesto a vértices ou arestas já existentes."""
    if not scene:
        return primitive
    points = primitive.points.copy()
    target_vertices = [
        existing.points[index]
        for existing in scene
        for index in _semantic_vertex_indices(existing)
    ]
    target_segments = [segment for existing in scene for segment in _primitive_segments(existing)]

    for index in _semantic_vertex_indices(primitive):
        point = points[index]
        candidate = None
        best = float(radius)
        for vertex in target_vertices:
            distance = float(np.linalg.norm(point - vertex))
            if distance <= best:
                candidate, best = vertex, distance
        if candidate is None:
            for a, b in target_segments:
                projection = _project_point_to_segment(point, a, b)
                distance = float(np.linalg.norm(point - projection))
                if distance <= best:
                    candidate, best = projection, distance
        if candidate is not None:
            points[index] = candidate

    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    relevant = edges if primitive.closed else edges[:-1]
    if len(relevant) and relevant.min() < 4.0:
        return primitive
    if primitive.closed and abs(cv2.contourArea(points.astype(np.float32))) < 50:
        return primitive
    return VectorPrimitive(primitive.shape_id, points, primitive.closed)


def render_primitive(primitive: VectorPrimitive, shape: tuple[int, int] = IMAGE_SIZE[::-1]) -> np.ndarray:
    result = np.full(shape, 255, dtype=np.uint8)
    if len(primitive.points) < 2:
        return result
    scaled = np.round(primitive.points * (1 << SUBPIXEL_SHIFT)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        result,
        [scaled],
        primitive.closed,
        0,
        OUTPUT_THICKNESS,
        cv2.LINE_AA,
        shift=SUBPIXEL_SHIFT,
    )
    return result


def render_scene(
    primitives: Sequence[VectorPrimitive],
    shape: tuple[int, int] = IMAGE_SIZE[::-1],
) -> np.ndarray:
    result = np.full(shape, 255, dtype=np.uint8)
    for primitive in primitives:
        result = np.minimum(result, render_primitive(primitive, shape))
    return result


# ==============================================================================
# PIPELINE
# ==============================================================================

class HybridShapeCorrector:
    """Limpa, classifica e regulariza matematicamente um rascunho."""

    def __init__(
        self,
        cleaner_checkpoint: Optional[str | Path] = None,
        classifier_checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cleaner = UNetExpert().to(self.device).eval()
        self.classifier: Optional[ShapeClassifier] = None
        self._load_checkpoint(self.cleaner, cleaner_checkpoint)
        if classifier_checkpoint:
            self.classifier = ShapeClassifier().to(self.device).eval()
            self._load_checkpoint(self.classifier, classifier_checkpoint)

    def _load_checkpoint(self, model: nn.Module, checkpoint: Optional[str | Path]) -> None:
        if not checkpoint:
            return
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint '{checkpoint}' incompativel com a arquitetura atual. "
                "Re-treine com train_hybrid.py."
            ) from exc
        model._hybrid_checkpoint_loaded = True  # type: ignore[attr-defined]

    @staticmethod
    def _has_checkpoint(model: nn.Module) -> bool:
        return bool(getattr(model, "_hybrid_checkpoint_loaded", False))

    @staticmethod
    def _as_gray(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("input_sketch_np must be a non-empty numpy array")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.ndim != 2:
            raise ValueError("input sketch must be grayscale or BGR")
        return cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)

    def clean_stroke(self, image: np.ndarray) -> np.ndarray:
        """Estágio 1: traço preto fino sobre fundo branco (IMAGE_SIZE)."""
        gray = self._as_gray(image)
        if self._has_checkpoint(self.cleaner):
            tensor = torch.from_numpy(to_network_input(gray))[None, None].to(self.device)
            with torch.inference_mode():
                prob = torch.sigmoid(self.cleaner(tensor))[0, 0].cpu().numpy()
            prob = cv2.resize(prob, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            foreground = np.where(prob > MASK_THRESHOLD, 255, 0).astype(np.uint8)
            stroke_pixels = np.count_nonzero(foreground)
            if MIN_STROKE_PIXELS <= stroke_pixels <= foreground.size * MAX_STROKE_FRACTION:
                return 255 - foreground

        # Sem rede treinada (ou saída degenerada): binarização determinística.
        _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return 255 - foreground

    def classify_shape(self, clean: np.ndarray) -> int:
        """Estágio 2: CNN quando treinada, senão heurística geométrica."""
        if self.classifier is not None:
            tensor = torch.from_numpy(to_network_input(clean))[None, None].to(self.device)
            with torch.inference_mode():
                return int(self.classifier(tensor).argmax(dim=1).item())
        return self._classify_by_geometry(clean)

    @staticmethod
    def _largest_contour(clean: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(255 - clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(contours, key=cv2.contourArea) if contours else None

    @classmethod
    def _classify_by_geometry(cls, clean: np.ndarray) -> int:
        contour = cls._largest_contour(clean)
        if contour is None or len(contour) < 2:
            return 0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area == 0 or perimeter == 0:
            return 0
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        _, _, width, height = cv2.boundingRect(contour)
        if len(polygon) == 3 and cv2.isContourConvex(polygon):
            return 4
        if 4 <= len(polygon) <= 6 and cv2.isContourConvex(polygon):
            return 1
        if circularity > 0.72 and abs(width - height) / max(width, height, 1) < 0.25:
            return 2
        if max(width, height) > 3 * max(min(width, height), 1):
            return 0
        polyline, _ = fit_polyline(cls._stroke_mask(clean))
        if len(polyline) >= 3:
            vectors = np.diff(polyline, axis=0)
            turns = []
            for a, b in zip(vectors[:-1], vectors[1:]):
                cosine = np.clip((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1.0, 1.0)
                turns.append(np.degrees(np.arccos(cosine)))
            if turns and max(turns) >= 25.0:
                return 5
        return 3

    @staticmethod
    def _stroke_mask(clean: np.ndarray) -> np.ndarray:
        """Máscara do traço sem respingos (componentes pequenos demais)."""
        fg = (clean < 128).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n <= 1:
            return fg.astype(bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= max(8, 0.1 * areas.max()))
        return np.isin(labels, keep)

    def vectorize(self, clean: np.ndarray, shape_id: int) -> Optional[VectorPrimitive]:
        """Ajusta uma primitiva vetorial sem descartar seus vértices e endpoints."""
        if not 0 <= shape_id < len(SHAPE_NAMES):
            raise ValueError(f"shape_id invalido: {shape_id}")
        mask = self._stroke_mask(clean)
        ys, xs = np.nonzero(mask)
        if len(xs) < MIN_STROKE_PIXELS:
            return None
        pts = np.column_stack([xs, ys]).astype(np.float64)

        if shape_id == 0:
            polyline, closed = fit_line(pts)
        elif shape_id == 1:
            polyline, closed = fit_rectangle(pts)
        elif shape_id == 2:
            polyline, closed = fit_ellipse(pts)
        elif shape_id == 3:
            polyline, closed = fit_curve(mask)
        elif shape_id == 4:
            polyline, closed = fit_triangle(pts)
        else:
            polyline, closed = fit_polyline(mask)
        if len(polyline) < 2:
            return None
        return VectorPrimitive(shape_id, polyline, closed)

    def regularize(self, clean: np.ndarray, shape_id: int) -> np.ndarray:
        """Estágio 3: ajusta a primitiva e renderiza um raster perfeito na posição original."""
        primitive = self.vectorize(clean, shape_id)
        return clean.copy() if primitive is None else render_primitive(primitive, clean.shape)

    def process_vector(self, input_sketch_np: np.ndarray) -> tuple[Optional[VectorPrimitive], str]:
        """Executa limpeza e classificação e preserva a primitiva vetorial resultante."""
        clean = self.clean_stroke(input_sketch_np)
        shape_id = self.classify_shape(clean)
        return self.vectorize(clean, shape_id), SHAPE_NAMES[shape_id]

    def process(self, input_sketch_np: np.ndarray) -> tuple[np.ndarray, str]:
        """Executa os 3 estágios e retorna ``(imagem_corrigida, nome_da_forma)``."""
        clean = self.clean_stroke(input_sketch_np)
        shape_id = self.classify_shape(clean)
        primitive = self.vectorize(clean, shape_id)
        corrected = clean.copy() if primitive is None else render_primitive(primitive, clean.shape)
        return corrected, SHAPE_NAMES[shape_id]


if __name__ == "__main__":
    sketch = np.full((*IMAGE_SIZE[::-1],), 255, dtype=np.uint8)
    cv2.rectangle(sketch, (55, 55), (200, 195), 0, 4)
    corrector = HybridShapeCorrector()
    corrected_image, shape_class_name = corrector.process(sketch)
    print(f"shape={shape_class_name}, output_shape={corrected_image.shape}")
