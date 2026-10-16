from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from hybrid_vectorizer import (
    IMAGE_SIZE,
    HybridShapeCorrector,
    VectorPrimitive,
    render_scene,
    snap_primitive,
)

CANVAS_W, CANVAS_H = IMAGE_SIZE
SKETCH_THICKNESS = 3


class CanvasHybridApp:
    """Canvas vetorial: corrige cada gesto e conecta-o às formas anteriores."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Canvas Hibrido (U-Net + Classifier + Fit Vetorial)")
        self.root.resizable(False, False)

        # Tamanho fixo: o pipeline trabalha exatamente em IMAGE_SIZE, sem reescala de coordenadas.
        self.canvas = tk.Canvas(
            root, width=CANVAS_W, height=CANVAS_H, bg="white",
            highlightthickness=0, cursor="pencil",
        )
        self.canvas.pack(padx=12, pady=(12, 6))

        self.raw_strokes: list[list[tuple[int, int]]] = []
        self.primitive_history: list[VectorPrimitive | None] = []
        self.scene: list[VectorPrimitive] = []
        self.current_points: list[tuple[int, int]] = []
        self.current_raw_id: int | None = None
        self.composite = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        self.composite_photo: ImageTk.PhotoImage | None = None

        project_dir = Path(__file__).resolve().parent.parent
        model_dir = next(
            (d for d in (project_dir / "models", project_dir / "model_artist_v2", project_dir) if (d / "classifier_model.pt").exists()),
            project_dir,
        )
        cleaner_path = model_dir / "cleaner_model.pt"
        classifier_path = model_dir / "classifier_model.pt"
        cleaner_ckpt = cleaner_path if cleaner_path.exists() else None
        classifier_ckpt = classifier_path if classifier_path.exists() else None
        self.corrector = HybridShapeCorrector(
            cleaner_checkpoint=cleaner_ckpt,
            classifier_checkpoint=classifier_ckpt,
            device="cpu",
        )

        toolbar = tk.Frame(root)
        toolbar.pack(padx=12, pady=(0, 8), fill="x")
        tk.Button(toolbar, text="Desfazer", command=self.undo).pack(side="left", padx=(0, 8))
        tk.Button(toolbar, text="Limpar", command=self.clear_canvas).pack(side="left")
        mode = "redes treinadas" if cleaner_ckpt and classifier_ckpt else "sem checkpoints (fallback OpenCV)"
        self.status = tk.Label(toolbar, text=f"Pronto - {mode}", anchor="w", fg="blue")
        self.status.pack(side="left", padx=8)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event: tk.Event) -> None:
        self.current_points = [(event.x, event.y)]
        self.current_raw_id = self.canvas.create_line(
            event.x, event.y, event.x, event.y,
            fill="#111827", width=SKETCH_THICKNESS, capstyle=tk.ROUND, joinstyle=tk.ROUND,
        )

    def on_drag(self, event: tk.Event) -> None:
        if not self.current_points:
            return
        self.current_points.append((event.x, event.y))
        flat = [coord for pt in self.current_points for coord in pt]
        self.canvas.coords(self.current_raw_id, *flat)

    def on_release(self, event: tk.Event) -> None:
        if not self.current_points:
            return
        self.current_points.append((event.x, event.y))
        stroke = self.current_points
        self.current_points = []
        self.current_raw_id = None
        if len(stroke) < 3:
            self.redraw()
            return
        self.raw_strokes.append(stroke)
        shape_name = self.correct_into_composite(stroke)
        self.redraw()
        self.status.config(text=f"Forma: {shape_name}", fg="green")

    def correct_into_composite(self, stroke: list[tuple[int, int]]) -> str:
        """Vetoriza o novo gesto, aplica snapping contextual e redesenha a cena."""
        sketch = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        path = np.asarray(stroke, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(sketch, [path], isClosed=False, color=0, thickness=SKETCH_THICKNESS)
        primitive, shape_name = self.corrector.process_vector(sketch)
        if primitive is not None:
            primitive = snap_primitive(primitive, self.scene)
            self.scene.append(primitive)
        self.primitive_history.append(primitive)
        self.composite = render_scene(self.scene, (CANVAS_H, CANVAS_W))
        return shape_name

    def redraw(self) -> None:
        self.canvas.delete("all")
        self.composite_photo = ImageTk.PhotoImage(Image.fromarray(self.composite))
        self.canvas.create_image(0, 0, image=self.composite_photo, anchor="nw")

    def undo(self) -> None:
        if not self.raw_strokes:
            return
        self.raw_strokes.pop()
        primitive = self.primitive_history.pop()
        if primitive is not None:
            self.scene.pop()
        self.composite = render_scene(self.scene, (CANVAS_H, CANVAS_W))
        self.redraw()
        self.status.config(text="Desfeito", fg="black")

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.raw_strokes.clear()
        self.primitive_history.clear()
        self.scene.clear()
        self.current_points = []
        self.composite = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        self.composite_photo = None
        self.status.config(text="Canvas limpo", fg="black")


if __name__ == "__main__":
    root = tk.Tk()
    app = CanvasHybridApp(root)
    root.mainloop()
