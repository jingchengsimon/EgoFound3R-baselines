"""3D world-space rendering: multi-view summary figure and temporal matrix video."""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper_viz_mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw

from . import style
from .scene import SKELETON_EDGES, Scene

VIEW_PRESETS = {
    "front": ("Front", 15.0, 90.0),
    "left": ("Oblique left", 30.0, 150.0),
    "top": ("Top", 65.0, 90.0),
    "right": ("Oblique right", 30.0, 30.0),
    "side": ("Side", 15.0, 0.0),
    "bottom": ("Bottom", -65.0, 90.0),
}
LIGHT_VECTOR = np.array([-0.4, 0.25, 0.88])
LIGHT_VECTOR = LIGHT_VECTOR / np.linalg.norm(LIGHT_VECTOR)


def render_cell(scene: Scene, method: str, elev: float, azim: float, pixel_size: int = 1024,
                frames: np.ndarray | None = None, gradient: bool = True,
                trajectory: bool = True) -> Image.Image:
    fig = Figure(figsize=(pixel_size / 150, pixel_size / 150), dpi=150, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho", computed_zorder=False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    low, high = scene.bounds
    center = (low + high) / 2
    span = max(high - low) * 0.50
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1), zoom=1.22)

    faces = scene.faces
    valid = scene.valid[method]
    xyz = scene.xyz[method]
    frames = scene.selected if frames is None else frames
    polygons = []
    colors = []
    for position, t in enumerate(frames):
        phase = (position / max(len(frames) - 1, 1)) if gradient else 0.5
        for side in range(2):
            if not valid[t, side]:
                continue
            tri = xyz[t, side][faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            intensity = 0.45 + 0.50 * np.abs(normal @ LIGHT_VECTOR)
            rgb = style.LIGHT[side] * (1 - phase) + style.DARK[side] * phase
            polygons.extend(tri)
            colors.extend(np.clip(intensity[:, None] * rgb, 0, 1))
    for scale, z, color in (((1.10, -0.012, [0.87, 0.88, 0.90]), (1, 0, [0.94, 0.945, 0.95]))
                            if elev >= 0 else ()):
        stage = scene.floor.copy()
        mid = stage.mean(0)
        stage[:, :2] = mid[:2] + (stage[:, :2] - mid[:2]) * scale
        stage[:, 2] += z
        ax.add_collection3d(Poly3DCollection([stage], facecolors=[color], edgecolors="none",
                                             zorder=0 if z else 1))
    if polygons:
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=colors, edgecolors="none",
                                             linewidths=0, antialiased=False, zsort="average", zorder=3))
    else:
        joints = scene.joints.get(method)
        if joints is not None:
            ok = scene.joints_valid.get(method)
            segments = []
            segment_colors = []
            rgb = np.array(style.METHOD_COLORS[method]) / 255.0
            for position, t in enumerate(frames):
                phase = (position / max(len(frames) - 1, 1)) if gradient else 0.5
                tint = style.LIGHT[0] * (1 - phase) + style.DARK[0] * phase
                for side in range(2):
                    if not ok[t, side]:
                        continue
                    for a, b in SKELETON_EDGES:
                        segments.append([joints[t, side, a], joints[t, side, b]])
                        segment_colors.append((*tint, 0.95))
            if segments:
                ax.add_collection3d(Line3DCollection(segments, colors=segment_colors,
                                                     linewidths=1.6, zorder=3))
    ax.add_collection3d(Line3DCollection(scene.camera_parts, colors=[(0.72, 0.48, 0.14, 0.60)],
                                         linewidths=0.55, zorder=2))
    if trajectory:
        traj = scene.trajectories[method]
        ok = scene.traj_valid[method]
        segments = []
        for side in range(2):
            mask = ok[:, side]
            idx = np.flatnonzero(mask)
            for a, b in zip(idx[:-1], idx[1:]):
                if b == a + 1:
                    segments.append([traj[a, side], traj[b, side]])
        if segments:
            rgb = np.array(style.METHOD_COLORS[method]) / 255.0
            ax.add_collection3d(Line3DCollection(segments, colors=[(*rgb, 0.75)],
                                                 linewidths=1.1, zorder=4))
    canvas.draw()
    result = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
    fig.clear()
    return result


def shared_crop(images: list, margin: int = 35) -> tuple:
    boxes = []
    for im in images:
        yy, xx = np.where((np.asarray(im) < 247).any(-1))
        boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    box = np.array(boxes)
    size = images[0].size
    return (max(0, int(box[:, 0].min()) - margin), max(0, int(box[:, 1].min()) - margin),
            min(size[0], int(box[:, 2].max()) + margin), min(size[1], int(box[:, 3].max()) + margin))


def figure1(scene: Scene, methods: tuple, views: tuple, cell: int = 760) -> Image.Image:
    """Rows = methods, columns = views; five temporal samples with light-to-dark gradient."""
    from .layout import compose_grid
    cells = {}
    for col, key in enumerate(views):
        label, elev, azim = VIEW_PRESETS[key]
        group = [render_cell(scene, method, elev, azim, pixel_size=cell) for method in methods]
        crop = shared_crop(group)
        for row, (method, im) in enumerate(zip(methods, group)):
            cropped = im.crop(crop)
            cropped = cropped.resize((cell, cell), Image.Resampling.LANCZOS)
            cells[(row, col)] = cropped
    row_labels = [style.METHOD_SHORT[m] for m in methods]
    col_labels = [VIEW_PRESETS[key][0] for key in views]
    return compose_grid(cells, len(methods), len(views), row_labels, col_labels,
                        title=f"{scene.dataset} | {scene.sequence_id} | {scene.segment_id}",
                        subtitle="3D world space | five temporal samples, earlier lighter to later darker | "
                                 "bronze: GT camera frustums and path | colored line: method wrist trajectory",
                        cell_size=cell)


def video1(scene: Scene, methods: tuple, views: tuple, cell: int = 480, fps: int = 30):
    """Yield composed frames: rows = methods, columns = views, single current frame."""
    from .layout import compose_grid
    total = len(scene.frame_ids)
    for t in range(total):
        cells = {}
        for col, key in enumerate(views):
            label, elev, azim = VIEW_PRESETS[key]
            group = [render_cell(scene, method, elev, azim, pixel_size=cell,
                                 frames=np.array([t]), gradient=False, trajectory=True)
                     for method in methods]
            crop = shared_crop(group)
            for row, (method, im) in enumerate(zip(methods, group)):
                cropped = im.crop(crop)
                cropped = cropped.resize((cell, cell), Image.Resampling.LANCZOS)
                cells[(row, col)] = cropped
        row_labels = [style.METHOD_SHORT[m] for m in methods]
        col_labels = [VIEW_PRESETS[key][0] for key in views]
        yield compose_grid(cells, len(methods), len(views), row_labels, col_labels,
                           title=f"{scene.dataset} | {scene.sequence_id} | frame {t}/{total - 1}",
                           subtitle="3D world space | current frame only | bronze: GT camera frustum and path",
                           cell_size=cell)
