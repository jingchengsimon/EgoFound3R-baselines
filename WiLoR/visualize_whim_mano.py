import argparse
import os
import pickle
import random
import warnings

import cv2
import numpy as np


DEFAULT_ROOT = "/media/hanyu/82FE717CFE716975/mnt/DATA/whim"
DEFAULT_MANO_DIR = "/home/hanyu/data1/mnt/MODELS/HUMAN/mano"


def load_faces(mano_dir):
    faces = {}
    for side, filename in (("right", "MANO_RIGHT.pkl"), ("left", "MANO_LEFT.pkl")):
        path = os.path.join(mano_dir, filename)
        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                data = pickle.load(f, encoding="latin1")
        faces[side] = np.asarray(data["f"], dtype=np.int32)
    return faces


def face_edges(faces):
    edges = set()
    for a, b, c in faces.tolist():
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(sorted(edges), dtype=np.int32)


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def side_name(value):
    value = float(np.asarray(value))
    return "right" if value > 0.5 else "left"


def project(points, intrinsics, trans):
    points_cam = points + trans.reshape(1, 3)
    z = points_cam[:, 2:3]
    uv = (points_cam @ intrinsics.T)[:, :2] / z
    return uv


def valid_point(point, width, height, margin=80):
    x, y = point
    return np.isfinite(x) and np.isfinite(y) and -margin <= x < width + margin and -margin <= y < height + margin


def draw_mesh(image, uv, faces, edges, color):
    height, width = image.shape[:2]
    overlay = image.copy()
    for face in faces:
        tri = uv[face]
        if all(valid_point(point, width, height) for point in tri):
            cv2.fillConvexPoly(overlay, np.round(tri).astype(np.int32), color)
    image[:] = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)

    edge_color = tuple(max(int(c * 0.65), 0) for c in color)
    for a, b in edges:
        pa, pb = uv[a], uv[b]
        if valid_point(pa, width, height) and valid_point(pb, width, height):
            cv2.line(image, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), edge_color, 1, cv2.LINE_AA)


def draw_joints(image, joints_3d, intrinsics, trans, color):
    uv = project(joints_3d, intrinsics, trans)
    height, width = image.shape[:2]
    for point in uv:
        if valid_point(point, width, height, margin=20):
            cv2.circle(image, tuple(np.round(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def draw_bbox(image, bbox, color):
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_annotation(image_path, anno_path, output_path, faces_by_side, edges_by_side):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"failed reading image {image_path}")

    annotation = np.load(anno_path, allow_pickle=True)
    palette = {
        "right": (80, 160, 255),
        "left": (80, 230, 120),
    }
    for index, item in enumerate(annotation.flat):
        side = side_name(item["side"])
        color = palette[side]
        intrinsics = np.asarray(item["K"], dtype=np.float32)
        trans = np.asarray(item["trans"], dtype=np.float32)
        vertices = np.asarray(item["vertices"], dtype=np.float32)
        joints_3d = np.asarray(item["joints_3d"], dtype=np.float32)
        uv = project(vertices, intrinsics, trans)

        draw_mesh(image, uv, faces_by_side[side], edges_by_side[side], color)
        draw_joints(image, joints_3d, intrinsics, trans, (255, 255, 255))
        draw_bbox(image, item["bbox"], color)

        x1, y1, _, _ = np.round(item["bbox"]).astype(int)
        label = f"{index}:{side}"
        cv2.putText(image, label, (max(x1, 0), max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    rel = os.path.basename(os.path.dirname(anno_path)) + "/" + os.path.basename(anno_path)
    cv2.putText(image, rel, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"failed writing {output_path}")


def collect_samples(anno_root, num_samples, samples_per_video, seed):
    video_dirs = [entry.path for entry in os.scandir(anno_root) if entry.is_dir()]
    rng = random.Random(seed)
    rng.shuffle(video_dirs)
    samples = []
    max_candidates = max(samples_per_video * 20, samples_per_video)

    for video_dir in video_dirs:
        frame_files = []
        for entry in os.scandir(video_dir):
            if not entry.name.endswith(".npy"):
                continue
            if not os.path.exists(os.path.join(video_dir, entry.name[:-4] + ".jpg")):
                continue
            frame_files.append(entry.name)
            if len(frame_files) >= max_candidates:
                break
        if not frame_files:
            continue
        chosen = rng.sample(frame_files, min(samples_per_video, len(frame_files)))
        for frame_file in sorted(chosen):
            anno_path = os.path.join(video_dir, frame_file)
            image_path = os.path.join(video_dir, frame_file[:-4] + ".jpg")
            samples.append((image_path, anno_path))
            if len(samples) >= num_samples:
                return samples
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT, help="WHIM root directory")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--anno-name", default="anno", help="Annotation folder name under WHIM/<mode>")
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--samples-per-video", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    anno_root = os.path.join(args.root, "WHIM", args.mode, args.anno_name)
    output_dir = args.output_dir or os.path.join(args.root, "visualizations", f"{args.mode}_{args.anno_name}_mano_overlay")

    faces_by_side = load_faces(args.mano_dir)
    edges_by_side = {side: face_edges(faces) for side, faces in faces_by_side.items()}
    samples = collect_samples(anno_root, args.num_samples, args.samples_per_video, args.seed)
    if not samples:
        raise RuntimeError(f"no image/annotation pairs found under {anno_root}")

    for idx, (image_path, anno_path) in enumerate(samples, start=1):
        video_id = os.path.basename(os.path.dirname(anno_path))
        frame_name = os.path.basename(anno_path)[:-4]
        output_path = os.path.join(output_dir, f"{idx:04d}_{video_id}_{frame_name}.jpg")
        draw_annotation(image_path, anno_path, output_path, faces_by_side, edges_by_side)
        print(f"[{idx}/{len(samples)}] {output_path}", flush=True)

    print(f"Saved {len(samples)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
