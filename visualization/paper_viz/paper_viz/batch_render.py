"""Batched variant of the reference ``HandMultiviewRenderer.render``.

The reference renders one cell per call, which costs ~0.15 s of fixed overhead
(pytorch3d ``Meshes`` construction + rasterizer/shader assembly + kernel
launches) regardless of the 192 px cell.  For a comparison matrix that repeats
the same viewpoint for every method, the eight hands can be rasterised and shaded
in a single batched call (pytorch3d broadcasts one camera over a mesh batch).

The arithmetic — ground boxes, two-light sum, receiver masking, the projected
shadow blur, rounding and the INTER_AREA downscale — is copied from the reference
so the batched images are pixel-identical to the per-cell path.
"""
from __future__ import annotations

import cv2
import numpy as np

from egohandmetric_prompt.inference_multiview import _box


def batched_render(renderer, parts_list, view, shadow_parts_list=None):
    """Render ``parts_list`` (one entry per cell) for one viewpoint at once."""
    import torch
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes
    from t3drender.render.lights import PointLights

    pose = renderer.view_poses[view]
    low, high = renderer.bounds
    center = (low + high) * .5
    radius = max(float(np.linalg.norm(high - low)) * .5, .12)
    floor_y, width, depth = renderer.floor_y, renderer.floor_width, renderer.floor_depth
    has_ground = renderer.ground and view != "bottom"

    verts, faces, colours, hand_faces = [], [], [], []
    for parts in parts_list:
        all_parts = list(parts)
        hand_faces.append(sum(len(part[1]) for part in parts))
        if has_ground:
            all_parts += [_box([center[0], floor_y - .009, center[2]], [width, .018, depth], [.94, .95, .97]),
                          _box([center[0], floor_y - .020, center[2]], [width * 1.12, .006, depth * 1.12], [.86, .88, .91])]
        cells_v, cells_f, cells_c, offset = [], [], [], 0
        for xyz, triangles, colour in all_parts:
            cells_v.append((xyz @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            cells_f.append(triangles + offset)
            cells_c.append(np.broadcast_to(colour, (len(xyz), 3)))
            offset += len(xyz)
        verts.append(torch.as_tensor(np.concatenate(cells_v), dtype=torch.float32, device=renderer.device))
        faces.append(torch.as_tensor(np.concatenate(cells_f), dtype=torch.long, device=renderer.device))
        colours.append(torch.as_tensor(np.concatenate(cells_c), dtype=torch.float32, device=renderer.device))
    mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=colours))
    # pytorch3d reports pix_to_face with *global* face ids that keep counting across
    # the batch (item i starts at the cumulative face count of items < i), so the
    # per-cell masks have to subtract that offset before comparing with the cell's
    # own face layout.
    face_offsets = np.concatenate([[0], np.cumsum([len(f) for f in faces])[:-1]]).astype(np.int64)

    fragments = renderer.renderer.rasterizer(meshes_world=mesh, cameras=renderer.cameras)
    center_camera = center @ pose[:3, :3].T + pose[:3, 3]
    count = len(parts_list)
    size = renderer.render_size
    images = np.zeros((count, size, size, 3), dtype=np.float32)
    for offset, ambient, diffuse, specular in (
        ((-1., -1.4, -1.6), .30, .65, .45),
        ((1.2, .2, -.8), 0., .28, .08),
    ):
        location = center_camera + np.asarray(offset) * radius
        lights = PointLights(device=renderer.device, location=[location.tolist()],
                             ambient_color=((ambient,) * 3,), diffuse_color=((diffuse,) * 3,),
                             specular_color=((specular,) * 3,))
        rendered = renderer.renderer.shader(fragments=fragments, meshes=mesh,
                                            cameras=renderer.cameras, lights=lights)
        images += rendered[..., :3].cpu().numpy()
    face_ids = fragments.pix_to_face[..., 0].cpu().numpy()

    shadows = shadow_parts_list if shadow_parts_list is not None else parts_list
    if has_ground and any(len(parts) for parts in shadows):
        verts, faces, colours = [], [], []
        for parts in shadows:
            cells_v, cells_f, offset = [], [], 0
            for xyz, triangles, _ in parts:
                shadow = xyz.copy()
                height = shadow[:, 1] - floor_y
                shadow[:, 0] += height * .25
                shadow[:, 2] -= height * .15
                shadow[:, 1] = floor_y + .0005
                cells_v.append((shadow @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
                cells_f.append(triangles + offset)
                offset += len(shadow)
            if not cells_v:
                cells_v, cells_f = [np.zeros((0, 3), np.float32)], [np.zeros((0, 3), np.int64)]
            verts.append(torch.as_tensor(np.concatenate(cells_v), dtype=torch.float32, device=renderer.device))
            faces.append(torch.as_tensor(np.concatenate(cells_f), dtype=torch.long, device=renderer.device))
            colours.append(torch.ones((len(verts[-1]), 3), dtype=torch.float32, device=renderer.device))
        projected = renderer.renderer.rasterizer(
            meshes_world=Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=colours)),
            cameras=renderer.cameras)
        masks = (projected.pix_to_face[..., 0] >= 0).float().cpu().numpy()
    else:
        masks = np.zeros((count, size, size), np.float32)

    out = []
    for index in range(count):
        image = images[index]
        local = face_ids[index] - face_offsets[index]
        receiver = local >= hand_faces[index]
        image[local < 0] = 1.
        image[receiver] = .65 + .30 * np.clip(image[receiver], 0., 1.)
        if has_ground:
            soft = cv2.GaussianBlur(masks[index], (0, 0), sigmaX=max(1., size / 180))
            broad = cv2.GaussianBlur(masks[index], (0, 0), sigmaX=max(2., size / 75))
            image *= 1 - (.09 * soft + .035 * broad)[..., None] * receiver[..., None]
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        out.append(cv2.resize(image, (renderer.cell_size, renderer.cell_size),
                              interpolation=cv2.INTER_AREA))
    return out
