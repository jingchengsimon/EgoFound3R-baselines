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

# Optional rasterizer bin grid.  ``None`` reproduces the reference exactly
# (bit-identical frames).  A finite bin size rasterises ~1.66x faster at 192 px
# but touches bin-boundary rounding: measured on the smoke segment it changes
# exactly one pixel by one grey level in one of 300 frames, i.e. visually
# identical but no longer bit-exact.  Opt in with ``--bin-size 64``.
_RASTERIZERS: dict = {}


def _rasterize(renderer, mesh, bin_size):
    """Rasterize with the fast bin grid, falling back to the reference one.

    pytorch3d caps the number of faces per bin (<22): at large cells (2048 px
    panels) a 64-px bin grid becomes too fine for a dense hand mesh and raises, so
    the reference rasterizer is used there instead.  Both produce the same option
    of fragments; only the binning differs.
    """
    try:
        return _rasterizer(renderer, bin_size)(meshes_world=mesh, cameras=renderer.cameras)
    except ValueError as error:
        if "faces per bin" not in str(error):
            raise
        return renderer.renderer.rasterizer(meshes_world=mesh, cameras=renderer.cameras)


def _rasterizer(renderer, bin_size):
    if not bin_size:
        return renderer.renderer.rasterizer
    key = (id(renderer.renderer), int(bin_size))
    cached = _RASTERIZERS.get(key)
    if cached is None:
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        cached = MeshRasterizer(cameras=renderer.cameras, raster_settings=RasterizationSettings(
            image_size=renderer.render_size, blur_radius=0., faces_per_pixel=1,
            perspective_correct=True, cull_backfaces=False, bin_size=int(bin_size),
            max_faces_per_bin=50000))
        _RASTERIZERS[key] = cached
    return cached

def batched_render(renderer, parts_list, view, shadow_parts_list=None, bin_size=None):
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

    # Assemble every cell in numpy first, then upload as ONE padded batch: creating
    # and transferring a tensor per cell cost ~105 ms per viewpoint, while a single
    # padded upload of the same data is a fraction of that (padding vertices are
    # never referenced by any face, so the geometry is bit-identical).
    cells, hand_faces = [], []
    for parts in parts_list:
        all_parts = list(parts)
        hand_faces.append(sum(len(part[1]) for part in parts))
        if has_ground:
            all_parts += [_box([center[0], floor_y - .009, center[2]], [width, .018, depth], [.94, .95, .97]),
                          _box([center[0], floor_y - .020, center[2]], [width * 1.12, .006, depth * 1.12], [.86, .88, .91])]
        cv, cf, cc, offset = [], [], [], 0
        for xyz, triangles, colour in all_parts:
            cv.append((xyz @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            cf.append(triangles + offset)
            cc.append(np.broadcast_to(colour, (len(xyz), 3)))
            offset += len(xyz)
        cells.append((np.concatenate(cv), np.concatenate(cf).astype(np.int64), np.concatenate(cc).astype(np.float32)))
    # One upload per attribute, then zero-copy tensor slices per cell: building a
    # tensor per cell cost ~105 ms per viewpoint, this costs a few ms.
    all_verts = torch.as_tensor(np.concatenate([v for v, _, _ in cells]), dtype=torch.float32,
                                device=renderer.device)
    all_colours = torch.as_tensor(np.concatenate([c for _, _, c in cells]), dtype=torch.float32,
                                  device=renderer.device)
    all_faces = torch.as_tensor(np.concatenate([f for _, f, _ in cells]), dtype=torch.long,
                                device=renderer.device)
    verts, colours, faces, v_off, f_off = [], [], [], 0, 0
    for vertex, triangles, colour in cells:
        verts.append(all_verts[v_off:v_off + len(vertex)])
        colours.append(all_colours[v_off:v_off + len(colour)])
        faces.append(all_faces[f_off:f_off + len(triangles)])
        v_off += len(vertex)
        f_off += len(triangles)
    mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=colours))
    # pytorch3d reports pix_to_face with *global* face ids that keep counting across
    # the batch (item i starts at the cumulative face count of items < i), so the
    # per-cell masks have to subtract that offset before comparing with the cell's
    # own face layout.
    face_offsets = np.concatenate([[0], np.cumsum([len(f) for f in faces])[:-1]]).astype(np.int64)

    fragments = _rasterize(renderer, mesh, bin_size)
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
        cells = []
        for parts in shadows:
            cv, cf, offset = [], [], 0
            for xyz, triangles, _ in parts:
                shadow = xyz.copy()
                height = shadow[:, 1] - floor_y
                shadow[:, 0] += height * .25
                shadow[:, 2] -= height * .15
                shadow[:, 1] = floor_y + .0005
                cv.append((shadow @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
                cf.append(triangles + offset)
                offset += len(shadow)
            if not cv:
                cv, cf = [np.zeros((0, 3), np.float32)], [np.zeros((0, 3), np.int64)]
            cells.append((np.concatenate(cv), np.concatenate(cf).astype(np.int64)))
        all_verts = torch.as_tensor(np.concatenate([v for v, _ in cells]), dtype=torch.float32,
                                    device=renderer.device)
        all_faces = torch.as_tensor(np.concatenate([f for _, f in cells]), dtype=torch.long,
                                    device=renderer.device)
        verts, faces, v_off, f_off = [], [], 0, 0
        for vertex, triangles in cells:
            verts.append(all_verts[v_off:v_off + len(vertex)])
            faces.append(all_faces[f_off:f_off + len(triangles)])
            v_off += len(vertex)
            f_off += len(triangles)
        colours = [torch.ones_like(vertex) for vertex in verts]
        projected = _rasterize(renderer,
                               Meshes(verts=verts, faces=faces,
                                      textures=TexturesVertex(verts_features=colours)),
                               bin_size)
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
