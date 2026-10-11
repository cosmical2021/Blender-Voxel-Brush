"""Grid Remesh core: Blender's Voxel Remesh applied to a brush area only.

The whole mesh is voxel-remeshed on a temporary copy; only the remeshed polygons inside the brush
area replace the original ones, and the seam is stitched back together.
"""

import bpy
import bmesh
import numpy as np
from mathutils import kdtree
from mathutils.geometry import intersect_point_line

from .core import grow, voxel_remesh
from .remesh import remesh_local


def _face_mask(mesh, centers, radius):
    n = len(mesh.polygons)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    co = np.empty(n * 3, dtype=np.float32)
    mesh.polygons.foreach_get("center", co)
    co = co.reshape(-1, 3)
    r2 = radius * radius
    for c in centers:
        d = co - np.asarray(c, dtype=np.float32)
        mask |= np.einsum("ij,ij->i", d, d) <= r2
    return mask


def _remeshed_copy(obj, voxel_size, adaptivity):
    """Voxel-remesh a temporary duplicate of obj's mesh; returns the new Mesh or None if too fine."""
    mesh = obj.data.copy()
    tmp = bpy.data.objects.new("_melt_merge_tmp", mesh)
    obj.users_collection[0].objects.link(tmp)
    ok = False
    try:
        with bpy.context.temp_override(active_object=tmp, object=tmp, selected_objects=[tmp],
                                       selected_editable_objects=[tmp]):
            ok = voxel_remesh(tmp, voxel_size, adaptivity)
    finally:
        bpy.data.objects.remove(tmp)
    if not ok:
        bpy.data.meshes.remove(mesh)
        return None
    return mesh


def _side_of_edge(e, b_faces):
    return e.link_faces[0] in b_faces


def _weld_pass(bm, movers, edges_pool, tol, b_faces, mover_is_b):
    """Join each mover vertex to the nearest boundary edge of the other piece (splitting it)."""
    edges = [e for e in edges_pool if e.is_valid and len(e.link_faces) == 1
             and _side_of_edge(e, b_faces) != mover_is_b]
    if not edges:
        return 0
    kd = kdtree.KDTree(len(edges))
    max_half = 0.0
    for i, e in enumerate(edges):
        a, b = e.verts[0].co, e.verts[1].co
        kd.insert((a + b) * 0.5, i)
        max_half = max(max_half, (a - b).length * 0.5)
    kd.balance()

    used_edges = set()
    welded = 0
    for v in movers:
        if not v.is_valid or not v.is_boundary:
            continue
        best = None
        for _co, i, _d in kd.find_range(v.co, tol + max_half):
            e = edges[i]
            if e in used_edges or not e.is_valid or v in e.verts:
                continue
            pt, t = intersect_point_line(v.co, e.verts[0].co, e.verts[1].co)
            if t <= 0.05 or t >= 0.95:
                continue
            d = (v.co - pt).length
            if d <= tol and (best is None or d < best[0]):
                best = (d, e, t)
        if best is None:
            continue
        _d, e, t = best
        used_edges.add(e)
        _ne, nv = bmesh.utils.edge_split(e, e.verts[0], t)
        if mover_is_b:
            bmesh.ops.weld_verts(bm, targetmap={v: nv})   # B vertex merges into the A-side point
        else:
            nv.co = v.co.copy()          # keep the A-side vertex position
            bmesh.ops.weld_verts(bm, targetmap={nv: v})
        welded += 1
    return welded


def stitch_seam(bm, n_orig_verts, b_faces, centers, radius, tol, hole_sides=12):
    """Weld the remeshed patch (b_faces) to the surrounding original mesh."""
    kd_c = kdtree.KDTree(len(centers))
    for i, c in enumerate(centers):
        kd_c.insert(c, i)
    kd_c.balance()
    zone = radius + 3.0 * tol

    def near(v):
        return bool(kd_c.find_range(v.co, zone))

    bm.verts.index_update()
    boundary = [v for v in bm.verts if v.is_boundary and near(v)]
    a_verts = [v for v in boundary if v.index < n_orig_verts]
    b_verts = [v for v in boundary if v.index >= n_orig_verts]

    if a_verts and b_verts:
        kd_a = kdtree.KDTree(len(a_verts))
        for i, v in enumerate(a_verts):
            kd_a.insert(v.co, i)
        kd_a.balance()
        cands = []
        for b in b_verts:
            for _co, i, d in kd_a.find_range(b.co, tol * 0.6):
                cands.append((d, a_verts[i], b))
        cands.sort(key=lambda c: c[0])
        used = set()
        targetmap = {}
        for _d, a, b in cands:
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            targetmap[b] = a
        if targetmap:
            bmesh.ops.weld_verts(bm, targetmap=targetmap)

    def boundary_now():
        verts = [v for v in bm.verts if v.is_valid and v.is_boundary and near(v)]
        edges = list({e for v in verts for e in v.link_edges if len(e.link_faces) == 1})
        return verts, edges

    verts, edges = boundary_now()
    movers_b = [v for v in verts if all(f in b_faces for f in v.link_faces)]
    _weld_pass(bm, movers_b, edges, tol, b_faces, True)
    verts, edges = boundary_now()
    movers_a = [v for v in verts if not any(f in b_faces for f in v.link_faces)]
    _weld_pass(bm, movers_a, edges, tol, b_faces, False)

    last = None
    for _ in range(10):
        bad = {e for v in bm.verts if v.is_valid and near(v) for e in v.link_edges
               if len(e.link_faces) > 2}
        if bad:
            kill = {min(e.link_faces, key=lambda f: f.calc_area()) for e in bad}
            bmesh.ops.delete(bm, geom=list(kill), context="FACES")
        verts, edges = boundary_now()
        if edges:
            res = bmesh.ops.holes_fill(bm, edges=edges, sides=hole_sides)
            faces = [f for f in res.get("faces", []) if f.is_valid and len(f.verts) > 3]
            if faces:
                bmesh.ops.triangulate(bm, faces=faces)
        state = (len(bad), len(edges))
        if state == (0, 0) or state == last:
            break
        last = state
    def is_defect(e):
        if len(e.link_faces) != 2:
            return True
        l0, l1 = e.link_loops
        return l0.vert == l1.vert

    def is_pinched(v):
        faces = list(v.link_faces)
        if len(faces) < 2:
            return False
        seen, stack = {faces[0]}, [faces[0]]
        while stack:
            f = stack.pop()
            for e in f.edges:
                if v in e.verts:
                    for g in e.link_faces:
                        if g not in seen:
                            seen.add(g)
                            stack.append(g)
        return len(seen) != len(faces)

    min_area = 1e-4 * tol * tol

    def defect_verts():
        out = set()
        for v in bm.verts:
            if not v.is_valid or not near(v):
                continue
            for e in v.link_edges:
                if is_defect(e):
                    out.update(e.verts)
            if is_pinched(v):
                out.add(v)
            for f in v.link_faces:
                if f.calc_area() < min_area:
                    out.update(f.verts)
        return out

    for attempt in range(4):
        ring = defect_verts()
        if not ring:
            break
        for _r in range(1 + attempt // 2):
            ring |= {w for v in ring for f in v.link_faces for w in f.verts}
        doomed = list({f for v in ring for f in v.link_faces})
        rim = {v for f in doomed for v in f.verts}
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        edges = list({e for v in rim if v.is_valid for e in v.link_edges if len(e.link_faces) == 1})
        if edges:
            res = bmesh.ops.holes_fill(bm, edges=edges, sides=1000)
            faces = [f for f in res.get("faces", []) if f.is_valid and len(f.verts) > 3]
            if faces:
                bmesh.ops.triangulate(bm, faces=faces)
            left = [e for e in edges if e.is_valid and len(e.link_faces) == 1]
            if left:
                bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=left)
    verts, _edges = boundary_now()
    return verts


def merge_region(obj, centers, radius, voxel_size, adaptivity=0.0, smooth_seam=True):
    """Voxel-remesh `obj` inside the brush area. centers/radius in object-local space.
    Returns False if voxel_size is too small for the mesh."""
    remeshed = _remeshed_copy(obj, voxel_size, adaptivity)
    if remeshed is None:
        return False
    splice_remeshed(obj, remeshed, centers, radius, voxel_size, smooth_seam)
    return True


def splice_remeshed(obj, remeshed, centers, radius, voxel_size, smooth_seam=True):
    """Replace obj's polygons inside the brush area with those of `remeshed` (a whole re-built
    version of the mesh, consumed here) and stitch the seam. centers/radius in object space."""
    mesh = obj.data
    mask_o = _face_mask(mesh, centers, radius)
    mask_r = _face_mask(remeshed, centers, radius)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    inside = [f for f in bm.faces if mask_o[f.index]]
    edge_len = [e.calc_length() for f in inside for e in f.edges]
    o_edge = float(np.mean(edge_len)) if edge_len else voxel_size
    was_open = any(len(e.link_faces) == 1 for f in inside for e in f.edges)
    bmesh.ops.delete(bm, geom=inside, context="FACES")

    bm_r = bmesh.new()
    bm_r.from_mesh(remeshed)
    bm_r.faces.ensure_lookup_table()
    outside = [f for f in bm_r.faces if not mask_r[f.index]]
    bmesh.ops.delete(bm_r, geom=outside, context="FACES")
    patch = bpy.data.meshes.new("_melt_merge_patch")
    bm_r.to_mesh(patch)
    bm_r.free()
    bpy.data.meshes.remove(remeshed)

    n_orig_verts = len(bm.verts)
    old_faces = set(bm.faces)
    bm.from_mesh(patch)
    bpy.data.meshes.remove(patch)
    b_faces = {f for f in bm.faces if f not in old_faces}

    tol = voxel_size * 1.5 + o_edge * 0.5
    seam = stitch_seam(bm, n_orig_verts, b_faces, centers, radius, tol,
                       hole_sides=12 if was_open else 256)
    wires = [e for e in bm.edges if not e.link_faces]
    if wires:
        bmesh.ops.delete(bm, geom=wires, context="EDGES")
    loose = [v for v in bm.verts if not v.link_edges]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    seam = [v for v in seam if v.is_valid]
    if smooth_seam and seam:
        remesh_local(bm, grow(seam, 2), voxel_size, 2, 0.3)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True
