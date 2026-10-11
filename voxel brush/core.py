"""Shared mesh helpers used by the Grid Remesh brush."""


def grow(verts, rings):
    """Return `verts` expanded by `rings` edge rings (only valid verts)."""
    seen = {v for v in verts if v.is_valid}
    frontier = list(seen)
    for _ in range(rings):
        nxt = []
        for a in frontier:
            for e in a.link_edges:
                b = e.other_vert(a)
                if b not in seen:
                    seen.add(b)
                    nxt.append(b)
        frontier = nxt
    return seen


MAX_VOXELS = 30_000_000


def voxel_remesh(obj, voxel_size, adaptivity=0.0):
    """Run Blender's Voxel Remesh on `obj` (Object Mode). False if the voxel grid would be too large."""
    import bpy

    xs = [c[0] for c in obj.bound_box]
    ys = [c[1] for c in obj.bound_box]
    zs = [c[2] for c in obj.bound_box]
    cells = 1.0
    for lo_hi in (xs, ys, zs):
        cells *= max((max(lo_hi) - min(lo_hi)) / voxel_size, 1.0)
    if cells > MAX_VOXELS:
        return False
    mesh = obj.data
    mesh.remesh_voxel_size = voxel_size
    mesh.remesh_voxel_adaptivity = adaptivity
    bpy.ops.object.voxel_remesh()
    return True
