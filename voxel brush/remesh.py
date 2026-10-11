"""Local isotropic remesher (Botsch-Kobbelt style) for the areas touched by a melt."""

import bmesh


def _live(region):
    return {v for v in region if v.is_valid}


def _split_long(bm, region, hi):
    edges = {e for v in region for e in v.link_edges
             if e.verts[0] in region and e.verts[1] in region and e.calc_length() > hi}
    if not edges:
        return
    res = bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
    geom = res["geom"]
    new_verts = [g for g in geom if isinstance(g, bmesh.types.BMVert)]
    faces = [g for g in geom if isinstance(g, bmesh.types.BMFace) and len(g.verts) > 3]
    if faces:
        bmesh.ops.triangulate(bm, faces=faces)
    region.update(new_verts)


def _collapse_short(bm, region, lo, hi):
    edges = {e for v in region for e in v.link_edges
             if e.verts[0] in region and e.verts[1] in region and e.calc_length() < lo}
    for e in sorted(edges, key=lambda e: e.calc_length() if e.is_valid else 0.0):
        if not e.is_valid or len(e.link_faces) != 2:
            continue
        a, b = e.verts
        if any(len(f.verts) != 3 for f in e.link_faces):
            continue
        na = {x for ed in a.link_edges for x in ed.verts} - {a, b}
        nb = {x for ed in b.link_edges for x in ed.verts} - {a, b}
        if len(na & nb) != 2:
            continue
        if any(len(ed.link_faces) != 2 for ed in a.link_edges) or \
                any(len(ed.link_faces) != 2 for ed in b.link_edges):
            continue
        mid = (a.co + b.co) * 0.5
        if any((mid - w.co).length > hi for w in na | nb):
            continue
        bmesh.ops.collapse(bm, edges=[e])
        region.discard(a)
        region.discard(b)
        region.update(w for w in na | nb if w.is_valid)


def _opposite_sides(p, q, r, s, n):
    """True if r and s lie on opposite sides of line p->q (projected on normal n)."""
    d = q - p
    return d.cross(r - p).dot(n) * d.cross(s - p).dot(n) < 0.0


def _flip(bm, region):
    edges = {e for v in region for e in v.link_edges
             if e.verts[0] in region and e.verts[1] in region and len(e.link_faces) == 2}
    for e in edges:
        if not e.is_valid or len(e.link_faces) != 2:
            continue
        f0, f1 = e.link_faces
        if len(f0.verts) != 3 or len(f1.verts) != 3:
            continue
        a, b = e.verts
        c = next(v for v in f0.verts if v not in (a, b))
        d = next(v for v in f1.verts if v not in (a, b))
        if c == d or any(x == d for ed in c.link_edges for x in ed.verts):
            continue
        before = sum((len(v.link_edges) - 6) ** 2 for v in (a, b, c, d))
        after = ((len(a.link_edges) - 7) ** 2 + (len(b.link_edges) - 7) ** 2 +
                 (len(c.link_edges) - 5) ** 2 + (len(d.link_edges) - 5) ** 2)
        if after >= before:
            continue
        n = f0.normal + f1.normal
        if n.length < 1e-9:
            continue
        if not (_opposite_sides(c.co, d.co, a.co, b.co, n) and
                _opposite_sides(a.co, b.co, c.co, d.co, n)):
            continue
        bmesh.utils.edge_rotate(e, True)


def _smooth(bm, region, amount):
    bm.normal_update()
    moves = []
    for v in region:
        if not v.is_valid or not v.link_edges:
            continue
        if any(len(e.link_faces) != 2 for e in v.link_edges):
            continue
        avg = sum((e.other_vert(v).co for e in v.link_edges), v.co * 0.0) / len(v.link_edges)
        delta = avg - v.co
        delta -= v.normal * delta.dot(v.normal)
        moves.append((v, delta * amount))
    for v, d in moves:
        v.co += d


def remesh_local(bm, verts, detail, iterations=3, smooth=0.5):
    """Re-tessellate the neighbourhood `verts` towards edge length `detail`."""
    region = _live(verts)
    lo, hi = detail * 0.8, detail * 4.0 / 3.0
    for _ in range(iterations):
        if not region:
            break
        _split_long(bm, region, hi)
        region = _live(region)
        _collapse_short(bm, region, lo, hi)
        region = _live(region)
        _flip(bm, region)
        if smooth > 0.0:
            _smooth(bm, region, smooth)
    bm.normal_update()
    return region
