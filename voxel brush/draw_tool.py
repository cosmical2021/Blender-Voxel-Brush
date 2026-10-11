"""Voxel Draw brush: paint voxel cells on and around the surface (Ctrl erases painted cells),
then press Enter or Space to apply - Add fills the drawn voxels with solid mesh, Subtract carves
them out of the mesh. B toggles Voxel Bridge: click point A, then point B, to connect the two
points with a straight tube of voxels."""

import math

import bpy
import bmesh
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from .grid_tool import (MAX_CELLS, TOOL_SETTINGS, _draw_circle, _viewport_region, _window_region,
                        cell_edges)
from .merge import merge_region

MAX_STAMPS = 512
MAX_BRIDGE_CELLS = 2_000_000
ADD_COLOR = (0.3, 0.75, 1.0)
SUB_COLOR = (1.0, 0.35, 0.3)
ERASE_COLOR = (0.8, 0.8, 0.8)
BRIDGE_COLOR = (1.0, 0.85, 0.2)


class VoxelDrawSettings(PropertyGroup):
    active: BoolProperty(name="Active", default=False, options={"SKIP_SAVE"})
    mode: EnumProperty(
        name="Mode", default="ADD",
        items=(
            ("ADD", "Add", "Fill the drawn voxels with solid mesh and connect it to the surface"),
            ("SUBTRACT", "Subtract", "Carve the drawn voxels out of the mesh"),
        ))
    radius: IntProperty(
        name="Radius", default=50, min=10, max=500, subtype="PIXEL",
        description="Brush radius in pixels (also [ and ] while the tool is active)")
    voxel_size: FloatProperty(
        name="Voxel Size", default=0.05, min=0.0001, soft_max=1.0, subtype="DISTANCE",
        description="Size of the drawn voxel cells and of the remesh (also R while the tool is active)")
    remesh_result: BoolProperty(
        name="Remesh Result", default=True,
        description="Voxel-remesh the affected area after applying so it becomes one even surface "
                    "(off: keep the sharp voxel cubes, joined with a boolean)")


def _ball_cells(center, radius, voxel):
    """Integer indices of the cells whose centers lie within `radius` of `center` (local space)."""
    c = np.asarray(center, np.float64) / voxel
    r = radius / voxel
    lo = np.floor(c - r).astype(np.int64)
    hi = np.ceil(c + r).astype(np.int64)
    i, j, k = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    pts = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    d = pts - c
    sel = pts[np.einsum("ij,ij->i", d, d) <= r * r]
    if len(sel) == 0:
        sel = np.rint(c)[None].astype(np.int64)
    return sel


def _segment_cells(a, b, radius, voxel):
    """Cells of a capsule (tube with round caps) from `a` to `b` (local space), as a set of
    integer (i, j, k) tuples. Returns None when the capsule would hold too many cells."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    length = float(np.linalg.norm(b - a))
    rv = radius / voxel
    estimate = math.pi * rv * rv * (length / voxel) + 4.19 * rv ** 3
    if estimate > MAX_BRIDGE_CELLS:
        return None
    step = max(radius * 0.5, voxel * 0.5)
    n = int(math.ceil(length / step)) if length > 0.0 else 0
    cells = set()
    for t in np.linspace(0.0, 1.0, n + 1):
        p = a + (b - a) * t
        cells.update(map(tuple, _ball_cells(p, radius, voxel).tolist()))
    return cells


def cells_to_mesh(cells, voxel):
    """Closed mesh of the outer boundary of the voxel cell set (cell i spans
    (i - 0.5) * voxel .. (i + 0.5) * voxel). Returns a new Mesh datablock."""
    verts = {}
    faces = []

    def vid(p):
        n = verts.get(p)
        if n is None:
            n = verts[p] = len(verts)
        return n

    for cell in cells:
        for ax in range(3):
            o1, o2 = (ax + 1) % 3, (ax + 2) % 3
            for d in (-1, 1):
                nb = list(cell)
                nb[ax] += d
                if tuple(nb) in cells:
                    continue
                base = list(cell)
                if d == 1:
                    base[ax] += 1
                quad = []
                for b1, b2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = list(base)
                    p[o1] += b1
                    p[o2] += b2
                    quad.append(vid(tuple(p)))
                faces.append(quad)

    mesh = bpy.data.meshes.new("_voxel_draw_blob")
    co = [((p[0] - 0.5) * voxel, (p[1] - 0.5) * voxel, (p[2] - 0.5) * voxel) for p in verts]
    mesh.from_pydata(co, [], faces)
    mesh.validate()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


class SCULPT_OT_voxel_draw(Operator):
    """Toggle the Voxel Draw brush: drag to draw voxels, Ctrl+drag to erase them, Enter/Space fills (Add) or carves (Subtract) the drawn voxels. B toggles Voxel Bridge (click point A, then point B, to connect them with a voxel tube). M switches Add/Subtract, R changes the voxel size, [ and ] the radius, Esc clears/stops"""
    bl_idname = "sculpt.voxel_draw"
    bl_label = "Voxel Draw Brush"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return context.mode == "SCULPT" and ob is not None and ob.type == "MESH"

    def _settings(self, context):
        return context.scene.voxel_draw

    def _color(self, context):
        if self._ctrl:
            return ERASE_COLOR
        if self._bridge:
            return BRIDGE_COLOR
        return SUB_COLOR if self._settings(context).mode == "SUBTRACT" else ADD_COLOR

    def invoke(self, context, event):
        mine = self._settings(context)
        if mine.active:
            mine.active = False
            return {"CANCELLED"}
        for attr in TOOL_SETTINGS:
            other = getattr(context.scene, attr, None)
            if other is not None and other is not mine:
                other.active = False
        mine.active = True
        self._area = context.area
        self._pressed = False
        self._ctrl = False
        self._mouse = None
        self._depth = None          # world point that sets the draw depth when the ray misses
        self._world_radius = 0.0
        self._scale = 1.0
        self._inv = Matrix.Identity(4)
        self._cells = set()         # painted voxel cells, integer (i, j, k) in object space
        self._stamps = []           # local-space stroke points (mask for the remesh afterwards)
        self._stamp_r = 0.0
        self._bridge = False        # bridge mode: click point A, then point B
        self._bridge_a = None       # local-space point A of a pending bridge
        self._bridge_cells = set()  # preview cells of the pending bridge
        self._bridge_over = False   # pending bridge exceeds MAX_BRIDGE_CELLS
        self._count = 0
        self._lines = None
        self._batch = None
        self._resizing = False
        self._rs_x = 0
        self._rs_size = 0.0
        self._shown_voxel = mine.voxel_size
        self._timer = context.window_manager.event_timer_add(0.4, window=context.window)
        self._h3d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cells, (context,), "WINDOW", "POST_VIEW")
        self._h2d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (context,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        self._hint(context)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._h3d, "WINDOW")
        bpy.types.SpaceView3D.draw_handler_remove(self._h2d, "WINDOW")
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        self._settings(context).active = False
        if self._area:
            self._area.tag_redraw()
        return {"FINISHED"}

    def _hint(self, context):
        s = self._settings(context)
        mode = "Add" if s.mode == "ADD" else "Subtract"
        if self._resizing:
            text = f"Voxel size {s.voxel_size:.4f}: move the mouse to change, click/Enter to confirm, Esc to revert"
        elif self._bridge:
            extra = ""
            if self._bridge_over:
                extra = " (bridge too large - reduce radius or increase voxel size)"
            elif self._count > MAX_CELLS:
                extra = " (too many cells - preview truncated)"
            target = "B" if self._bridge_a is not None else "A"
            text = (f"Voxel Bridge ({mode}): {self._count} cells{extra} | click: set point {target} | "
                    f"Ctrl+click: erase bridge | Enter/Space: apply | B: back to draw | M: mode | "
                    f"R: voxel size | [ ]: radius | Esc: cancel/stop")
        else:
            extra = " (too many cells - preview truncated)" if self._count > MAX_CELLS else ""
            text = (f"Voxel Draw ({mode}): {self._count} cells{extra} | drag: draw | Ctrl+drag: erase | "
                    f"Enter/Space: apply | B: bridge | M: mode | R: voxel size | [ ]: radius | Esc: clear/stop")
        context.workspace.status_text_set(text)

    def _draw(self, context):
        if bpy.context.area != self._area or self._mouse is None:
            return
        r = self._settings(context).radius
        _draw_circle(self._mouse, r, (*self._color(context), 0.9), width=2.0)

    def _draw_cells(self, context):
        if bpy.context.area != self._area or self._lines is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        if self._batch is None:
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            self._batch = (shader, batch_for_shader(shader, "LINES", {"pos": self._lines}))
        shader, batch = self._batch
        color = SUB_COLOR if self._settings(context).mode == "SUBTRACT" else ADD_COLOR
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(1.0)
        shader.uniform_float("color", (*color, 0.85))
        batch.draw(shader)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")

    def _clear(self, context):
        self._cells.clear()
        self._stamps = []
        self._stamp_r = 0.0
        self._bridge_a = None
        self._bridge_cells = set()
        self._bridge_over = False
        self._count = 0
        self._lines = None
        self._batch = None
        self._hint(context)
        self._area.tag_redraw()

    def _refresh(self, context):
        """Rebuild the preview lines of the painted cells."""
        s = self._settings(context)
        obj = context.active_object
        self._shown_voxel = s.voxel_size
        shown = self._cells | self._bridge_cells
        self._count = len(shown)
        if not shown or obj is None:
            self._lines = None
            self._batch = None
            self._hint(context)
            self._area.tag_redraw()
            return
        v = s.voxel_size / self._scale
        cells = np.array(list(shown), np.int64)
        if len(cells) > MAX_CELLS:
            cells = cells[::int(math.ceil(len(cells) / MAX_CELLS))]
        pts = cell_edges(cells, v).astype(np.float64)
        mw = np.array(obj.matrix_world)
        self._lines = (pts @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
        self._batch = None
        self._hint(context)
        self._area.tag_redraw()

    def _point(self, context, event):
        """World-space draw point under the mouse: surface hit, or the stored stroke depth."""
        win, rv3d = _window_region(context)
        obj = context.active_object
        if win is None or rv3d is None or obj is None:
            return None, None, None
        coord = (event.mouse_x - win.x, event.mouse_y - win.y)
        origin = view3d_utils.region_2d_to_origin_3d(win, rv3d, coord)
        vec = view3d_utils.region_2d_to_vector_3d(win, rv3d, coord)
        inv = obj.matrix_world.inverted()
        hit, loc, _n, _i = obj.ray_cast(inv @ origin, inv.to_3x3() @ vec)
        if hit:
            self._depth = obj.matrix_world @ loc
            return self._depth, coord, (win, rv3d)
        if self._depth is not None:
            return view3d_utils.region_2d_to_location_3d(win, rv3d, coord, self._depth), coord, (win, rv3d)
        return None, None, None

    def _setup_stroke(self, context, pt, coord, reg):
        """Capture the object transform and the brush world radius at the given draw point."""
        obj = context.active_object
        mw = obj.matrix_world
        self._inv = mw.inverted()
        self._scale = (sum(mw.to_scale()) / 3.0) or 1.0
        win, rv3d = reg
        r_px = self._settings(context).radius
        p0 = view3d_utils.region_2d_to_location_3d(win, rv3d, coord, pt)
        p1 = view3d_utils.region_2d_to_location_3d(win, rv3d, (coord[0] + r_px, coord[1]), pt)
        self._world_radius = (p1 - p0).length

    def _begin(self, context, event):
        pt, coord, reg = self._point(context, event)
        if pt is None:
            return
        self._setup_stroke(context, pt, coord, reg)
        self._pressed = True
        self._stamp(context, event)

    def _bridge_radius(self, voxel):
        """Bridge tube radius in local space: the brush radius, but at least one cell thick."""
        return max(self._world_radius / self._scale, voxel * 0.87)

    def _bridge_click(self, context, event):
        """First click stores point A, second click commits the bridge from A to the click point
        (Ctrl erases the bridge cells instead of adding them)."""
        pt, coord, reg = self._point(context, event)
        if pt is None:
            return
        self._setup_stroke(context, pt, coord, reg)
        s = self._settings(context)
        v = s.voxel_size / self._scale
        r = self._bridge_radius(v)
        c_local = self._inv @ pt
        if self._bridge_a is None:
            self._bridge_a = c_local.copy()
            self._bridge_cells = set(map(tuple, _ball_cells(c_local, r, v).tolist()))
            self._bridge_over = False
            self._refresh(context)
            return
        cells = _segment_cells(self._bridge_a, c_local, r, v)
        if cells is None:
            self._bridge_over = True
            self._hint(context)
            return
        if event.ctrl:
            self._cells.difference_update(cells)
        else:
            self._cells.update(cells)
            a = np.asarray(self._bridge_a, np.float64)
            b = np.asarray(c_local, np.float64)
            n = min(int(math.ceil(float(np.linalg.norm(b - a)) / max(r, v))), 64)
            for t in np.linspace(0.0, 1.0, n + 1):
                self._stamps.append(Vector(a + (b - a) * t))
            self._stamp_r = max(self._stamp_r, r)
            if len(self._stamps) > MAX_STAMPS:
                self._stamps = self._stamps[::2]
        self._bridge_a = None
        self._bridge_cells = set()
        self._bridge_over = False
        self._refresh(context)

    def _bridge_preview(self, context, event):
        """Live preview of the pending bridge from point A to the mouse."""
        pt, _coord, _reg = self._point(context, event)
        if pt is None:
            return
        s = self._settings(context)
        v = s.voxel_size / self._scale
        r = self._bridge_radius(v)
        cells = _segment_cells(self._bridge_a, self._inv @ pt, r, v)
        over = cells is None
        if over != self._bridge_over:
            self._bridge_over = over
            self._hint(context)
        if not over:
            self._bridge_cells = cells
            self._refresh(context)

    def _stamp(self, context, event):
        pt, _coord, _reg = self._point(context, event)
        if pt is None:
            return
        s = self._settings(context)
        v = s.voxel_size / self._scale
        c_local = self._inv @ pt
        r = self._world_radius / self._scale
        cells = _ball_cells(c_local, r, v)
        if event.ctrl:
            self._cells.difference_update(map(tuple, cells.tolist()))
        else:
            self._cells.update(map(tuple, cells.tolist()))
            self._stamps.append(c_local.copy())
            self._stamp_r = max(self._stamp_r, r)
            if len(self._stamps) > MAX_STAMPS:
                self._stamps = self._stamps[::2]
        self._refresh(context)

    def _boolean(self, context, obj, blob, operation):
        """Apply `blob` (a Mesh in obj's local space, consumed here) to obj with a boolean modifier."""
        tmp = bpy.data.objects.new("_voxel_draw_tmp", blob)
        obj.users_collection[0].objects.link(tmp)
        tmp.matrix_world = obj.matrix_world.copy()
        mod = obj.modifiers.new("_voxel_draw_bool", "BOOLEAN")
        mod.operation = operation
        mod.object = tmp
        try:
            mod.solver = "EXACT"
        except Exception:
            pass
        name = mod.name
        try:
            with context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                       selected_editable_objects=[obj]):
                bpy.ops.object.modifier_apply(modifier=name)
        finally:
            leftover = obj.modifiers.get(name)
            if leftover is not None:
                obj.modifiers.remove(leftover)
            bpy.data.objects.remove(tmp)
            if blob.users == 0:
                bpy.data.meshes.remove(blob)

    def _apply(self, context):
        obj = context.active_object
        s = self._settings(context)
        if obj is None or obj.type != "MESH" or not self._cells:
            return
        v = s.voxel_size / self._scale
        centers = list(self._stamps)
        mask_r = self._stamp_r + v * 2.0
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
            blob = cells_to_mesh(set(self._cells), v)
            if s.mode == "SUBTRACT":
                self._boolean(context, obj, blob, "DIFFERENCE")
                if s.remesh_result:
                    if not merge_region(obj, centers, mask_r, v, 0.0, True):
                        self.report({"WARNING"}, "Voxel Size is too small to remesh the result")
            elif s.remesh_result:
                # Append the voxel shell, then voxel-remesh the area: the remesher merges the
                # overlapping volumes into one clean surface.
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                bm.from_mesh(blob)
                bm.to_mesh(obj.data)
                bm.free()
                bpy.data.meshes.remove(blob)
                obj.data.update()
                if not merge_region(obj, centers, mask_r, v, 0.0, True):
                    self.report({"WARNING"}, "Voxel shell added as-is: Voxel Size is too small to merge it")
            else:
                self._boolean(context, obj, blob, "UNION")
        except Exception as ex:
            self.report({"ERROR"}, f"Voxel Draw failed: {ex}")
        finally:
            bpy.ops.object.mode_set(mode="SCULPT")
        self._clear(context)

    def modal(self, context, event):
        s = self._settings(context)
        if not s.active or context.mode != "SCULPT":
            return self._finish(context)

        self._ctrl = event.ctrl
        win, _rv3d = _window_region(context)
        if win is not None:
            self._mouse = (event.mouse_x - win.x, event.mouse_y - win.y)

        if self._cells and not self._resizing and self._shown_voxel != s.voxel_size:
            self._refresh(context)

        pressed = event.value == "PRESS"

        if self._resizing:
            if event.type == "MOUSEMOVE":
                size = self._rs_size * math.exp((event.mouse_x - self._rs_x) * 0.005)
                s.voxel_size = min(max(size, 0.0001), 100.0)
                self._refresh(context)
                return {"RUNNING_MODAL"}
            if pressed and event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "R"}:
                self._resizing = False
                self._hint(context)
                return {"RUNNING_MODAL"}
            if pressed and event.type in {"ESC", "RIGHTMOUSE"}:
                s.voxel_size = self._rs_size
                self._resizing = False
                self._refresh(context)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type == "R" and pressed:
            self._resizing = True
            self._rs_x = event.mouse_x
            self._rs_size = s.voxel_size
            self._hint(context)
            return {"RUNNING_MODAL"}

        if event.type == "M" and pressed:
            s.mode = "SUBTRACT" if s.mode == "ADD" else "ADD"
            self._hint(context)
            self._area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "B" and pressed:
            self._bridge = not self._bridge
            self._bridge_a = None
            self._bridge_cells = set()
            self._bridge_over = False
            self._refresh(context)
            return {"RUNNING_MODAL"}

        if event.type == "ESC" or (event.type == "RIGHTMOUSE" and pressed):
            if self._bridge_a is not None:
                self._bridge_a = None
                self._bridge_cells = set()
                self._bridge_over = False
                self._refresh(context)
                return {"RUNNING_MODAL"}
            if self._cells:
                self._clear(context)
                return {"RUNNING_MODAL"}
            return self._finish(context)

        if event.type in {"RET", "NUMPAD_ENTER", "SPACE"} and pressed and self._cells:
            self._apply(context)
            return {"RUNNING_MODAL"}

        if event.type in {"LEFTBRACKET", "RIGHTBRACKET"} and pressed:
            step = 1.15 if event.type == "RIGHTBRACKET" else 1 / 1.15
            s.radius = max(10, min(500, int(round(s.radius * step))))
            self._area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if pressed:
                if _viewport_region(context, event)[0] is not None and not (event.alt or event.shift):
                    if self._bridge:
                        self._bridge_click(context, event)
                    else:
                        self._begin(context, event)
                    return {"RUNNING_MODAL"}
            elif event.value == "RELEASE" and self._pressed:
                self._pressed = False
                return {"RUNNING_MODAL"}
        elif event.type == "MOUSEMOVE":
            if self._pressed:
                self._stamp(context, event)
            elif self._bridge_a is not None:
                self._bridge_preview(context, event)
            self._area.tag_redraw()
        return {"PASS_THROUGH"}


class VIEW3D_PT_voxel_draw(Panel):
    bl_label = "Voxel Draw Brush"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    bl_context = ".sculpt_mode"

    def draw(self, context):
        s = context.scene.voxel_draw
        col = self.layout.column()
        col.operator("sculpt.voxel_draw", text="Stop Voxel Draw" if s.active else "Start Voxel Draw",
                     icon="PAUSE" if s.active else "SNAP_VOLUME", depress=s.active)
        row = col.row(align=True)
        row.prop(s, "mode", expand=True)
        col.prop(s, "radius")
        col.prop(s, "voxel_size")
        col.prop(s, "remesh_result")
        if s.active:
            box = col.box().column(align=True)
            box.label(text="Drag: draw voxels")
            box.label(text="Ctrl+drag: erase voxels")
            box.label(text="B: bridge (click A, then B)")
            box.label(text="Enter/Space: apply")
            box.label(text="M: Add / Subtract")
            box.label(text="R: voxel size, [ ]: radius")
            box.label(text="Esc: clear / stop")


classes = (VoxelDrawSettings, SCULPT_OT_voxel_draw, VIEW3D_PT_voxel_draw)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.voxel_draw = bpy.props.PointerProperty(type=VoxelDrawSettings)


def unregister():
    del bpy.types.Scene.voxel_draw
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
