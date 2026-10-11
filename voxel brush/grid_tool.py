"""Grid Remesh brush: pick an area, preview the Voxel Remesh resolution grid on it, then remesh
only that area.

Drag over the surface to select the area. The voxel grid (the cells Voxel Remesh would build,
aligned to the object's axes) is drawn on the selected area. R changes the voxel size by moving the
mouse (like Blender's own R in Sculpt Mode), Enter remeshes the area, Esc discards the selection.
"""

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, IntProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras import view3d_utils
from mathutils import Vector

from .merge import _face_mask, merge_region

TOOL_SETTINGS = ("grid_remesh", "voxel_draw")
MAX_SAMPLES = 256
MAX_CELLS = 20000
GRID_COLOR = (0.25, 1.0, 0.45, 0.85)
AREA_COLOR = (0.25, 1.0, 0.45)


class GridRemeshSettings(PropertyGroup):
    active: BoolProperty(name="Active", default=False, options={"SKIP_SAVE"})
    radius: IntProperty(
        name="Radius", default=70, min=10, max=500, subtype="PIXEL",
        description="Brush radius in pixels (also [ and ] while the tool is active)")
    voxel_size: FloatProperty(
        name="Voxel Size", default=0.05, min=0.0001, soft_max=1.0, subtype="DISTANCE",
        description="Voxel Remesh resolution for the selected area (also R while the grid is shown)")
    adaptivity: FloatProperty(
        name="Adaptivity", default=0.0, min=0.0, max=1.0,
        description="Reduces polygon count on flat areas")
    smooth_seam: BoolProperty(
        name="Relax Seam", default=True,
        description="Re-tessellate the border between the remeshed area and the rest of the mesh")


def _viewport_region(context, event):
    """The 3D view WINDOW region under the mouse, or None (e.g. over sidebar/toolbar)."""
    area = context.area
    if area is None or area.type != "VIEW_3D":
        return None, None
    win = None
    for r in area.regions:
        inside = r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height
        if not inside:
            continue
        if r.type == "WINDOW":
            win = r
        elif r.type in {"UI", "TOOLS", "HEADER", "TOOL_HEADER", "NAVIGATION_BAR", "ASSET_SHELF",
                        "ASSET_SHELF_HEADER"}:
            return None, None
    if win is None:
        return None, None
    rv3d = next((s.region_3d for s in area.spaces if s.type == "VIEW_3D"), None)
    return win, rv3d


def _window_region(context):
    area = context.area
    if area is None or area.type != "VIEW_3D":
        return None, None
    win = next((r for r in area.regions if r.type == "WINDOW"), None)
    rv3d = next((s.region_3d for s in area.spaces if s.type == "VIEW_3D"), None)
    return win, rv3d


def _draw_circle(center, radius, color, segments=48, width=1.0):
    import gpu
    from gpu_extras.batch import batch_for_shader

    pts = [(center[0] + math.cos(a) * radius, center[1] + math.sin(a) * radius)
           for a in (i * 2.0 * math.pi / segments for i in range(segments + 1))]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": pts})
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(width)
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def collect_area_tris(mesh, centers, radius):
    """Triangles (T, 3, 3) of the polygons whose centers lie within `radius` of any center."""
    mask = _face_mask(mesh, centers, radius)
    mesh.calc_loop_triangles()
    lt = mesh.loop_triangles
    n = len(lt)
    if n == 0 or not mask.any():
        return np.empty((0, 3, 3), np.float64)
    tv = np.empty(n * 3, np.int32)
    tp = np.empty(n, np.int32)
    lt.foreach_get("vertices", tv)
    lt.foreach_get("polygon_index", tp)
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    tv = tv.reshape(-1, 3)[mask[tp]]
    return co.reshape(-1, 3).astype(np.float64)[tv]


def surface_cells(tris, voxel):
    """Integer indices of the voxels (cell i spans (i - 0.5) * voxel .. (i + 0.5) * voxel) that the
    triangles pass through."""
    if len(tris) == 0:
        return np.empty((0, 3), np.int64)
    edge = np.maximum.reduce([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                              np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                              np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)])
    steps = np.clip(np.ceil(edge / (voxel * 0.7)).astype(int), 1, 24)
    pts = []
    for k in np.unique(steps):
        sel = tris[steps == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        ok = i + j <= k
        a, b = i[ok] / k, j[ok] / k
        c = 1.0 - a - b
        p = (sel[:, 0, None, :] * a[None, :, None] + sel[:, 1, None, :] * b[None, :, None]
             + sel[:, 2, None, :] * c[None, :, None])
        pts.append(p.reshape(-1, 3))
    return np.unique(np.rint(np.concatenate(pts) / voxel).astype(np.int64), axis=0)


def cell_edges(cells, voxel):
    """Unique cube edges of the cells as an (2N, 3) float array of line end points."""
    segs = []
    for ax in range(3):
        o1, o2 = (ax + 1) % 3, (ax + 2) % 3
        for b in (0, 1):
            for c in (0, 1):
                s = cells.copy()
                s[:, o1] += b
                s[:, o2] += c
                segs.append(np.concatenate([s, np.full((len(s), 1), ax)], axis=1))
    segs = np.unique(np.concatenate(segs), axis=0)
    a = (segs[:, :3] - 0.5) * voxel
    b = a.copy()
    b[np.arange(len(segs)), segs[:, 3]] += voxel
    out = np.empty((len(segs) * 2, 3), np.float32)
    out[0::2], out[1::2] = a, b
    return out


class SCULPT_OT_grid_remesh(Operator):
    """Toggle the Grid Remesh brush: drag to select an area, check the voxel grid, press R to change its size and Enter to remesh only that area. Esc discards the selection, [ and ] change the radius"""
    bl_idname = "sculpt.grid_remesh"
    bl_label = "Grid Remesh Brush"
    label = "Grid Remesh"
    color = AREA_COLOR

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return context.mode == "SCULPT" and ob is not None and ob.type == "MESH"

    def _settings(self, context):
        return context.scene.grid_remesh

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
        self._points = []        # world-space surface samples of the current stroke
        self._points_2d = []     # same, in region pixels (for the overlay)
        self._mouse = None
        self._world_radius = 0.0
        self._selected = False
        self._resizing = False
        self._rs_x = 0
        self._rs_size = 0.0
        self._tris = None
        self._centers_obj = []
        self._radius_obj = 0.0
        self._scale = 1.0
        self._lines = None
        self._batch = None
        self._cells = 0
        self._shown_voxel = None
        self._timer = context.window_manager.event_timer_add(0.4, window=context.window)
        self._h3d = bpy.types.SpaceView3D.draw_handler_add(self._draw_grid, (), "WINDOW", "POST_VIEW")
        self._h2d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (context,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        self._hint(context)
        if context.area:
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _hint(self, context):
        s = self._settings(context)
        if self._resizing:
            text = f"Voxel size {s.voxel_size:.4f}: move the mouse to change, click/Enter to confirm, Esc to revert"
        elif self._selected:
            extra = " (too many cells - grid truncated)" if self._cells > MAX_CELLS else ""
            text = (f"Grid Remesh: voxel {s.voxel_size:.4f}, {self._cells} cells{extra} | "
                    "R: voxel size | Enter: remesh area | Esc: discard | drag: new area")
        else:
            text = "Grid Remesh: drag over the surface to select the area | [ ]: radius | Esc: stop"
        context.workspace.status_text_set(text)

    def _finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._h3d, "WINDOW")
        bpy.types.SpaceView3D.draw_handler_remove(self._h2d, "WINDOW")
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        self._settings(context).active = False
        if self._area:
            self._area.tag_redraw()
        return {"FINISHED"}

    def _draw(self, context):
        if bpy.context.area != self._area:
            return
        r = self._settings(context).radius
        if self._selected:
            region, rv3d = bpy.context.region, bpy.context.region_data
            if region is None or rv3d is None:
                return
            right = rv3d.view_matrix.inverted().to_3x3() @ Vector((1.0, 0.0, 0.0))
            for p in self._points:
                c = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
                e = view3d_utils.location_3d_to_region_2d(region, rv3d, p + right * self._world_radius)
                if c is not None and e is not None:
                    _draw_circle(c, (e - c).length, (*self.color, 0.35))
        elif self._mouse is not None:
            _draw_circle(self._mouse, r, (*self.color, 0.9), width=2.0)
            for p in self._points_2d:
                _draw_circle(p, r, (*self.color, 0.25))

    def _draw_grid(self):
        if bpy.context.area != self._area or self._lines is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        if self._batch is None:
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            self._batch = (shader, batch_for_shader(shader, "LINES", {"pos": self._lines}))
        shader, batch = self._batch
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(1.0)
        shader.uniform_float("color", GRID_COLOR)
        batch.draw(shader)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")

    def _clear(self, context):
        self._selected = False
        self._resizing = False
        self._tris = None
        self._lines = None
        self._batch = None
        self._points, self._points_2d = [], []
        self._cells = 0
        self._shown_voxel = None
        self._hint(context)
        self._area.tag_redraw()

    def _refresh(self, context):
        """Rebuild the grid lines for the current voxel size."""
        s = self._settings(context)
        obj = context.active_object
        self._shown_voxel = s.voxel_size
        if self._tris is None or obj is None:
            return
        v = s.voxel_size / self._scale
        cells = surface_cells(self._tris, v)
        self._cells = len(cells)
        if len(cells) > MAX_CELLS:
            cells = cells[::int(math.ceil(len(cells) / MAX_CELLS))]
        if len(cells) == 0:
            self._lines = None
        else:
            pts = cell_edges(cells, v).astype(np.float64)
            mw = np.array(obj.matrix_world)
            self._lines = (pts @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
        self._batch = None
        self._hint(context)
        self._area.tag_redraw()

    def _hit(self, context, event):
        win, rv3d = _window_region(context)
        obj = context.active_object
        if win is None or rv3d is None or obj is None:
            return None
        coord = (event.mouse_x - win.x, event.mouse_y - win.y)
        origin = view3d_utils.region_2d_to_origin_3d(win, rv3d, coord)
        vec = view3d_utils.region_2d_to_vector_3d(win, rv3d, coord)
        inv = obj.matrix_world.inverted()
        hit, loc, _n, _i = obj.ray_cast(inv @ origin, inv.to_3x3() @ vec)
        if not hit:
            return None
        return obj.matrix_world @ loc, coord, win, rv3d

    def _begin(self, context, event):
        res = self._hit(context, event)
        if res is None:
            return
        pt, coord, win, rv3d = res
        r_px = self._settings(context).radius
        p0 = view3d_utils.region_2d_to_location_3d(win, rv3d, coord, pt)
        p1 = view3d_utils.region_2d_to_location_3d(win, rv3d, (coord[0] + r_px, coord[1]), pt)
        self._world_radius = (p1 - p0).length
        self._points = [pt]
        self._points_2d = [coord]
        self._pressed = True

    def _sample(self, context, event, force=False):
        res = self._hit(context, event)
        if res is None:
            return
        pt, coord, _win, _rv3d = res
        if (pt - self._points[-1]).length > self._world_radius * (0.01 if force else 0.3):
            if len(self._points) >= MAX_SAMPLES:
                self._points = self._points[::2]
                self._points_2d = self._points_2d[::2]
            self._points.append(pt)
            self._points_2d.append(coord)

    def _select(self, context):
        """Stroke released: freeze the area and show its grid."""
        obj = context.active_object
        points, self._points_2d = list(self._points), []
        if obj is None or obj.type != "MESH" or not points:
            self._points = []
            return
        mw = obj.matrix_world
        inv = mw.inverted()
        self._scale = (sum(mw.to_scale()) / 3.0) or 1.0
        self._centers_obj = [inv @ p for p in points]
        self._radius_obj = self._world_radius / self._scale
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
            self._tris = collect_area_tris(obj.data, self._centers_obj, self._radius_obj)
        except Exception as ex:
            self.report({"ERROR"}, f"Grid Remesh failed: {ex}")
            self._tris = None
        finally:
            bpy.ops.object.mode_set(mode="SCULPT")
        if self._tris is None or len(self._tris) == 0:
            self._clear(context)
            return
        self._selected = True
        self._refresh(context)

    def _remesh(self, context):
        obj = context.active_object
        s = self._settings(context)
        if obj is None or obj.type != "MESH" or not self._selected:
            return
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
            ok = merge_region(obj, self._centers_obj, self._radius_obj, s.voxel_size / self._scale,
                              s.adaptivity, s.smooth_seam)
            if not ok:
                self.report({"WARNING"}, "Voxel Size is too small for this mesh - increase it")
        except Exception as ex:
            self.report({"ERROR"}, f"Grid Remesh failed: {ex}")
        finally:
            bpy.ops.object.mode_set(mode="SCULPT")
        self._clear(context)

    def modal(self, context, event):
        s = self._settings(context)
        if not s.active or context.mode != "SCULPT":
            return self._finish(context)

        win, _rv3d = _window_region(context)
        if win is not None:
            self._mouse = (event.mouse_x - win.x, event.mouse_y - win.y)

        if self._selected and not self._resizing and self._shown_voxel != s.voxel_size:
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
            if self._selected:
                self._resizing = True
                self._rs_x = event.mouse_x
                self._rs_size = s.voxel_size
                self._hint(context)
            return {"RUNNING_MODAL"}

        if event.type == "ESC" or (event.type == "RIGHTMOUSE" and pressed):
            if self._selected:
                self._clear(context)
                return {"RUNNING_MODAL"}
            return self._finish(context)

        if event.type in {"RET", "NUMPAD_ENTER", "SPACE"} and pressed and self._selected:
            self._remesh(context)
            return {"RUNNING_MODAL"}

        if event.type in {"LEFTBRACKET", "RIGHTBRACKET"} and pressed:
            step = 1.15 if event.type == "RIGHTBRACKET" else 1 / 1.15
            s.radius = max(10, min(500, int(round(s.radius * step))))
            self._area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if pressed:
                if _viewport_region(context, event)[0] is not None and not (event.alt or event.shift):
                    if self._selected:
                        self._clear(context)
                    self._begin(context, event)
                    return {"RUNNING_MODAL"}
            elif event.value == "RELEASE" and self._pressed:
                self._sample(context, event, force=True)
                self._pressed = False
                self._select(context)
                return {"RUNNING_MODAL"}
        elif event.type == "MOUSEMOVE":
            if self._pressed:
                self._sample(context, event)
            self._area.tag_redraw()
        return {"PASS_THROUGH"}


class VIEW3D_PT_grid_remesh(Panel):
    bl_label = "Grid Remesh Brush"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    bl_context = ".sculpt_mode"

    def draw(self, context):
        s = context.scene.grid_remesh
        col = self.layout.column()
        col.operator("sculpt.grid_remesh", text="Stop Grid Remesh" if s.active else "Start Grid Remesh",
                     icon="PAUSE" if s.active else "MESH_GRID", depress=s.active)
        col.prop(s, "radius")
        col.prop(s, "voxel_size")
        col.prop(s, "adaptivity")
        col.prop(s, "smooth_seam")
        if s.active:
            box = col.box().column(align=True)
            box.label(text="Drag: select the area")
            box.label(text="R: change voxel size")
            box.label(text="Enter: remesh the area")
            box.label(text="Esc: discard / stop")


classes = (GridRemeshSettings, SCULPT_OT_grid_remesh, VIEW3D_PT_grid_remesh)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.grid_remesh = bpy.props.PointerProperty(type=GridRemeshSettings)


def unregister():
    del bpy.types.Scene.grid_remesh
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
