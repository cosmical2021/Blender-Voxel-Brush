bl_info = {
    "name": "Grid Remesh + Voxel Draw Brushes",
    "author": "Juha Mukari",
    "version": (0, 6, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Tool (Sculpt Mode)",
    "description": "Grid Remesh Brush (select an area, preview the voxel grid, remesh it) and "
                   "Voxel Draw Brush (draw/erase voxels, then add or subtract them as mesh)",
    "category": "Sculpt",
}

from . import draw_tool, grid_tool


def register():
    grid_tool.register()
    draw_tool.register()


def unregister():
    draw_tool.unregister()
    grid_tool.unregister()
