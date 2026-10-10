"""Shared helpers for the IT'S JUST MATH music video build (Blender 5.2)."""
import math, random
import bpy, bmesh
from mathutils import Vector, Matrix, Euler

FONT_DIR = "C:/Windows/Fonts/"
_fonts = {}


def font(name):
    if name not in _fonts:
        _fonts[name] = bpy.data.fonts.load(FONT_DIR + name, check_existing=True)
    return _fonts[name]


def hexcol(h, a=1.0):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


# ------------------------------------------------------------------ collections / objects
def coll(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    p = parent or bpy.context.scene.collection
    if c.name not in p.children:
        p.children.link(c)
    return c


def link(obj, c):
    for uc in list(obj.users_collection):
        uc.objects.unlink(obj)
    c.objects.link(obj)
    return obj


def empty(name, loc=(0, 0, 0), c=None, parent=None, size=0.2, kind="PLAIN_AXES"):
    o = bpy.data.objects.new(name, None)
    o.empty_display_type = kind
    o.empty_display_size = size
    o.location = loc
    (c or bpy.context.scene.collection).objects.link(o)
    if parent:
        o.parent = parent
    return o


def obj_from_mesh(name, me, c, loc=(0, 0, 0), parent=None, mats=()):
    o = bpy.data.objects.new(name, me)
    o.location = loc
    c.objects.link(o)
    if parent:
        o.parent = parent
    if mats and not me.materials:
        for m in mats:
            me.materials.append(m)
    return o


# ------------------------------------------------------------------ meshes (bmesh, no ops)
def _finish(bm, name, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.shade_smooth()
    return me


def mesh_box(name, sx, sy, sz, bevel=0.0, segs=3, origin="center", smooth=True):
    """Rounded box. origin: 'center' or 'bottom'."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= sx; v.co.y *= sy; v.co.z *= sz
        if origin == "bottom":
            v.co.z += sz / 2
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=min(bevel, min(sx, sy, sz) * 0.49),
                        segments=segs, profile=0.5, affect="EDGES", clamp_overlap=True)
    me = _finish(bm, name, smooth and bevel > 0)
    if bevel > 0:
        _harden(me)
    return me


def _harden(me):
    """Flat big faces, smooth bevels: set custom normals = face normal on large faces."""
    try:
        me.shade_smooth()
        areas = [p.area for p in me.polygons]
        big = max(areas) * 0.2 if areas else 0   # only the real flat sides stay flat
        # vertex-averaged normals weighted by area (approximates "harden normals")
        vn = [Vector((0, 0, 0)) for _ in me.vertices]
        for p in me.polygons:
            w = p.area * (50.0 if p.area >= big else 1.0)
            for vi in p.vertices:
                vn[vi] += p.normal * w
        cn = []
        for p in me.polygons:
            for li, vi in zip(p.loop_indices, p.vertices):
                n = p.normal if p.area >= big else vn[vi].normalized()
                cn.append(n)
        me.normals_split_custom_set(cn)
    except Exception as e:  # never fatal
        print("harden failed", me.name, e)


def mesh_cyl(name, r, h, verts=32, bevel=0.0, segs=2, origin="bottom", r2=None, smooth=True, caps=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=verts,
                          radius1=r, radius2=r if r2 is None else r2, depth=h)
    if origin == "bottom":
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, h / 2))
    if bevel > 0 and caps:
        rim = [e for e in bm.edges if all(abs(abs(v.co.z - (h / 2 if origin == "bottom" else 0)) - h / 2) < 1e-4 for v in e.verts)
               and len(e.link_faces) == 2 and abs(e.link_faces[0].normal.z) != abs(e.link_faces[1].normal.z)]
        bmesh.ops.bevel(bm, geom=rim, offset=bevel, segments=segs, profile=0.5, affect="EDGES", clamp_overlap=True)
    me = _finish(bm, name, smooth)
    if bevel > 0:
        _harden(me)
    return me


def mesh_sphere(name, r, u=32, v=16, scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    for vt in bm.verts:
        vt.co.x *= scale[0]; vt.co.y *= scale[1]; vt.co.z *= scale[2]
    return _finish(bm, name, True)


def mesh_capsule(name, r, length, u=24, v=12):
    """Capsule along Z from 0 to -length (hangs down from origin: good for arms)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    for vt in bm.verts:
        if vt.co.z < 0:
            vt.co.z -= (length - 2 * r)
        vt.co.z -= r
    return _finish(bm, name, True)


def mesh_plane(name, sx, sy):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    for v in bm.verts:
        v.co.x *= sx; v.co.y *= sy
    return _finish(bm, name, False)


def merge_meshes(name, parts):
    """parts: list of (mesh, matrix, material_index). Returns one mesh."""
    bm = bmesh.new()
    for me, mat, mi in parts:
        tmp = bmesh.new()
        tmp.from_mesh(me)
        bmesh.ops.transform(tmp, matrix=mat, verts=tmp.verts)
        for f in tmp.faces:
            f.material_index = mi
        m2 = bpy.data.meshes.new("_tmp")
        tmp.to_mesh(m2); tmp.free()
        bm.from_mesh(m2)
        bpy.data.meshes.remove(m2)
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out); bm.free()
    out.shade_smooth()
    return out


def text_obj(name, body, c, size=1.0, extrude=0.02, bevel=0.0, fontfile="ariblk.ttf", align="CENTER",
             valign="CENTER", mat=None, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), spacing=1.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font(fontfile)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.bevel_resolution = 2
    cu.align_x = align
    cu.align_y = valign
    cu.space_character = spacing
    o = bpy.data.objects.new(name, cu)
    c.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    if parent:
        o.parent = parent
    if mat:
        cu.materials.append(mat)
    return o


# ------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.get(name)
    if m:
        return m, None
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree


def principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, coat=0.0,
               use_obj_color=False, bump=0.0, bump_scale=200.0, spec=0.5, alpha=1.0):
    m, nt = new_mat(name)
    if nt is None:
        return m
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = color if len(color) == 4 else (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    if emit is not None:
        b.inputs["Emission Color"].default_value = emit if len(emit) == 4 else (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    if use_obj_color:
        oi = N.new("ShaderNodeObjectInfo")
        L.new(oi.outputs["Color"], b.inputs["Base Color"])
    if bump > 0:
        tx = N.new("ShaderNodeTexNoise")
        tx.inputs["Scale"].default_value = bump_scale
        bp = N.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        L.new(tx.outputs["Fac"], bp.inputs["Height"])
        L.new(bp.outputs["Normal"], b.inputs["Normal"])
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


def emission(name, color, strength=1.0):
    m, nt = new_mat(name)
    if nt is None:
        return m
    N, L = nt.nodes, nt.links
    for n in list(N):
        if n.type != "OUTPUT_MATERIAL":
            N.remove(n)
    e = N.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = color if len(color) == 4 else (*color, 1)
    e.inputs["Strength"].default_value = strength
    L.new(e.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def node_value_socket(mat, label):
    """Return a node in mat with the given label (for keyframing a default_value)."""
    for n in mat.node_tree.nodes:
        if n.label == label:
            return n
    raise KeyError(label)


# ------------------------------------------------------------------ fcurve baking
def bake(id_data, data_path, values, index=-1, frame_start=1, interp="LINEAR", group=None):
    """Write one keyframe per frame for data_path[index] fast (foreach_set)."""
    if id_data.animation_data is None:
        id_data.animation_data_create()
    ad = id_data.animation_data
    if ad.action is None:
        ad.action = bpy.data.actions.new(id_data.name + "Action")
    act = ad.action
    idx = 0 if index < 0 else index
    fc = None
    # Blender 4.4+ layered actions: use the legacy-compatible helper when available
    try:
        fc = act.fcurve_ensure_for_datablock(id_data, data_path, index=idx)
    except Exception:
        fc = act.fcurves.find(data_path, index=idx) or act.fcurves.new(data_path, index=idx)
    n = len(values)
    fc.keyframe_points.clear()
    fc.keyframe_points.add(n)
    co = [0.0] * (2 * n)
    for i, v in enumerate(values):
        co[2 * i] = frame_start + i
        co[2 * i + 1] = float(v)
    fc.keyframe_points.foreach_set("co", co)
    ip = {"LINEAR": 1, "CONSTANT": 0, "BEZIER": 2}[interp]
    fc.keyframe_points.foreach_set("interpolation", [ip] * n)
    fc.update()
    return fc


def bake_steps(id_data, data_path, keys, index=-1, interp="CONSTANT"):
    """keys: list of (frame, value). Sparse keyframes."""
    if id_data.animation_data is None:
        id_data.animation_data_create()
    ad = id_data.animation_data
    if ad.action is None:
        ad.action = bpy.data.actions.new(id_data.name + "Action")
    act = ad.action
    idx = 0 if index < 0 else index
    try:
        fc = act.fcurve_ensure_for_datablock(id_data, data_path, index=idx)
    except Exception:
        fc = act.fcurves.find(data_path, index=idx) or act.fcurves.new(data_path, index=idx)
    fc.keyframe_points.clear()
    fc.keyframe_points.add(len(keys))
    co = []
    for f, v in keys:
        co += [float(f), float(v)]
    fc.keyframe_points.foreach_set("co", co)
    ip = {"LINEAR": 1, "CONSTANT": 0, "BEZIER": 2}[interp]
    fc.keyframe_points.foreach_set("interpolation", [ip] * len(keys))
    fc.update()
    return fc


def add_driver(obj, path, index, expr, vars_):
    """vars_: list of (name, target_id, data_path)."""
    fc = obj.driver_add(path, index) if index >= 0 else obj.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    for nm, tid, dp in vars_:
        v = d.variables.new()
        v.name = nm
        v.type = "SINGLE_PROP"
        v.targets[0].id = tid
        v.targets[0].data_path = dp
    d.expression = expr
    return fc


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def vlerp(a, b, t):
    return Vector(a).lerp(Vector(b), t)
