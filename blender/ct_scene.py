"""MIDNIGHT: a 30-second clock tower film (Blender 5.2, EEVEE).

blender -b --factory-startup -P ct_scene.py -- [--stills 3,9.5,15] [--scale 50] [--samples 24] [--save]

Shots (seconds):  est 0-7 | face 7-12.5 (flies through the dial) | inside 12.5-18 | macro 18-20.5
                  | count 20.5-22.75 (12:00:00 at t=22) | bell 22.75-25 (strike at 23) | finale 25-30 (2nd strike 27)
"""
import sys, math, random, time
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix, Euler, Quaternion

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ctlib import (hexcol, coll, link, empty, obj_from_mesh, mesh_box, mesh_cyl, mesh_sphere, mesh_plane,
                   merge_meshes, text_obj, new_mat, principled, emission, bake, bake_steps, ease, lerp)

ROOT = HERE.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
    return default


FPS = 24
DUR = 30.0
NF = int(FPS * DUR)
TT = np.arange(NF) / FPS          # time of frame f is (f - 1) / FPS


def F(t):
    return int(round(t * FPS)) + 1


SHOTS = [("est", 0.0), ("face", 7.0), ("inside", 12.5), ("macro", 18.0), ("count", 20.5), ("bell", 22.75), ("finale", 25.0)]
SHOT_END = {n: (SHOTS[i + 1][1] if i + 1 < len(SHOTS) else DUR) for i, (n, _) in enumerate(SHOTS)}
SHOT_START = dict(SHOTS)
MIDNIGHT = 22.0
STRIKES = (23.0, 27.0)
EXTERIOR = {"est", "face", "count", "finale"}

rng = random.Random(7)
D2R = math.pi / 180

# ============================================================================ clock timing
def tick_f(u):
    """step response of a tower clock hand: snaps in ~1 frame with a small overshoot."""
    if u <= 0:
        return 0.0
    return 1 - math.exp(-u / 0.035) * math.cos(2 * math.pi * u / 0.11)


def steps(t):
    """number of escapement steps since t=0 (continuous, with bounce)."""
    if t < 1:
        return 0.0
    k = math.floor(t)
    return (k - 1) + tick_f(t - k)


def sec_angle(t):
    return (38 + steps(t)) * 6.0


def min_angle(t):
    return 354.0 + 6.0 * tick_f(t - MIDNIGHT)


def hour_angle(t):
    return 359.5 + 0.5 * tick_f(t - MIDNIGHT)


def pend_angle(t):
    return 6.5 * D2R * math.cos(math.pi * t)


def pulse(t, times, decay=0.5, attack=0.03):
    v = 0.0
    for t0 in times:
        d = t - t0
        if d >= -attack:
            v = max(v, (1 + d / attack) if d < 0 else math.exp(-d / decay))
    return v


# ============================================================================ render setup
def setup_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, NF
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.resolution_percentage = int(arg("--scale", 100))
    ee = scene.eevee
    ee.taa_render_samples = int(arg("--samples", 24))
    ee.use_raytracing = False
    ee.volumetric_tile_size = str(arg("--voltile", "8"))
    ee.volumetric_samples = int(arg("--volsamples", 48))
    ee.volumetric_start = 0.3
    ee.volumetric_end = 260.0
    ee.volumetric_sample_distribution = 0.85
    ee.use_volumetric_shadows = True
    ee.volumetric_shadow_samples = int(arg("--volshadow", 24))
    ee.shadow_resolution_scale = float(arg("--shadowscale", 1.0))
    ee.bokeh_max_size = 80
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    try:
        scene.render.motion_blur_position = "START"   # never sample across a camera cut
    except Exception as e:
        print("[scene] mb position", e)
    try:
        ee.motion_blur_steps = 1
    except Exception:
        pass
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = float(arg("--exposure", 0.0))
    scene.render.film_transparent = False


def setup_compositor(scene):
    ng = bpy.data.node_groups.new("CT_Comp", "CompositorNodeTree")
    try:
        N, L = ng.nodes, ng.links
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = N.new("CompositorNodeRLayers")
        out = N.new("NodeGroupOutput")
        gl = N.new("CompositorNodeGlare")
        for k, v in (("Type", "Bloom"), ("Quality", "High")):
            try:
                gl.inputs[k].default_value = v
            except Exception as e:
                print("[scene] glare", k, e)
        for nm, val in (("Threshold", 1.2), ("Strength", 0.4), ("Size", 0.7), ("Saturation", 1.0)):
            try:
                gl.inputs[nm].default_value = val
            except Exception as e:
                print("[scene] glare", nm, e)
        L.new(rl.outputs["Image"], gl.inputs["Image"])
        em = N.new("CompositorNodeEllipseMask")
        em.inputs["Size"].default_value = (1.1, 0.95)
        bl = N.new("CompositorNodeBlur")
        bl.inputs["Size"].default_value = (300, 300)
        L.new(em.outputs["Mask"], bl.inputs["Image"])
        mr = N.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.45
        mr.inputs["To Max"].default_value = 1.0
        L.new(bl.outputs["Image"], mr.inputs["Value"])
        mul = N.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        L.new(gl.outputs["Image"], mul.inputs[6])
        L.new(mr.outputs["Result"], mul.inputs[7])
        L.new(mul.outputs[2], out.inputs[0])
        scene.compositing_node_group = ng
        print("[scene] compositor ok")
    except Exception as e:
        print("[scene] compositor skipped:", repr(e))
        bpy.data.node_groups.remove(ng)


MOON_POS = Vector((-150, 470, 215))


def setup_world(scene):
    w = bpy.data.worlds.new("Night")
    scene.world = w
    w.use_nodes = True
    N, L = w.node_tree.nodes, w.node_tree.links
    bg = N["Background"]
    tc = N.new("ShaderNodeTexCoord")
    dirv = tc.outputs["Generated"]
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(dirv, sep.inputs[0])
    # sky gradient by elevation
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.05
    mr.inputs["From Max"].default_value = 0.65
    L.new(sep.outputs["Z"], mr.inputs["Value"])
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = hexcol("#4A4A5C")
    cr.elements[1].position = 1.0; cr.elements[1].color = hexcol("#070D1C")
    e = cr.elements.new(0.1); e.color = hexcol("#27324F")
    e = cr.elements.new(0.38); e.color = hexcol("#121C36")
    L.new(mr.outputs["Result"], ramp.inputs["Fac"])
    # moon halo
    mdir = (MOON_POS / MOON_POS.length)
    dot = N.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = tuple(mdir)
    L.new(dirv, dot.inputs[0])
    mx = N.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; mx.inputs[1].default_value = 0.0
    L.new(dot.outputs["Value"], mx.inputs[0])
    p1 = N.new("ShaderNodeMath"); p1.operation = "POWER"; p1.inputs[1].default_value = 260.0
    p2 = N.new("ShaderNodeMath"); p2.operation = "POWER"; p2.inputs[1].default_value = 22.0
    L.new(mx.outputs[0], p1.inputs[0]); L.new(mx.outputs[0], p2.inputs[0])
    h1 = N.new("ShaderNodeMath"); h1.operation = "MULTIPLY"; h1.inputs[1].default_value = 0.55
    h2 = N.new("ShaderNodeMath"); h2.operation = "MULTIPLY"; h2.inputs[1].default_value = 0.1
    L.new(p1.outputs[0], h1.inputs[0]); L.new(p2.outputs[0], h2.inputs[0])
    halo = N.new("ShaderNodeMath"); halo.operation = "ADD"
    L.new(h1.outputs[0], halo.inputs[0]); L.new(h2.outputs[0], halo.inputs[1])
    # clouds: stretched noise, lit near the moon
    cmap = N.new("ShaderNodeMapping"); cmap.inputs["Scale"].default_value = (1.6, 1.6, 5.0)
    cmap.name = "CLOUDMAP"
    L.new(dirv, cmap.inputs["Vector"])
    cn = N.new("ShaderNodeTexNoise"); cn.inputs["Scale"].default_value = 2.2; cn.inputs["Detail"].default_value = 9
    cn.inputs["Roughness"].default_value = 0.62
    L.new(cmap.outputs[0], cn.inputs["Vector"])
    cramp = N.new("ShaderNodeValToRGB")
    cramp.color_ramp.elements[0].position = 0.5; cramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    cramp.color_ramp.elements[1].position = 0.72; cramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    L.new(cn.outputs["Fac"], cramp.inputs["Fac"])
    hz = N.new("ShaderNodeMapRange")   # clouds only above the horizon, thinning near zenith
    hz.inputs["From Min"].default_value = 0.0; hz.inputs["From Max"].default_value = 0.1
    L.new(sep.outputs["Z"], hz.inputs["Value"])
    cm = N.new("ShaderNodeMath"); cm.operation = "MULTIPLY"
    L.new(cramp.outputs["Color"], cm.inputs[0]); L.new(hz.outputs["Result"], cm.inputs[1])
    cmul = N.new("ShaderNodeMath"); cmul.operation = "MULTIPLY"; cmul.inputs[1].default_value = 0.85
    L.new(cm.outputs[0], cmul.inputs[0])
    lit = N.new("ShaderNodeMath"); lit.operation = "POWER"; lit.inputs[1].default_value = 12.0
    L.new(mx.outputs[0], lit.inputs[0])
    ccol = N.new("ShaderNodeMix"); ccol.data_type = "RGBA"
    ccol.inputs[6].default_value = hexcol("#1A2134"); ccol.inputs[7].default_value = hexcol("#8E9BB8")
    L.new(lit.outputs[0], ccol.inputs["Factor"])
    sky_c = N.new("ShaderNodeMix"); sky_c.data_type = "RGBA"
    L.new(cmul.outputs[0], sky_c.inputs["Factor"])
    L.new(ramp.outputs["Color"], sky_c.inputs[6]); L.new(ccol.outputs[2], sky_c.inputs[7])
    # stars (hidden behind clouds)
    vor = N.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 420.0
    L.new(dirv, vor.inputs["Vector"])
    st = N.new("ShaderNodeMath"); st.operation = "LESS_THAN"; st.inputs[1].default_value = 0.07
    L.new(vor.outputs["Distance"], st.inputs[0])
    wn = N.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"
    L.new(vor.outputs["Position"], wn.inputs["Vector"])
    sb = N.new("ShaderNodeMath"); sb.operation = "POWER"; sb.inputs[1].default_value = 6.0
    L.new(wn.outputs["Value"], sb.inputs[0])
    s1 = N.new("ShaderNodeMath"); s1.operation = "MULTIPLY"
    L.new(st.outputs[0], s1.inputs[0]); L.new(sb.outputs[0], s1.inputs[1])
    nc = N.new("ShaderNodeMath"); nc.operation = "SUBTRACT"; nc.inputs[0].default_value = 1.0
    L.new(cmul.outputs[0], nc.inputs[1])
    s2 = N.new("ShaderNodeMath"); s2.operation = "MULTIPLY"
    L.new(s1.outputs[0], s2.inputs[0]); L.new(nc.outputs[0], s2.inputs[1])
    s3 = N.new("ShaderNodeMath"); s3.operation = "MULTIPLY"
    L.new(s2.outputs[0], s3.inputs[0]); L.new(hz.outputs["Result"], s3.inputs[1])
    s4 = N.new("ShaderNodeMath"); s4.operation = "MULTIPLY"; s4.inputs[1].default_value = 3.0
    L.new(s3.outputs[0], s4.inputs[0])
    add1 = N.new("ShaderNodeMath"); add1.operation = "ADD"
    L.new(s4.outputs[0], add1.inputs[0]); L.new(halo.outputs[0], add1.inputs[1])
    glow = N.new("ShaderNodeMix"); glow.data_type = "RGBA"; glow.blend_type = "ADD"
    glow.inputs["Factor"].default_value = 1.0
    L.new(sky_c.outputs[2], glow.inputs[6])
    cmb = N.new("ShaderNodeCombineColor")
    L.new(add1.outputs[0], cmb.inputs[0]); L.new(add1.outputs[0], cmb.inputs[1]); L.new(add1.outputs[0], cmb.inputs[2])
    L.new(cmb.outputs[0], glow.inputs[7])
    L.new(glow.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    # thin moonlit haze over everything
    vs = N.new("ShaderNodeVolumePrincipled")
    vs.inputs["Color"].default_value = hexcol("#9AA9C6")
    vs.inputs["Density"].default_value = float(arg("--fog", 0.0016))
    vs.inputs["Anisotropy"].default_value = 0.55
    L.new(vs.outputs[0], N["World Output"].inputs["Volume"])


# ============================================================================ materials
def node(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def math_node(nt, op, a=None, b=None, va=None, vb=None):
    n = nt.nodes.new("ShaderNodeMath"); n.operation = op
    if a is not None:
        nt.links.new(a, n.inputs[0])
    elif va is not None:
        n.inputs[0].default_value = va
    if b is not None:
        nt.links.new(b, n.inputs[1])
    elif vb is not None:
        n.inputs[1].default_value = vb
    return n.outputs[0]


def wall_uv(nt, mode="wall"):
    """(x+y, z) for vertical faces, (x, y) for floors, in world space."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ")
    if mode == "wall":
        nt.links.new(math_node(nt, "ADD", sep.outputs["X"], sep.outputs["Y"]), cmb.inputs[0])
        nt.links.new(sep.outputs["Z"], cmb.inputs[1])
    else:
        nt.links.new(sep.outputs["X"], cmb.inputs[0])
        nt.links.new(sep.outputs["Y"], cmb.inputs[1])
    return cmb.outputs[0], geo


def stone_mat(name, c1, c2, mortar, bw=0.92, rh=0.42, rough=0.86, bump=0.45):
    m, nt = new_mat(name)
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    uv, geo = wall_uv(nt)
    br = node(nt, "ShaderNodeTexBrick", Scale=1.0)
    br.inputs["Color1"].default_value = hexcol(c1)
    br.inputs["Color2"].default_value = hexcol(c2)
    br.inputs["Mortar"].default_value = hexcol(mortar)
    br.inputs["Mortar Size"].default_value = 0.016
    br.inputs["Mortar Smooth"].default_value = 0.3
    br.inputs["Brick Width"].default_value = bw
    br.inputs["Row Height"].default_value = rh
    br.inputs["Bias"].default_value = 0.0
    L.new(uv, br.inputs["Vector"])
    nz = node(nt, "ShaderNodeTexNoise", Scale=1.3, Detail=10.0, Roughness=0.6)
    L.new(geo.outputs["Position"], nz.inputs["Vector"])
    # vertical weathering streaks
    sm = node(nt, "ShaderNodeMapping"); sm.inputs["Scale"].default_value = (5.0, 5.0, 0.35)
    L.new(geo.outputs["Position"], sm.inputs["Vector"])
    sn = node(nt, "ShaderNodeTexNoise", Scale=1.6, Detail=4.0)
    L.new(sm.outputs[0], sn.inputs["Vector"])
    v1 = math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], None, vb=0.7)
    N_ = v1.node; N_.inputs[1].default_value = 0.8; N_.inputs[2].default_value = 0.6
    streak = math_node(nt, "MULTIPLY_ADD", sn.outputs["Fac"], None, vb=0.6)
    sN = streak.node; sN.inputs[1].default_value = 0.7; sN.inputs[2].default_value = 0.62
    mulc = N.new("ShaderNodeMix"); mulc.data_type = "RGBA"; mulc.blend_type = "MULTIPLY"
    mulc.inputs["Factor"].default_value = 1.0
    L.new(br.outputs["Color"], mulc.inputs[6])
    cc = N.new("ShaderNodeCombineColor")
    vv = math_node(nt, "MULTIPLY", v1, streak)
    for i in range(3):
        L.new(vv, cc.inputs[i])
    L.new(cc.outputs[0], mulc.inputs[7])
    L.new(mulc.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    inv = math_node(nt, "SUBTRACT", None, br.outputs["Fac"], va=1.0)
    hgt = math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], None, vb=0.25)
    hN = hgt.node; hN.inputs[1].default_value = 0.25; L.new(inv, hN.inputs[2])
    bp = node(nt, "ShaderNodeBump", Strength=bump, Distance=0.03)
    L.new(hgt, bp.inputs["Height"])
    L.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def metal_mat(name, col, rough=0.3, var=0.15, bump=0.0):
    m, nt = new_mat(name)
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = hexcol(col)
    b.inputs["Metallic"].default_value = 1.0
    geo = N.new("ShaderNodeNewGeometry")
    nz = node(nt, "ShaderNodeTexNoise", Scale=6.0, Detail=6.0)
    L.new(geo.outputs["Position"], nz.inputs["Vector"])
    r = math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], None, vb=var * 2)
    r.node.inputs[2].default_value = rough - var
    L.new(r, b.inputs["Roughness"])
    if bump > 0:
        bp = node(nt, "ShaderNodeBump", Strength=bump, Distance=0.01)
        L.new(nz.outputs["Fac"], bp.inputs["Height"])
        L.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def bronze_mat():
    m, nt = new_mat("Bronze")
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    geo = N.new("ShaderNodeNewGeometry")
    nz = node(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=8.0, Roughness=0.65)
    L.new(geo.outputs["Position"], nz.inputs["Vector"])
    rp = N.new("ShaderNodeValToRGB")
    rp.color_ramp.elements[0].position = 0.45; rp.color_ramp.elements[0].color = hexcol("#8A5A30")
    rp.color_ramp.elements[1].position = 0.7; rp.color_ramp.elements[1].color = hexcol("#3D6656")
    L.new(nz.outputs["Fac"], rp.inputs["Fac"])
    L.new(rp.outputs["Color"], b.inputs["Base Color"])
    mt = math_node(nt, "MAP_RANGE" if False else "SUBTRACT", None, nz.outputs["Fac"], va=1.35)
    mc = math_node(nt, "MINIMUM", mt, None, vb=1.0)
    L.new(mc, b.inputs["Metallic"])
    b.inputs["Roughness"].default_value = 0.38
    return m


def dial_mat():
    m, nt = new_mat("DialGlass")
    N, L = nt.nodes, nt.links
    for n in list(N):
        if n.type != "OUTPUT_MATERIAL":
            N.remove(n)
    tc = N.new("ShaderNodeTexCoord")
    ln = N.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
    L.new(tc.outputs["Object"], ln.inputs[0])
    rad = N.new("ShaderNodeMapRange")
    rad.inputs["From Min"].default_value = 0.0; rad.inputs["From Max"].default_value = 3.5
    rad.inputs["To Min"].default_value = 1.15; rad.inputs["To Max"].default_value = 0.72
    L.new(ln.outputs["Value"], rad.inputs["Value"])
    nz = node(nt, "ShaderNodeTexNoise", Scale=2.5, Detail=6.0)
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    nv = math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], None, vb=0.3)
    nv.node.inputs[2].default_value = 0.85
    strength = N.new("ShaderNodeValue"); strength.name = strength.label = "DIALSTR"
    strength.outputs[0].default_value = 5.0
    s = math_node(nt, "MULTIPLY", rad.outputs["Result"], nv)
    s2 = math_node(nt, "MULTIPLY", s, strength.outputs[0])
    em = N.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = hexcol("#FFE0A6")
    L.new(s2, em.inputs["Strength"])
    # seen from inside the tower the opal glass is mostly see-through: the numerals and hands
    # silhouette against the backing glow behind them
    # room side = faces whose object-space normal points along local -Z (local +Z is outward)
    nsep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Normal"], nsep.inputs[0])
    room_side = math_node(nt, "LESS_THAN", nsep.outputs["Z"], None, vb=-0.5)
    tr = N.new("ShaderNodeBsdfTransparent")
    inner = N.new("ShaderNodeMixShader"); inner.inputs[0].default_value = 0.25
    L.new(tr.outputs[0], inner.inputs[1]); L.new(em.outputs[0], inner.inputs[2])
    side = N.new("ShaderNodeMixShader")
    L.new(room_side, side.inputs[0])
    L.new(em.outputs[0], side.inputs[1]); L.new(inner.outputs[0], side.inputs[2])
    L.new(side.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def window_mat(name, col, strength, flicker=False):
    m, nt = new_mat(name)
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = hexcol("#1A1410")
    b.inputs["Roughness"].default_value = 0.2
    b.inputs["Emission Color"].default_value = hexcol(col)
    tc = N.new("ShaderNodeTexCoord")
    nz = node(nt, "ShaderNodeTexNoise", Scale=1.5, Detail=3.0)
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    s = math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], None, vb=strength)
    s.node.inputs[2].default_value = strength * 0.5
    L.new(s, b.inputs["Emission Strength"])
    return m


def city_mat():
    m, nt = new_mat("CityFacade")
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    uv, geo = wall_uv(nt)
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(uv, sep.inputs[0])
    u, v = sep.outputs["X"], sep.outputs["Y"]
    fu = math_node(nt, "FRACT", math_node(nt, "DIVIDE", u, None, vb=3.1))
    fv = math_node(nt, "FRACT", math_node(nt, "DIVIDE", v, None, vb=3.3))
    mx = math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", fu, None, vb=0.3), math_node(nt, "LESS_THAN", fu, None, vb=0.68))
    my = math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", fv, None, vb=0.32), math_node(nt, "LESS_THAN", fv, None, vb=0.78))
    mask = math_node(nt, "MULTIPLY", mx, my)
    mask = math_node(nt, "MULTIPLY", mask, math_node(nt, "GREATER_THAN", v, None, vb=2.0))
    oi = N.new("ShaderNodeObjectInfo")
    cell = N.new("ShaderNodeCombineXYZ")
    L.new(math_node(nt, "FLOOR", math_node(nt, "DIVIDE", u, None, vb=3.1)), cell.inputs[0])
    L.new(math_node(nt, "FLOOR", math_node(nt, "DIVIDE", v, None, vb=3.3)), cell.inputs[1])
    L.new(math_node(nt, "MULTIPLY", oi.outputs["Random"], None, vb=97.0), cell.inputs[2])
    wn = N.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"
    L.new(cell.outputs[0], wn.inputs["Vector"])
    lit = math_node(nt, "GREATER_THAN", wn.outputs["Value"], None, vb=0.66)
    em = math_node(nt, "MULTIPLY", mask, lit)
    warm = N.new("ShaderNodeMix"); warm.data_type = "RGBA"
    warm.inputs[6].default_value = hexcol("#FFB060"); warm.inputs[7].default_value = hexcol("#FFD9A0")
    L.new(wn.outputs["Color"], warm.inputs["Factor"])
    L.new(warm.outputs[2], b.inputs["Emission Color"])
    L.new(math_node(nt, "MULTIPLY", em, None, vb=3.5), b.inputs["Emission Strength"])
    pal = N.new("ShaderNodeValToRGB")
    cols = ["#6B5E52", "#7C6B5A", "#5A5650", "#8A7A66", "#6E6458"]
    cr = pal.color_ramp
    cr.interpolation = "CONSTANT"
    cr.elements[0].position = 0.0; cr.elements[0].color = hexcol(cols[0])
    cr.elements[1].position = 0.8; cr.elements[1].color = hexcol(cols[4])
    for i, c in enumerate(cols[1:4]):
        e = cr.elements.new(0.2 * (i + 1)); e.color = hexcol(c)
    L.new(oi.outputs["Random"], pal.inputs["Fac"])
    base = N.new("ShaderNodeMix"); base.data_type = "RGBA"
    L.new(mask, base.inputs["Factor"])
    L.new(pal.outputs["Color"], base.inputs[6]); base.inputs[7].default_value = hexcol("#0E0F12")
    L.new(base.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    return m


def volume_mat(name, density, color="#C9B79A", aniso=0.6):
    m, nt = new_mat(name)
    N, L = nt.nodes, nt.links
    for n in list(N):
        if n.type != "OUTPUT_MATERIAL":
            N.remove(n)
    v = N.new("ShaderNodeVolumePrincipled")
    v.inputs["Density"].default_value = density
    v.inputs["Color"].default_value = hexcol(color)
    v.inputs["Anisotropy"].default_value = aniso
    L.new(v.outputs[0], N["Material Output"].inputs["Volume"])
    return m


def moon_mat():
    m, nt = new_mat("Moon")
    N, L = nt.nodes, nt.links
    for n in list(N):
        if n.type != "OUTPUT_MATERIAL":
            N.remove(n)
    tc = N.new("ShaderNodeTexCoord")
    nz = node(nt, "ShaderNodeTexNoise", Scale=0.18, Detail=8.0, Roughness=0.7)
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    rp = N.new("ShaderNodeValToRGB")
    rp.color_ramp.elements[0].position = 0.42; rp.color_ramp.elements[0].color = hexcol("#C3C9D4")
    rp.color_ramp.elements[1].position = 0.6; rp.color_ramp.elements[1].color = hexcol("#FBF8F0")
    L.new(nz.outputs["Fac"], rp.inputs["Fac"])
    em = N.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 14.0
    L.new(rp.outputs["Color"], em.inputs["Color"])
    L.new(em.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def wood_mat(name, floor=False):
    m, nt = new_mat(name)
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    uv, geo = wall_uv(nt, "floor" if floor else "wall")
    br = node(nt, "ShaderNodeTexBrick", Scale=1.0)
    br.inputs["Color1"].default_value = hexcol("#4A3020")
    br.inputs["Color2"].default_value = hexcol("#5E3D27")
    br.inputs["Mortar"].default_value = hexcol("#1A100A")
    br.inputs["Mortar Size"].default_value = 0.006
    br.inputs["Brick Width"].default_value = 2.6 if floor else 6.0
    br.inputs["Row Height"].default_value = 0.24 if floor else 0.32
    L.new(uv, br.inputs["Vector"])
    gm = node(nt, "ShaderNodeMapping"); gm.inputs["Scale"].default_value = (0.4, 18.0, 18.0) if floor else (18.0, 18.0, 0.5)
    L.new(geo.outputs["Position"], gm.inputs["Vector"])
    gn = node(nt, "ShaderNodeTexNoise", Scale=2.0, Detail=6.0)
    L.new(gm.outputs[0], gn.inputs["Vector"])
    mul = N.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 0.6
    L.new(br.outputs["Color"], mul.inputs[6]); L.new(gn.outputs["Color"], mul.inputs[7])
    L.new(mul.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.7
    return m


def build_materials():
    M = {}
    M["stone"] = stone_mat("Stone", "#8E887E", "#837C70", "#4E4A44", bw=0.64, rh=0.32, bump=0.3)
    M["stone2"] = stone_mat("StoneTrim", "#A39C90", "#9A9286", "#5E5850", bw=1.3, rh=0.5, bump=0.2)
    M["slate"] = stone_mat("Slate", "#3A404C", "#454B58", "#15171B", bw=0.36, rh=0.2, rough=0.45, bump=0.6)
    M["gold"] = principled("Gilt", hexcol("#D9A84E"), rough=0.32, metal=0.65, coat=0.3)
    M["iron"] = metal_mat("Iron", "#17161A", rough=0.45, var=0.15, bump=0.2)
    M["brass"] = metal_mat("Brass", "#C1913F", rough=0.3, var=0.12, bump=0.05)
    M["steel"] = metal_mat("Steel", "#8C8F94", rough=0.32, var=0.1)
    M["bronze"] = bronze_mat()
    M["red"] = principled("HandRed", hexcol("#8E1414"), rough=0.35, metal=0.3, coat=0.6)
    M["dial"] = dial_mat()
    M["win_lit"] = window_mat("WindowLit", "#FFB566", 5.0)
    M["win_dark"] = principled("WindowDark", hexcol("#0D0F14"), rough=0.15)
    M["city"] = city_mat()
    M["roof"] = stone_mat("CityRoof", "#3A2E2A", "#2F2A2C", "#141010", bw=0.3, rh=0.18, rough=0.6, bump=0.4)
    M["ground"] = stone_mat("Cobble", "#3B3833", "#33302C", "#1A1816", bw=0.28, rh=0.18, rough=0.55, bump=0.5)
    M["moon"] = moon_mat()
    M["wood"] = wood_mat("WoodBeam")
    M["floor"] = wood_mat("WoodFloor", floor=True)
    M["dust"] = principled("Dust", hexcol("#FFF3DA"), rough=1.0, emit=hexcol("#FFE2B0"), emit_strength=0.6)
    M["bird"] = principled("Bird", hexcol("#0A0A0C"), rough=0.9)
    M["bulb"] = emission("Bulb", hexcol("#FFC27A"), 25.0)
    M["vol_room"] = volume_mat("RoomHaze", float(arg("--roomfog", 0.022)), "#D8C6A6", 0.6)
    M["vol_bell"] = volume_mat("BelfryHaze", 0.012, "#C4B69E", 0.5)
    # warm city glow hanging low between the roofs
    m, nt = new_mat("CityGlow")
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    v = nt.nodes.new("ShaderNodeVolumePrincipled")
    v.inputs["Density"].default_value = 0.003
    v.inputs["Color"].default_value = hexcol("#B9A38A")
    v.inputs["Emission Color"].default_value = hexcol("#FF9A52")
    v.inputs["Emission Strength"].default_value = float(arg("--cityglow", 0.0003))
    nt.links.new(v.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    M["cityglow"] = m
    return M


# ============================================================================ mesh builders
def extrude_poly(name, pts, depth, smooth=False):
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0.0)) for x, y in pts]
    f = bm.faces.new(vs)
    r = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv, vec=(0, 0, depth))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    return me


def strip_mesh(name, outer, inner, depth, closed=False):
    bm = bmesh.new()
    vo = [bm.verts.new((x, y, 0.0)) for x, y in outer]
    vi = [bm.verts.new((x, y, 0.0)) for x, y in inner]
    n = len(outer)
    faces = []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        try:
            faces.append(bm.faces.new((vo[i], vo[j], vi[j], vi[i])))
        except ValueError:
            pass
    r = bmesh.ops.extrude_face_region(bm, geom=faces)
    nv = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv, vec=(0, 0, depth))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    return me


def circle_pts(r, n, a0=0.0):
    return [(r * math.sin(a0 + 2 * math.pi * i / n), r * math.cos(a0 + 2 * math.pi * i / n)) for i in range(n)]


def ring_mesh(name, r_out, r_in, depth, n=96):
    return strip_mesh(name, circle_pts(r_out, n), circle_pts(r_in, n), depth, closed=True)


def lancet(a, h_side, rise, n_arc=10):
    """U-shaped outline of a pointed arch opening: (-a,0) up, over the apex, down to (a,0)."""
    c = max(0.0, (rise * rise - a * a) / (2 * a))
    R = a + c
    pts = [(-a, 0.0), (-a, h_side)] if h_side > 1e-6 else [(-a, 0.0)]
    phi_apex = math.acos(-c / R)
    for i in range(1, n_arc + 1):
        ph = math.pi + (phi_apex - math.pi) * i / n_arc
        pts.append((c + R * math.cos(ph), h_side + R * math.sin(ph)))
    right = [(-x, y) for x, y in reversed(pts[:-1])]
    return pts + right


def slab_with_hole(name, w, h, cx, cy, r, depth, n=96):
    """Rectangle (-w/2..w/2, 0..h) with a circular hole, extruded along +Z."""
    corners = [(w / 2, h), (w / 2, 0), (-w / 2, 0), (-w / 2, h)]
    angs = [2 * math.pi * i / n for i in range(n)]
    for qx, qy in corners:   # clockwise-from-up angle of each corner
        angs.append(math.atan2(qx - cx, qy - cy) % (2 * math.pi))
    angs.sort()
    outer, inner = [], []
    for a in angs:
        dx, dy = math.sin(a), math.cos(a)
        ts = []
        if dx > 1e-9: ts.append((w / 2 - cx) / dx)
        if dx < -1e-9: ts.append((-w / 2 - cx) / dx)
        if dy > 1e-9: ts.append((h - cy) / dy)
        if dy < -1e-9: ts.append((0 - cy) / dy)
        t = min(ts)
        outer.append((cx + dx * t, cy + dy * t))
        inner.append((cx + dx * r, cy + dy * r))
    return strip_mesh(name, outer, inner, depth, closed=True)


def gear_mesh(name, n, module, thick, spokes=6, escape=False, hub=None):
    r = module * n / 2
    ra, rr = r + module, r - 1.25 * module
    outer = []
    if escape:
        for i in range(n):
            a = 2 * math.pi * i / n
            outer += [(rr * math.sin(a), rr * math.cos(a)),
                      ((ra + 0.4 * module) * math.sin(a + 0.55 * 2 * math.pi / n), (ra + 0.4 * module) * math.cos(a + 0.55 * 2 * math.pi / n)),
                      ((ra + 0.25 * module) * math.sin(a + 0.62 * 2 * math.pi / n), (ra + 0.25 * module) * math.cos(a + 0.62 * 2 * math.pi / n)),
                      (rr * math.sin(a + 0.66 * 2 * math.pi / n), rr * math.cos(a + 0.66 * 2 * math.pi / n))]
    else:
        hw = math.pi / (2 * n)
        for i in range(n):
            a = 2 * math.pi * i / n
            for da, rad in ((-math.pi / n, rr), (-hw * 1.25, rr), (-hw * 0.62, ra), (hw * 0.62, ra), (hw * 1.25, rr)):
                outer.append((rad * math.sin(a + da), rad * math.cos(a + da)))
    rim_in = rr - max(module * 2.2, r * 0.12)
    inner = [(rim_in * x / math.hypot(x, y), rim_in * y / math.hypot(x, y)) for x, y in outer]
    rim = strip_mesh(name + "_rim", outer, inner, thick, closed=True)
    parts = [(rim, Matrix.Identity(4), 0)]
    hub_r = hub or max(0.06, r * 0.16)
    hb = mesh_cyl(name + "_hub", hub_r, thick * 1.8, verts=24, bevel=0.0, origin="center")
    parts.append((hb, Matrix.Translation((0, 0, thick / 2)), 0))
    if spokes:
        L_ = rim_in - hub_r * 0.6
        sp = mesh_box(name + "_sp", max(0.03, r * 0.07), L_, thick * 0.7, bevel=0.0)
        for i in range(spokes):
            a = 2 * math.pi * i / spokes
            mat = Matrix.Rotation(-a, 4, "Z") @ Matrix.Translation((0, hub_r * 0.6 + L_ / 2, thick / 2))
            parts.append((sp, mat, 0))
    me = merge_meshes(name, parts)
    for p in {id(p): p for p, _, _ in parts}.values():
        bpy.data.meshes.remove(p)
    me.shade_flat()
    return me, r


def lathe_mesh(name, profile, steps=64):
    bm = bmesh.new()
    vs = [bm.verts.new((r, 0.0, z)) for r, z in profile]
    es = [bm.edges.new((vs[i], vs[i + 1])) for i in range(len(vs) - 1)]
    bmesh.ops.spin(bm, geom=vs + es, cent=(0, 0, 0), axis=(0, 0, 1), angle=2 * math.pi, steps=steps, use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.shade_smooth()
    return me


def pyramid_mesh(name, w, h, sides=4, top_w=0.0):
    bm = bmesh.new()
    rad = w / 2 / math.cos(math.pi / sides)
    a0 = math.pi / sides
    base = [bm.verts.new((rad * math.cos(a0 + 2 * math.pi * i / sides), rad * math.sin(a0 + 2 * math.pi * i / sides), 0)) for i in range(sides)]
    if top_w > 0:
        rt = top_w / 2 / math.cos(math.pi / sides)
        top = [bm.verts.new((rt * math.cos(a0 + 2 * math.pi * i / sides), rt * math.sin(a0 + 2 * math.pi * i / sides), h)) for i in range(sides)]
        for i in range(sides):
            j = (i + 1) % sides
            bm.faces.new((base[i], base[j], top[j], top[i]))
        bm.faces.new(top)
    else:
        apex = bm.verts.new((0, 0, h))
        for i in range(sides):
            bm.faces.new((base[i], base[(i + 1) % sides], apex))
    bm.faces.new(list(reversed(base)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    return me


def mirror_profile(half):
    """half: right-side (x>=0) points from bottom to top; returns a closed outline."""
    left = [(-x, y) for x, y in reversed(half) if x > 1e-6]
    return half + left


def ob(name, me, c, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), mat=None, parent=None):
    o = bpy.data.objects.new(name, me)
    c.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    o.scale = scale
    if parent:
        o.parent = parent
    if mat is not None and not me.materials:
        me.materials.append(mat)
    return o


def aim_euler(pos, tgt, prev=None, roll=0.0):
    d = Vector(tgt) - Vector(pos)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")


def aim(o, tgt):
    o.rotation_euler = aim_euler(o.location, tgt)


# ============================================================================ the tower
W = 10.0           # shaft width
CS0, CS1 = 40.0, 52.0      # clock stage
CW = 11.0          # clock stage width
DIAL_Z = 46.0
DIAL_R = 3.45
GLASS_Y = -CW / 2 + 0.15   # dial glass plane (front)
BEL0, BEL1 = 52.4, 60.2    # belfry
SPIRE0, SPIRE_TOP = 60.9, 82.0


def face_parents(c):
    ps = []
    for k in range(4):
        e = empty(f"Face{k}", (0, 0, 0), c)
        e.rotation_euler = (0, 0, k * math.pi / 2)
        ps.append(e)
    return ps


R90X = (math.pi / 2, 0, 0)


def build_tower(M):
    C = coll("Tower")
    S = M["stone"]
    ob("Plinth", mesh_box("Plinth", 12.4, 12.4, 3.0, bevel=0.06, segs=1, origin="bottom"), C, (0, 0, 0), mat=M["stone2"])
    ob("Plinth2", mesh_box("Plinth2", 11.2, 11.2, 0.7, bevel=0.08, segs=1, origin="bottom"), C, (0, 0, 3.0), mat=M["stone2"])
    ob("Shaft", mesh_box("Shaft", W, W, CS0 - 3.7, origin="bottom"), C, (0, 0, 3.7), mat=S)
    for z in (14.0, 27.0):
        ob(f"String{int(z)}", mesh_box(f"String{int(z)}", W + 0.7, W + 0.7, 0.42, bevel=0.08, segs=2, origin="bottom"), C, (0, 0, z), mat=M["stone2"])
    # corner buttresses, stepped
    for sx in (-1, 1):
        for sy in (-1, 1):
            ob("Butt", mesh_box("ButtA", 1.7, 1.7, 20.0, bevel=0.03, segs=1, origin="bottom"), C, (sx * 5.05, sy * 5.05, 3.7), mat=S)
            ob("ButtCap", pyramid_mesh("ButtCapA", 1.7, 0.9, top_w=1.3), C, (sx * 5.05, sy * 5.05, 23.7), mat=M["stone2"])
            ob("Butt2", mesh_box("ButtB", 1.3, 1.3, 15.5, bevel=0.03, segs=1, origin="bottom"), C, (sx * 4.95, sy * 4.95, 24.6), mat=S)
            ob("ButtCap2", pyramid_mesh("ButtCapB", 1.3, 1.2), C, (sx * 4.95, sy * 4.95, 40.1), mat=M["stone2"])
    faces = face_parents(C)
    # lancet windows on the shaft, two tiers
    for k, fp in enumerate(faces):
        for tier, (z0, hs, rise, a) in enumerate(((16.0, 6.0, 1.3, 0.55), (29.0, 6.4, 1.4, 0.6))):
            for j, x in enumerate((-1.55, 1.55)):
                lit = (k + tier + j) % 3 == 0 or (k == 0 and tier == 1)
                pts = lancet(a, hs, rise)
                panel = extrude_poly("Lancet", pts, 0.05)
                ob("Win", panel, C, (x, -W / 2 - 0.01, z0), R90X, mat=M["win_lit"] if lit else M["win_dark"], parent=fp)
                outer = lancet(a + 0.24, hs, rise + 0.3)
                frame = strip_mesh("LancetFrame", outer, pts, 0.22)
                ob("WinFrame", frame, C, (x, -W / 2 + 0.02, z0), R90X, mat=M["stone2"], parent=fp)
                ob("Sill", mesh_box("Sill", 2 * a + 0.7, 0.45, 0.22, bevel=0.04, segs=1, origin="bottom"), C, (x, -W / 2 - 0.12, z0 - 0.22), mat=M["stone2"], parent=fp)
    # clock stage: corbel, walls with dial holes, corner turrets
    ob("Corbel", mesh_box("Corbel", CW + 0.9, CW + 0.9, 0.7, bevel=0.1, segs=2, origin="bottom"), C, (0, 0, CS0 - 0.2), mat=M["stone2"])
    dent = mesh_box("Dentil", 0.16, 0.3, 0.26, origin="bottom")
    dent.materials.append(M["stone2"])
    for i in range(36):
        for k, fp in enumerate(faces):
            x = -CW / 2 + 0.3 + i * (CW - 0.6) / 35
            ob("Dentil", dent, C, (x, -CW / 2 - 0.4, CS0 - 0.46), parent=fp)
    for k, fp in enumerate(faces):
        slab = slab_with_hole(f"ClockWall{k}", CW, CS1 - CS0, 0.0, DIAL_Z - CS0, DIAL_R + 0.12, 0.6)
        o = ob(f"ClockWall{k}", slab, C, (0, -CW / 2 + 0.6, CS0), R90X, mat=S, parent=fp)
        # moulded stone ring around the dial and a square frame
        ob("DialMould", ring_mesh("DialMould", DIAL_R + 0.75, DIAL_R + 0.12, 0.28, 96), C, (0, -CW / 2 + 0.02, DIAL_Z), R90X, mat=M["stone2"], parent=fp)
        sq = slab_with_hole("DialFrame", 8.6, 8.6, 0, 4.3, DIAL_R + 0.75, 0.16)
        ob("DialFrame", sq, C, (0, -CW / 2 + 0.02, DIAL_Z - 4.3), R90X, mat=M["stone2"], parent=fp)
        for sx in (-1, 1):
            for sy in (-1, 1):   # carved quatrefoils in the spandrels with a gilt boss
                qf = []
                for i in range(64):
                    a = 2 * math.pi * i / 64
                    rr = 0.42 + 0.12 * abs(math.cos(2 * a))
                    qf.append((rr * math.sin(a), rr * math.cos(a)))
                qin = [(0.72 * x, 0.72 * y) for x, y in qf]
                ob("Quatrefoil", strip_mesh("Quatrefoil", qf, qin, 0.12, closed=True), C, (sx * 3.65, -CW / 2 - 0.12, DIAL_Z + sy * 3.65), R90X, mat=M["stone2"], parent=fp)
                ob("Boss", mesh_sphere("SpandrelBoss", 0.13, 16, 8, scale=(1, 1, 0.5)), C, (sx * 3.65, -CW / 2 - 0.16, DIAL_Z + sy * 3.65), R90X, mat=M["gold"], parent=fp)
        t = text_obj("Motto", "TEMPUS  FUGIT", C, size=0.42, extrude=0.03, fontfile="timesbd.ttf", mat=M["gold"],
                     loc=(0, -CW / 2 - 0.05, CS0 + 0.95), rot=R90X, spacing=1.25)
        t.parent = fp
        # hood mould + gable above the dial
        ob("Cornice", mesh_box("CorniceA", CW + 0.2, 0.6, 0.45, bevel=0.06, segs=1, origin="bottom"), C, (0, -CW / 2 - 0.1, CS1 - 0.2), mat=M["stone2"], parent=fp)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = (sx * CW / 2, sy * CW / 2)
            ob("Turret", mesh_cyl("Turret", 0.78, CS1 - CS0 + 1.6, verts=8, origin="bottom"), C, (p[0], p[1], CS0 - 0.1), mat=M["stone2"])
            ob("TurretBand", mesh_cyl("TurretBand", 0.92, 0.3, verts=8, origin="bottom"), C, (p[0], p[1], CS1 + 1.4), mat=M["stone2"])
            ob("Pinnacle", pyramid_mesh("Pinnacle", 1.5, 4.6, sides=8), C, (p[0], p[1], CS1 + 1.7), mat=M["slate"])
            ob("PinBall", mesh_sphere("PinBall", 0.16, 16, 8), C, (p[0], p[1], CS1 + 6.35), mat=M["gold"])
    # belfry: parapet, piers, mullions, arch spandrels, cornice
    ob("BelfryFloor", mesh_box("BelfryFloor", CW, CW, 0.5, origin="bottom"), C, (0, 0, CS1 - 0.1), mat=M["floor"])
    ob("BelfryBase", mesh_box("BelfryBase", W + 1.0, W + 1.0, 0.5, bevel=0.08, segs=1, origin="bottom"), C, (0, 0, CS1 + 0.25), mat=M["stone2"])
    bw = W
    for k, fp in enumerate(faces):
        ob("Parapet", mesh_box("Parapet", bw, 0.7, 0.9, origin="bottom"), C, (0, -bw / 2 + 0.35, BEL0 + 0.3), mat=S, parent=fp)
        ob("Mullion", mesh_box("Mullion", 0.5, 0.7, BEL1 - BEL0 - 1.2, origin="bottom"), C, (0, -bw / 2 + 0.35, BEL0 + 1.2), mat=M["stone2"], parent=fp)
        a, spring = 1.65, BEL0 + 4.8
        for x in (-1.9, 1.9):
            arch = lancet(a, 0.0, 1.85)
            top = BEL1 - 0.6
            # spandrel above the arch: strip from the arch curve to the top line
            outer = [(px, top - spring) for px, _ in arch]
            sp = strip_mesh("Spandrel", arch, outer, 0.7)
            ob("Spandrel", sp, C, (x, -bw / 2 + 0.7, spring), R90X, mat=S, parent=fp)
            ob("ArchRoll", strip_mesh("ArchRoll", lancet(a + 0.16, 0.0, 2.05), arch, 0.16), C, (x, -bw / 2 - 0.02, spring), R90X, mat=M["stone2"], parent=fp)
            ob("Louvre", mesh_box("Jamb", 0.25, 0.7, 4.0, origin="bottom"), C, (x + (a + 0.12) * (1 if x > 0 else -1), -bw / 2 + 0.35, BEL0 + 1.2), mat=S, parent=fp)
        ob("BelCornice", mesh_box("BelCornice", W + 0.9, 0.9, 0.55, bevel=0.08, segs=1, origin="bottom"), C, (0, -W / 2 - 0.05, BEL1 - 0.6), mat=M["stone2"], parent=fp)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ob("BelPier", mesh_box("BelPier", 1.7, 1.7, BEL1 - BEL0 + 0.3, bevel=0.03, segs=1, origin="bottom"), C, (sx * 4.3, sy * 4.3, BEL0), mat=S)
            ob("BelPin", pyramid_mesh("BelPin", 1.4, 5.5, sides=4), C, (sx * 4.55, sy * 4.55, BEL1 + 0.5), mat=M["slate"])
            ob("BelPinBall", mesh_sphere("BelPinBall", 0.2, 16, 8), C, (sx * 4.55, sy * 4.55, BEL1 + 6.0), mat=M["gold"])
    ob("BelfryRoofBase", mesh_box("BelRoofBase", W + 0.4, W + 0.4, 0.7, bevel=0.06, segs=1, origin="bottom"), C, (0, 0, BEL1 - 0.05), mat=M["stone2"])
    # spire with dormers
    ob("Spire", pyramid_mesh("Spire", W - 0.6, SPIRE_TOP - SPIRE0), C, (0, 0, SPIRE0 - 0.2), mat=M["slate"])
    for k, fp in enumerate(faces):
        z = SPIRE0 + 2.6
        d = ob("Dormer", mesh_box("Dormer", 1.4, 2.2, 1.6, origin="bottom"), C, (0, -3.15, z), mat=M["stone2"], parent=fp)
        ob("DormerRoof", pyramid_mesh("DormerRoof", 1.7, 0.9), C, (0, -3.15, z + 1.6), (0, 0, 0), (1.0, 1.3, 1.0), mat=M["slate"], parent=fp)
        ob("DormerWin", extrude_poly("DormerWin", lancet(0.38, 0.7, 0.45), 0.04), C, (0, -4.27, z + 0.15), R90X, mat=M["win_lit"], parent=fp)
    ob("FinialBall", mesh_sphere("FinialBall", 0.5, 24, 12), C, (0, 0, SPIRE_TOP + 0.2), mat=M["gold"])
    ob("FinialRod", mesh_cyl("FinialRod", 0.06, 3.2, verts=12), C, (0, 0, SPIRE_TOP + 0.5), mat=M["gold"])
    ob("FinialCross", mesh_box("FinialCross", 1.3, 0.08, 0.08), C, (0, 0, SPIRE_TOP + 2.6), mat=M["gold"])
    vane = extrude_poly("Vane", [(-0.9, -0.05), (0.55, -0.05), (0.55, -0.22), (1.0, 0.0), (0.55, 0.22), (0.55, 0.05), (-0.9, 0.05), (-1.15, 0.25), (-1.15, -0.25)], 0.03)
    ob("Vane", vane, C, (0, 0, SPIRE_TOP + 3.3), (math.pi / 2, 0, 0.6), mat=M["gold"])
    return C, faces


# ============================================================================ the clock (front built for real, other faces instance it)
def hand_shapes():
    minute = mirror_profile([(0.0, -0.8), (0.1, -0.76), (0.16, -0.64), (0.16, -0.48), (0.1, -0.38), (0.055, -0.3), (0.075, 0.0),
                             (0.062, 1.8), (0.17, 2.12), (0.05, 2.42), (0.042, 2.72), (0.1, 2.78), (0.0, 3.08)])
    hour = mirror_profile([(0.0, -0.58), (0.12, -0.52), (0.15, -0.38), (0.09, -0.26), (0.11, 0.0), (0.09, 1.22),
                           (0.22, 1.42), (0.28, 1.6), (0.24, 1.77), (0.14, 1.91), (0.0, 2.18)])
    sec = [(0.0, -1.02)]
    for i in range(1, 16):
        a = math.pi - math.pi * i / 16 * 0.85
        sec.append((0.19 * math.sin(a), -0.78 + 0.19 * math.cos(a)))
    sec += [(0.03, -0.55), (0.022, 2.95), (0.05, 3.0), (0.0, 3.28)]
    second = mirror_profile(sec)
    return hour, minute, second


def build_clock(M):
    C = coll("Clock")
    root = empty("ClockFront", (0, GLASS_Y, DIAL_Z), C)
    root.rotation_euler = R90X
    glass = ob("DialGlass", mesh_cyl("DialGlass", DIAL_R + 0.05, 0.04, verts=96, origin="center"), C, (0, 0, -0.03), mat=M["dial"], parent=root)
    iron, gold = M["iron"], M["gold"]
    # tracery rings + radial bars
    for r_, w_ in ((DIAL_R, 0.1), (2.62, 0.07), (1.78, 0.06), (0.58, 0.06), (3.1, 0.03)):
        ob("Trac", ring_mesh("Trac", r_, r_ - w_, 0.07, 96), C, (0, 0, -0.01), mat=iron, parent=root)
    bar = mesh_box("TracBar", 1.0, 1.0, 0.07, origin="center")
    bar.materials.append(iron)
    # many small panes like a real cast-iron dial: fine bars in the outer band, heavier spokes inside
    for i in range(48):
        a = i * 7.5 * D2R
        r0, r1 = 1.78, 2.62
        ob("TracBar", bar, C, ((r0 + r1) / 2 * math.sin(a), (r0 + r1) / 2 * math.cos(a), 0.025), (0, 0, -a), (0.03, r1 - r0, 1), parent=root)
    for i in range(24):
        a = i * 15 * D2R
        r0, r1 = 0.58, 1.78
        ob("TracBar", bar, C, ((r0 + r1) / 2 * math.sin(a), (r0 + r1) / 2 * math.cos(a), 0.025), (0, 0, -a), (0.045 if i % 2 == 0 else 0.03, r1 - r0, 1), parent=root)
    ob("Trac", ring_mesh("Trac", 1.2, 1.17, 0.07, 96), C, (0, 0, -0.01), mat=iron, parent=root)
    ob("Trac", ring_mesh("Trac", 2.2, 2.175, 0.07, 96), C, (0, 0, -0.01), mat=iron, parent=root)
    # minute marks
    mk = mesh_box("Mark", 1.0, 1.0, 0.05, origin="center")
    mk.materials.append(iron)
    for i in range(60):
        a = i * 6 * D2R
        big = i % 5 == 0
        L_ = 0.3 if big else 0.18
        r_ = 3.3 - L_ / 2
        ob("Mark", mk, C, (r_ * math.sin(a), r_ * math.cos(a), 0.03), (0, 0, -a), (0.085 if big else 0.035, L_, 1), parent=root)
    # roman numerals (radial, traditional IIII)
    nums = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    for h, s in enumerate(nums):
        a = h * 30 * D2R
        r_ = 2.86
        t = text_obj(f"Num{h}", s, C, size=0.5, extrude=0.025, fontfile="timesbd.ttf", mat=iron,
                     loc=(r_ * math.sin(a), r_ * math.cos(a), 0.03), rot=(0, 0, -a), spacing=0.95)
        t.parent = root
    # bezel + boss
    bez = bpy.data.meshes.new("Bezel")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=False, segments=96, radius=DIAL_R + 0.1)
    bm.to_mesh(bez); bm.free()
    bz = ob("Bezel", mesh_cyl("BezelC", DIAL_R + 0.22, 0.16, verts=96, origin="center", caps=False), C, (0, 0, 0.05), mat=gold, parent=root)
    bpy.data.meshes.remove(bez)
    ob("BezelIn", ring_mesh("BezelIn", DIAL_R + 0.22, DIAL_R + 0.02, 0.18, 96), C, (0, 0, -0.04), mat=gold, parent=root)
    # hands
    hour, minute, second = hand_shapes()
    hands = {}
    for nm, pts, z, mat, th in (("Hour", hour, 0.12, iron, 0.06), ("Minute", minute, 0.22, iron, 0.06), ("Second", second, 0.32, M["red"], 0.04)):
        piv = empty(f"{nm}Pivot", (0, 0, z), C, parent=root)
        ob(f"{nm}Hand", extrude_poly(nm, pts, th), C, (0, 0, 0), parent=piv, mat=mat)
        hands[nm] = piv
    ob("Boss", mesh_sphere("Boss", 0.2, 24, 12, scale=(1, 1, 0.6)), C, (0, 0, 0.42), mat=gold, parent=root)
    ob("BossRing", mesh_cyl("BossRing", 0.24, 0.1, verts=32, origin="bottom"), C, (0, 0, 0.36), mat=gold, parent=root)
    glass.visible_shadow = True
    return C, root, hands, glass


# ============================================================================ the mechanism room
def build_interior(M):
    C = coll("Interior")
    ob("RoomFloor", mesh_box("RoomFloor", CW - 1.2, CW - 1.2, 0.3, origin="bottom"), C, (0, 0, CS0 + 0.2), mat=M["floor"])
    ob("RoomCeil", mesh_box("RoomCeil", CW - 1.2, CW - 1.2, 0.3, origin="bottom"), C, (0, 0, CS1 - 0.4), mat=M["wood"])
    beam = mesh_box("Beam", CW - 1.2, 0.34, 0.42, bevel=0.02, segs=1)
    beam.materials.append(M["wood"])
    for i, y in enumerate((-3.6, -1.2, 1.2, 3.6)):
        ob("Beam", beam, C, (0, y, CS1 - 0.62))
    post = mesh_box("Post", 0.34, 0.34, CS1 - CS0 - 0.9, bevel=0.02, segs=1, origin="bottom")
    post.materials.append(M["wood"])
    for p in ((-4.1, 3.9), (4.1, 3.9), (-4.1, -2.6), (4.1, -2.6)):
        ob("Post", post, C, (p[0], p[1], CS0 + 0.5))
    # ladder up to the belfry
    tilt = 0.12
    for s in (-0.35, 0.35):
        ob("LadderRail", mesh_box("LRail", 0.08, 0.1, 11.2, origin="bottom"), C, (-3.6 + s, 4.6, CS0 + 0.5), (tilt, 0, 0), mat=M["wood"])
    for i in range(25):
        h = 0.3 + 0.42 * i
        ob("Rung", mesh_cyl("Rung", 0.03, 0.7, verts=8, origin="center"), C, (-3.6, 4.6 - h * math.sin(tilt), CS0 + 0.5 + h * math.cos(tilt)), (0, math.pi / 2, 0), mat=M["wood"])
    # iron legs under the movement frame
    for x in (-3.5, 3.1):
        ob("FrameLeg", mesh_box("FrameLeg", 0.3, 0.6, 2.5, bevel=0.02, segs=1, origin="bottom"), C, (x, 1.4, CS0 + 0.5), mat=M["iron"])
    # hanging work lamps
    for p in ((-3.0, 2.8, CS1 - 2.2), (3.3, 1.5, CS1 - 1.9)):
        ob("Cord", mesh_cyl("Cord", 0.01, 1.6, verts=6), C, (p[0], p[1], p[2] + 0.1), mat=M["iron"])
        ob("Shade", mesh_cyl("Shade", 0.22, 0.18, verts=24, r2=0.06, origin="bottom", caps=False), C, (p[0], p[1], p[2] - 0.05), mat=M["iron"])
        b_ = ob("Bulb", mesh_sphere("Bulb", 0.07, 16, 8), C, (p[0], p[1], p[2] - 0.06), mat=M["bulb"])
        b_.visible_shadow = False
        lt = bpy.data.lights.new("Lamp", "POINT"); lt.energy = 120; lt.color = hexcol("#FFB46A")[:3]; lt.shadow_soft_size = 0.06
        lt.volume_factor = 0.0; lt.use_shadow = False
        lo = bpy.data.objects.new("Lamp", lt); C.objects.link(lo); lo.location = (p[0], p[1], p[2] - 0.18)
    # room haze
    hz = ob("RoomHaze", mesh_box("RoomHaze", CW - 1.25, CW - 1.25, CS1 - CS0 - 0.95, origin="bottom"), C, (0, 0, CS0 + 0.5), mat=M["vol_room"])
    hz.visible_shadow = False
    return C


def build_movement(M):
    C = coll("Movement")
    base = Vector((0.4, 1.7, 45.0))
    mv = empty("Movement", base, C)
    mv.rotation_euler = R90X          # local XY = world XZ plane; local +Z = world -Y (toward the dial)
    brass, steel, iron = M["brass"], M["steel"], M["iron"]
    mod = 0.06
    gears = {}

    def add_gear(name, n, pos, thick=0.07, spokes=6, escape=False, mat=brass):
        me, r = gear_mesh(name, n, mod if not escape else 0.05, thick, spokes=spokes, escape=escape)
        me.materials.append(mat)
        piv = empty(name + "Piv", pos, C, parent=mv)
        o = ob(name, me, C, (0, 0, -thick / 2), parent=piv)
        gears[name] = dict(piv=piv, n=n, r=r, pos=Vector(pos))
        # arbor through the gear
        ob(name + "Arbor", mesh_cyl(name + "Arb", 0.035, 1.6, verts=12, origin="center"), C, (0, 0, 0.35), mat=steel, parent=piv)
        return gears[name]

    big = add_gear("GBig", 56, (-2.0, -0.2, 0.0), spokes=6)
    phi1 = 22 * D2R
    small_pos = big["pos"] + Vector((math.cos(phi1), math.sin(phi1), 0)) * (mod * (56 + 16) / 2)
    small = add_gear("GSmall", 16, tuple(small_pos), spokes=0, mat=steel)
    mid = add_gear("GMid", 44, (small_pos.x, small_pos.y, -0.32), spokes=5)
    phi2 = -38 * D2R
    c_pos = mid["pos"] + Vector((math.cos(phi2), math.sin(phi2), 0)) * (mod * (44 + 14) / 2)
    cg = add_gear("GC", 14, tuple(c_pos), spokes=0, mat=steel)
    esc = add_gear("GEsc", 30, (c_pos.x, c_pos.y, -1.05), thick=0.05, spokes=5, escape=True)
    # a fly-fan gear high up and a second train for depth
    st2 = add_gear("GStrike", 64, (-1.0, 2.0, 0.55), spokes=8, thick=0.08)
    st3_pos = st2["pos"] + Vector((math.cos(-30 * D2R), math.sin(-30 * D2R), 0)) * (mod * (64 + 18) / 2)
    st3 = add_gear("GStrikeP", 18, (st3_pos.x, st3_pos.y, 0.55), spokes=0, mat=steel)
    # cast iron frame: two posts and rails behind the gears (dial side)
    for x in (-3.9, 2.7):
        ob("FramePost", mesh_box("FramePost", 0.22, 4.4, 0.18, bevel=0.02, segs=1), C, (x, 0.1, 0.4), mat=iron, parent=mv)
    for y in (-1.95, 2.05):
        ob("FrameRail", mesh_box("FrameRail", 6.8, 0.2, 0.18, bevel=0.02, segs=1), C, (-0.6, y, 0.4), mat=iron, parent=mv)
    ob("FrameFoot", mesh_box("FrameFoot", 7.4, 0.5, 1.2, bevel=0.03, segs=1), C, (-0.6, -2.3, 0.3), mat=iron, parent=mv)
    # anchor (pallets) above the escape wheel
    ar = 0.05 * 30 / 2
    apiv = empty("AnchorPiv", (c_pos.x, c_pos.y + ar + 0.42, -1.09), C, parent=mv)
    arm = []
    for i in range(13):
        a = (-50 + 100 * i / 12) * D2R
        arm.append((0.62 * math.sin(a), -0.62 * math.cos(a)))
    arm_in = [(0.5 * x / 0.62, 0.5 * y / 0.62) for x, y in arm]
    ob("Anchor", strip_mesh("Anchor", arm, arm_in, 0.05), C, (0, 0, 0), mat=steel, parent=apiv)
    for s in (-1, 1):
        a = 50 * s * D2R
        ob("Pallet", mesh_box("Pallet", 0.06, 0.16, 0.06), C, (0.56 * math.sin(a), -0.56 * math.cos(a) - 0.08, 0.02), (0, 0, a), mat=M["red"], parent=apiv)
    ob("AnchorHub", mesh_cyl("AnchorHub", 0.07, 0.12, verts=16, origin="center"), C, (0, 0, 0.02), mat=brass, parent=apiv)
    ob("Crutch", mesh_box("Crutch", 0.04, 0.9, 0.04), C, (0, 0.45, -0.05), mat=steel, parent=apiv)
    # pendulum (swings parallel to the dial)
    ppiv = empty("PendPiv", (c_pos.x, c_pos.y + ar + 0.9, -1.2), C, parent=mv)
    ob("PendBracket", mesh_box("PendBracket", 0.5, 0.14, 1.3, bevel=0.01, segs=1), C, (c_pos.x, c_pos.y + ar + 0.98, -0.6), mat=iron, parent=mv)
    PL = 3.6
    ob("PendRod", mesh_cyl("PendRod", 0.025, PL, verts=12, origin="bottom"), C, (0, 0, 0), (math.pi / 2, 0, 0), mat=steel, parent=ppiv)
    bob = ob("PendBob", mesh_cyl("PendBob", 0.42, 0.12, verts=48, bevel=0.03, origin="center"), C, (0, -PL, 0), mat=brass, parent=ppiv)
    ob("PendBobRim", ring_mesh("BobRim", 0.44, 0.36, 0.14, 48), C, (0, -PL, -0.07), mat=M["gold"], parent=ppiv)
    # leading-off rod from the train to the dial centre
    dial_back = Vector((0, GLASS_Y + 0.45, DIAL_Z))
    gb_world = base + Vector((big["pos"].x, 0, big["pos"].y))
    src = gb_world + Vector((0, -0.35, 0))
    rod_vec = dial_back - src
    rod = ob("LeadRod", mesh_cyl("LeadRod", 0.045, rod_vec.length, verts=12, origin="bottom"), C, tuple(src), mat=steel)
    rod.rotation_euler = rod_vec.to_track_quat("Z", "Y").to_euler()
    ob("MotionBox", mesh_box("MotionBox", 0.7, 0.4, 0.7, bevel=0.03, segs=1), C, (0, GLASS_Y + 0.6, DIAL_Z), mat=iron)
    ob("MotionGear", gear_mesh("MotionGear", 32, 0.04, 0.05, spokes=4)[0], C, (0, GLASS_Y + 0.85, DIAL_Z), R90X, mat=brass)
    return C, mv, gears, apiv, ppiv, gb_world


def build_bell(M):
    C = coll("Bell")
    prof_out = [(0.05, 2.38), (0.42, 2.36), (0.6, 2.28), (0.69, 2.08), (0.72, 1.72), (0.78, 1.3), (0.92, 0.86),
                (1.1, 0.48), (1.3, 0.18), (1.42, 0.04), (1.41, -0.03)]
    prof_in = [(1.3, -0.02), (1.2, 0.18), (1.0, 0.52), (0.84, 0.95), (0.7, 1.4), (0.63, 1.9), (0.5, 2.18), (0.05, 2.24)]
    bell_me = lathe_mesh("Bell", prof_out + prof_in, 72)
    BZ = 54.5
    bell = ob("Bell", bell_me, C, (0, 0, BZ), mat=M["bronze"])
    ob("BellCrown", ring_mesh("Crown", 0.32, 0.18, 0.35, 32), C, (0, 0, BZ + 2.36), mat=M["bronze"])
    ob("Headstock", mesh_box("Headstock", 3.8, 0.6, 0.55, bevel=0.03, segs=1), C, (0, 0, BZ + 2.95), mat=M["wood"])
    for s in (-1, 1):
        ob("FrameA", mesh_box("FrameA", 0.4, 0.45, 6.7, bevel=0.02, segs=1, origin="bottom"), C, (s * 2.0, 0, BEL0), mat=M["wood"])
        ob("FrameBrace", mesh_box("FrameBrace", 0.3, 0.3, 3.0, bevel=0.02, segs=1), C, (s * 2.0, 0.9, BEL0 + 1.6), (0.65, 0, 0), mat=M["wood"])
        ob("FrameBrace", mesh_box("FrameBrace2", 0.3, 0.3, 3.0, bevel=0.02, segs=1), C, (s * 2.0, -0.9, BEL0 + 1.6), (-0.65, 0, 0), mat=M["wood"])
    ob("FrameTop", mesh_box("FrameTop", 4.6, 0.5, 0.5, bevel=0.02, segs=1), C, (0, 0, BEL0 + 6.9), mat=M["wood"])
    # clock hammer: pivots on the right, head rests just off the lip
    hp = empty("HammerPiv", (2.45, -0.25, BZ + 0.34), C)
    ob("HammerArm", mesh_box("HammerArm", 0.95, 0.09, 0.09), C, (-0.47, 0, -0.05), mat=M["iron"], parent=hp)
    ob("HammerHead", mesh_cyl("HammerHead", 0.17, 0.42, verts=24, origin="center"), C, (-0.93, 0, -0.14), (math.pi / 2, 0, 0), mat=M["iron"], parent=hp)
    ob("HammerHub", mesh_cyl("HammerHub", 0.1, 0.3, verts=16, origin="center"), C, (0, 0, 0), (math.pi / 2, 0, 0), mat=M["iron"], parent=hp)
    ob("HammerBlock", mesh_box("HammerBlock", 0.4, 0.5, BZ + 0.3 - BEL0, bevel=0.02, segs=1, origin="bottom"), C, (2.6, -0.25, BEL0), mat=M["wood"])
    hz = ob("BelfryHaze", mesh_box("BelfryHaze", W - 1.2, W - 1.2, BEL1 - BEL0 - 0.6, origin="bottom"), C, (0, 0, BEL0 + 0.35), mat=M["vol_bell"])
    hz.visible_shadow = False
    return C, bell, hp


# ============================================================================ city, ground, moon
def build_city(M):
    C = coll("City")
    ob("Ground", mesh_plane("Ground", 900, 900), C, (0, 0, -0.02), mat=M["ground"])
    ob("Plaza", mesh_box("Plaza", 34, 34, 0.15, bevel=0.03, segs=1, origin="bottom"), C, (0, 0, -0.02), mat=M["stone2"])
    cube = mesh_box("CityCube", 1, 1, 1, origin="bottom")
    cube.materials.append(M["city"])
    roof = bpy.data.meshes.new("CityGable")
    bm = bmesh.new()
    v = [bm.verts.new(p) for p in ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0), (-0.5, 0, 1), (0.5, 0, 1))]
    bm.faces.new((v[0], v[1], v[5], v[4])); bm.faces.new((v[2], v[3], v[4], v[5]))
    bm.faces.new((v[1], v[2], v[5])); bm.faces.new((v[3], v[0], v[4])); bm.faces.new((v[3], v[2], v[1], v[0]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(roof); bm.free()
    roof.materials.append(M["roof"])
    n = 0
    for gx in range(-24, 25):
        for gy in range(-24, 25):
            cx, cy = gx * 13.0 + rng.uniform(-2, 2), gy * 13.0 + rng.uniform(-2, 2)
            d = math.hypot(cx, cy)
            if d < 24 or d > 320:
                continue
            if rng.random() < 0.12:
                continue   # squares and streets
            # keep a boulevard open from the opening camera to the tower
            a_, b_ = Vector((-24, -96)), Vector((0, 0))
            pp = Vector((cx, cy))
            u_ = max(0.0, min(1.0, (pp - a_).dot(b_ - a_) / (b_ - a_).length_squared))
            if (a_ + (b_ - a_) * u_ - pp).length < 12.0 and u_ < 0.85:
                continue
            w, dp = rng.uniform(7, 11.5), rng.uniform(7, 11.5)
            h = rng.uniform(7, 15) * (1.0 + 0.35 * (d < 60))
            ob("Bldg", cube, C, (cx, cy, 0), (0, 0, rng.choice((0, 0, math.pi / 2)) + rng.uniform(-0.05, 0.05)), (w, dp, h))
            if rng.random() < 0.75:
                ob("Gable", roof, C, (cx, cy, h), (0, 0, rng.choice((0, math.pi / 2))), (w + 0.5, dp + 0.5, rng.uniform(2.5, 4.5)))
            n += 1
    # a couple of distant church spires on the skyline
    for p, hgt in (((-120, 160), 46), ((140, 210), 38), ((60, -200), 34)):
        ob("FarTower", cube, C, (p[0], p[1], 0), (0, 0, 0.3), (7, 7, hgt))
        ob("FarSpire", pyramid_mesh("FarSpire", 7.2, 18), C, (p[0], p[1], hgt), (0, 0, 0.3), mat=M["slate"])
    print("[scene] city buildings:", n)
    glow = ob("CityGlow", mesh_box("CityGlow", 640, 640, 16, origin="bottom"), C, (0, 0, 0), mat=M["cityglow"])
    glow.visible_shadow = False
    moon = ob("Moon", mesh_sphere("Moon", 15.0, 48, 24), C, tuple(MOON_POS), mat=M["moon"])
    moon.visible_shadow = False
    return C


def build_lights(M, clock_root):
    C = coll("Lights")

    def light(name, kind, loc, energy, col, tgt=None, size=None, radius=0.1, blend=0.2, shadow=True, vol=1.0):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = hexcol(col)[:3]
        ld.use_shadow = shadow
        ld.volume_factor = vol
        if kind == "SPOT":
            ld.spot_size = size or 0.8
            ld.spot_blend = blend
        if kind in ("SPOT", "POINT"):
            ld.shadow_soft_size = radius
        if kind == "AREA":
            ld.shape = "DISK"; ld.size = size or 1
        o = bpy.data.objects.new(name, ld)
        C.objects.link(o)
        o.location = loc
        if tgt is not None:
            aim(o, tgt)
        return o

    moon_dir = (Vector((0, 0, 0)) - MOON_POS).normalized()
    sun = light("MoonLight", "SUN", (0, 0, 100), float(arg("--moon", 0.55)), "#A8BBFF")
    sun.rotation_euler = moon_dir.to_track_quat("-Z", "Y").to_euler()
    sun.data.angle = 0.02
    floods = []
    for k in range(4):
        rot = Matrix.Rotation(k * math.pi / 2, 3, "Z")
        for x in (-3.4, 3.4):
            p = rot @ Vector((x, -14.0, 0.6))
            t = rot @ Vector((x * 0.55, -5.0, 36.0))
            floods.append(light("Flood", "SPOT", tuple(p), float(arg("--flood", 90000)), "#FFBE80", tuple(t), size=0.42, blend=0.6, radius=0.25, vol=0.25))
    bel = light("BelfryLamp", "POINT", (-2.9, -2.6, 58.6), 1400, "#FFB36E", radius=0.3, vol=0.15)
    bel2 = light("BelfryLamp2", "POINT", (2.9, 2.9, 58.2), 700, "#FFB36E", radius=0.3, shadow=False, vol=0.1)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):   # spire washers on the belfry cornice
        light("SpireWash", "SPOT", (sx * 4.6, sy * 4.6, SPIRE0 + 0.3), 9000, "#FFD2A0", (0, 0, SPIRE0 + 11), size=0.9, blend=0.6, radius=0.2, shadow=False, vol=0.2)
    fill = light("DialFill", "AREA", (0, GLASS_Y + 0.6, DIAL_Z), float(arg("--fill", 300)), "#FFD9A0", (0, 5, DIAL_Z - 1.0), size=6.0, shadow=False, vol=0.0)
    beam = light("BeamSpot", "SPOT", (-3.2, GLASS_Y - 9.0, DIAL_Z + 4.6), float(arg("--beam", 40000)), "#FFE6BE", (0.4, 2.0, DIAL_Z - 2.5), size=0.95, blend=0.12, radius=0.1)
    rim = light("GearRim", "SPOT", (3.8, 4.3, CS1 - 0.9), 1500, "#FFC48A", (0.0, 1.0, 45.0), size=1.2, blend=0.5, radius=0.2, vol=0.0)
    flash = light("StrikeFlash", "POINT", (1.1, -0.9, 55.2), 0.0, "#FFD7A0", radius=0.5, shadow=False, vol=0.25)
    return C, dict(sun=sun, floods=floods, bel=bel, fill=fill, beam=beam, rim=rim, flash=flash)


# ============================================================================ dust, birds
def seg_dist(p, a, b):
    a, b = Vector(a), Vector(b)
    u = max(0.0, min(1.0, (p - a).dot(b - a) / max(1e-9, (b - a).length_squared)))
    return (a + (b - a) * u - p).length


def dust_cloud(name, C, M, n, box, rmin=0.004, rmax=0.011, seed=1, avoid=()):
    r_ = random.Random(seed)
    bm = bmesh.new()
    made = 0
    while made < n:
        c = Vector((r_.uniform(box[0][0], box[1][0]), r_.uniform(box[0][1], box[1][1]), r_.uniform(box[0][2], box[1][2])))
        if any(seg_dist(c, a, b) < rad_ for a, b, rad_ in avoid):
            continue
        made += 1
        rad = r_.uniform(rmin, rmax)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rad)
        bmesh.ops.translate(bm, verts=res["verts"], vec=c)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(M["dust"])
    o = bpy.data.objects.new(name, me)
    C.objects.link(o)
    o.visible_shadow = False
    return o


def build_birds(M, n=44):
    C = coll("Birds")
    body = mesh_sphere("BirdBody", 0.2, 12, 8, scale=(1.0, 2.2, 0.9))
    body.materials.append(M["bird"])
    wing = bpy.data.meshes.new("Wing")
    bm = bmesh.new()
    v = [bm.verts.new(p) for p in ((0, 0.18, 0), (0.75, 0.05, 0), (0.95, -0.12, 0), (0, -0.2, 0))]
    bm.faces.new(v)
    bm.to_mesh(wing); bm.free()
    wing.materials.append(M["bird"])
    birds = []
    r_ = random.Random(11)
    for i in range(n):
        side = r_.choice((0, 0, 1, 3))
        along = r_.uniform(-3.4, 3.4)
        rot = Matrix.Rotation(side * math.pi / 2, 3, "Z")
        start = rot @ Vector((along, -r_.uniform(4.45, 4.85), BEL0 + 1.36))
        ang = side * math.pi / 2 + r_.uniform(-0.5, 0.5)
        out = Vector((math.sin(ang), -math.cos(ang), 0))
        # most of the flock curls toward the camera side and the moon
        drift = Vector((-0.55, -0.25, 0)) * r_.uniform(0.3, 1.0)
        vel = (out * r_.uniform(5.5, 9.0) + drift * 6 + Vector((0, 0, r_.uniform(2.5, 5.0))))
        t0 = STRIKES[0] + r_.uniform(0.02, 0.6)
        b = bpy.data.objects.new(f"Bird{i}", body); C.objects.link(b)
        wl = bpy.data.objects.new(f"BirdWL{i}", wing); C.objects.link(wl); wl.parent = b
        wr = bpy.data.objects.new(f"BirdWR{i}", wing); C.objects.link(wr); wr.parent = b; wr.scale = (-1, 1, 1)
        for o in (b, wl, wr):
            o.visible_shadow = False
        sc = r_.uniform(1.0, 1.5)
        b.scale = (sc, sc, sc)
        birds.append(dict(b=b, wl=wl, wr=wr, start=start, vel=vel, t0=t0, freq=r_.uniform(5.5, 8.0), ph=r_.uniform(0, 6.28),
                          wob=r_.uniform(0.5, 1.4), seed=r_.uniform(0, 100)))
    return C, birds


def animate_birds(birds):
    for bd in birds:
        locs = [[], [], []]
        rots = [[], [], []]
        wing = []
        prev = None
        for f in range(1, NF + 1):
            t = (f - 1) / FPS
            u = t - bd["t0"]
            if u <= 0:
                p = bd["start"].copy()
                d = Vector((bd["vel"].x, bd["vel"].y, 0)).normalized()
                flap = 0.0
            else:
                acc = Vector((0, 0, -0.6))
                wob = Vector((math.sin(u * bd["wob"] + bd["seed"]) * 2.2, math.cos(u * bd["wob"] * 0.8 + bd["seed"]) * 2.2, math.sin(u * 1.3 + bd["seed"]) * 0.8))
                p = bd["start"] + bd["vel"] * u + 0.5 * acc * u * u + wob * min(1.0, u / 1.5) * u * 0.5
                d = bd["vel"] + acc * u + Vector((math.cos(u * bd["wob"] + bd["seed"]) * bd["wob"] * 2.2 * 0.5,
                                                  -math.sin(u * bd["wob"] * 0.8 + bd["seed"]) * bd["wob"] * 0.8 * 2.2 * 0.5, 0))
                glide = 0.35 if math.sin(u * 0.9 + bd["seed"]) > 0.6 and u > 1.5 else 1.0
                flap = (0.9 * glide) * math.sin(2 * math.pi * bd["freq"] * u + bd["ph"])
            q = d.to_track_quat("Y", "Z")
            e = q.to_euler("XYZ", prev) if prev else q.to_euler("XYZ")
            prev = e
            for k in range(3):
                locs[k].append(p[k]); rots[k].append(e[k])
            wing.append(flap)
        for k in range(3):
            bake(bd["b"], "location", locs[k], index=k)
            bake(bd["b"], "rotation_euler", rots[k], index=k)
        bake(bd["wl"], "rotation_euler", [w for w in wing], index=1)
        bake(bd["wr"], "rotation_euler", [-w for w in wing], index=1)


# ============================================================================ cameras
def ease_kind(u, kind):
    u = max(0.0, min(1.0, u))
    if kind == "io":
        return u * u * (3 - 2 * u)
    if kind == "out":
        return 1 - (1 - u) ** 3
    if kind == "in":
        return u ** 3
    if kind == "accel":
        return 0.2 * u + 0.8 * u ** 3
    if kind == "soft":   # gentle start, linear finish
        return u * u * (2 - u) if u < 1 else 1
    return u


def shake(t, starts, amp, decay=0.7, freq=1.0):
    v = Vector((0.0, 0.0, 0.0))
    for t0 in starts:
        d = t - t0
        if d < 0:
            continue
        e = amp * math.exp(-d / decay)
        v += Vector((math.sin(d * 41 * freq + 0.3) + 0.5 * math.sin(d * 67 * freq + 1.1),
                     math.sin(d * 37 * freq + 2.0) + 0.5 * math.sin(d * 59 * freq + 0.4),
                     math.sin(d * 29 * freq + 1.7))) * e
    return v


def make_camera(name, keys, lens, fstop, focus, t_range, roll_fn=None, shake_fn=None, sensor=36):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.03
    cd.clip_end = 2000
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    if isinstance(focus, bpy.types.Object):
        cd.dof.focus_object = focus
    else:
        fe = empty(name + "Focus", focus, bpy.context.scene.collection)
        cd.dof.focus_object = fe
    f0 = F(t_range[0])
    f1 = min(NF, F(t_range[1]) + 1)
    locs = [[], [], []]
    rots = [[], [], []]
    prev = None
    for f in range(f0, f1 + 1):
        t = (f - 1) / FPS
        # find segment
        if t <= keys[0][0]:
            pos, tgt = Vector(keys[0][1]), Vector(keys[0][2])
        elif t >= keys[-1][0]:
            pos, tgt = Vector(keys[-1][1]), Vector(keys[-1][2])
        else:
            for i in range(len(keys) - 1):
                ta, tb = keys[i][0], keys[i + 1][0]
                if ta <= t <= tb:
                    kind = keys[i + 1][3] if len(keys[i + 1]) > 3 else "io"
                    u = ease_kind((t - ta) / (tb - ta), kind)
                    pos = Vector(keys[i][1]).lerp(Vector(keys[i + 1][1]), u)
                    tgt = Vector(keys[i][2]).lerp(Vector(keys[i + 1][2]), u)
                    break
        sh = shake_fn(t) if shake_fn else Vector((0, 0, 0))
        roll = roll_fn(t) if roll_fn else 0.0
        e = aim_euler(pos, tgt, prev, roll + sh.z * 0.3)
        e = Euler((e.x + sh.x * 0.6, e.y, e.z + sh.y * 0.6), "XYZ")
        prev = e
        p = pos + Vector((sh.y, 0, sh.x)) * 0.8
        for k in range(3):
            locs[k].append(p[k]); rots[k].append(e[k])
    for k in range(3):
        bake(cam, "location", locs[k], index=k, frame_start=f0)
        bake(cam, "rotation_euler", rots[k], index=k, frame_start=f0)
    return cam


# ============================================================================ build everything
t_build = time.time()
scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
setup_render(scene)
setup_world(scene)
setup_compositor(scene)
M = build_materials()
TOWER, FACES = build_tower(M)
CLOCK, CROOT, HANDS, GLASS = build_clock(M)
for k in (1, 2, 3):   # the other three dials are instances of the front one
    inst = bpy.data.objects.new(f"ClockInst{k}", None)
    inst.instance_type = "COLLECTION"
    inst.instance_collection = CLOCK
    inst.rotation_euler = (0, 0, k * math.pi / 2)
    TOWER.objects.link(inst)
INTER = build_interior(M)
MOVE, MV, GEARS, ANCHOR, PEND, GB_WORLD = build_movement(M)
BELLC, BELL, HAMMER = build_bell(M)
CITY = build_city(M)
LIGHTS, LT = build_lights(M, CROOT)
DUSTC = coll("Dust")
bpy.context.view_layer.update()
_esc = GEARS["GEsc"]["piv"].matrix_world.translation.copy()
AVOID = [((2.9, 4.4, 42.3), (-1.5, 4.2, 42.8), 1.7),
         (tuple(_esc + Vector((1.5, 2.4, 1.0))), tuple(_esc + Vector((0.5, 0.9, 0.7))), 1.2)]
DUST = [dust_cloud(f"Dust{i}", DUSTC, M, 300, ((-4.6, -4.8, CS0 + 0.6), (4.6, 4.6, CS1 - 0.8)), seed=i + 3, avoid=AVOID) for i in range(4)]
FALL = [dust_cloud(f"Fall{i}", DUSTC, M, 160, ((-3.5, -3.5, BEL1 - 1.6), (3.5, 3.5, BEL1 - 0.9)), 0.006, 0.015, seed=40 + i,
                   avoid=[((3.4, -3.1, 53.1), (3.6, -3.4, 52.9), 1.2)]) for i in range(5)]
BIRDC, BIRDS = build_birds(M)

# god-ray spot: everything but the dial glass casts its shadows (light linking blocker collection)
blockers = bpy.data.collections.new("BeamBlockers")
for c_ in (TOWER, CLOCK, MOVE, INTER):
    for o in c_.all_objects:
        if o is not GLASS and o.type in ("MESH", "FONT", "CURVE") and o.visible_shadow:
            blockers.objects.link(o)
# glowing backing outside the front dial: only the inside shots see it (through the glass, behind the numerals)
BACKC = coll("DialBacking")
backing = ob("DialBacking", mesh_cyl("DialBacking", DIAL_R + 0.3, 0.02, verts=96, origin="center"), BACKC,
             (0, GLASS_Y - 0.95, DIAL_Z), R90X, mat=emission("BackingGlow", hexcol("#FFE4B5"), float(arg("--backing", 7.0))))
backing.visible_shadow = False
bake_steps(backing, "hide_render", [(F(t0), 0.0 if n in ("inside", "macro") else 1.0) for n, t0 in SHOTS])
try:
    LT["beam"].light_linking.blocker_collection = blockers
    print("[scene] beam blocker collection set:", len(blockers.objects), "objects")
except Exception as e:
    print("[scene] light linking failed:", e)

# ---------------------------------------------------------------- animation: clock, movement, bell, lights, dust
fr = range(1, NF + 1)
ts = [(f - 1) / FPS for f in fr]
bake(HANDS["Second"], "rotation_euler", [-sec_angle(t) * D2R for t in ts], index=2)
bake(HANDS["Minute"], "rotation_euler", [-min_angle(t) * D2R for t in ts], index=2)
bake(HANDS["Hour"], "rotation_euler", [-hour_angle(t) * D2R for t in ts], index=2)


def mesh_phase(a, b, phi, th_a):
    """rotation of gear b (meshing with a) given a's rotation th_a; phi = direction a->b."""
    na, nb = a["n"], b["n"]
    return phi + math.pi - math.pi / nb - (na / nb) * (th_a - phi)


def to_cw(phi):     # gears are built with tooth 0 at 'up' and angles clockwise; convert math angle to that frame
    return math.pi / 2 - phi


esc_rot = [-6.0 * D2R * steps(t) for t in ts]
G = GEARS
phi_cm = math.atan2(G["GC"]["pos"].y - G["GMid"]["pos"].y, G["GC"]["pos"].x - G["GMid"]["pos"].x)
phi_bs = math.atan2(G["GSmall"]["pos"].y - G["GBig"]["pos"].y, G["GSmall"]["pos"].x - G["GBig"]["pos"].x)
phi_st = math.atan2(G["GStrikeP"]["pos"].y - G["GStrike"]["pos"].y, G["GStrikeP"]["pos"].x - G["GStrike"]["pos"].x)
# our meshes rotate about +Z (counter-clockwise positive) with tooth 0 on +Y; work in math angles (from +X, CCW)
c_rot = esc_rot
mid_rot = []
for th in c_rot:
    # GC drives GMid: theta_mid = phi' + pi - pi/n_mid - (n_c/n_mid)(theta_c - phi') in math-angle convention
    th_c_math = th + math.pi / 2
    phi = phi_cm + math.pi   # direction GC -> GMid
    th_m_math = phi + math.pi - math.pi / G["GMid"]["n"] - (G["GC"]["n"] / G["GMid"]["n"]) * (th_c_math - phi)
    mid_rot.append(th_m_math - math.pi / 2)
big_rot = []
for th in mid_rot:
    th_s_math = th + math.pi / 2
    phi = phi_bs + math.pi   # GSmall -> GBig
    th_b_math = phi + math.pi - math.pi / G["GBig"]["n"] - (G["GSmall"]["n"] / G["GBig"]["n"]) * (th_s_math - phi)
    big_rot.append(th_b_math - math.pi / 2)
strike_drive = [0.0 if t < MIDNIGHT else -2.4 * (1 - math.exp(-(t - MIDNIGHT) / 0.5)) - 0.15 * max(0, t - MIDNIGHT) for t in ts]
st_rot = [s + 0.0 for s in strike_drive]
stp_rot = []
for th in st_rot:
    th_a = th + math.pi / 2
    phi = phi_st
    th_b = phi + math.pi - math.pi / G["GStrikeP"]["n"] - (G["GStrike"]["n"] / G["GStrikeP"]["n"]) * (th_a - phi)
    stp_rot.append(th_b - math.pi / 2)
for name, vals in (("GEsc", esc_rot), ("GC", c_rot), ("GMid", mid_rot), ("GSmall", mid_rot), ("GBig", big_rot),
                   ("GStrike", st_rot), ("GStrikeP", stp_rot)):
    bake(G[name]["piv"], "rotation_euler", vals, index=2)
bake(ANCHOR, "rotation_euler", [0.07 * math.cos(math.pi * t) for t in ts], index=2)
bake(PEND, "rotation_euler", [pend_angle(t) for t in ts], index=2)

# hammer: cocks back during the whirr, drops onto the lip at each strike
ham = []
for t in ts:
    a = 0.0
    for s in STRIKES:
        lift0 = s - 0.9
        if lift0 <= t < s - 0.08:
            a = max(a, 0.75 * ease((t - lift0) / 0.7))
        elif s - 0.08 <= t < s:
            a = max(a, 0.75 * (1 - ((t - (s - 0.08)) / 0.08) ** 2))
        elif s <= t < s + 0.5:
            d = t - s
            a = max(a, 0.12 * math.exp(-d / 0.08) * abs(math.sin(d * 40)))
    ham.append(a)
bake(HAMMER, "rotation_euler", ham, index=1)
# bell shivers a little on impact
bake(BELL, "rotation_euler", [0.006 * pulse(t, STRIKES, 0.6) * math.sin(t * 90) for t in ts], index=0)

dial_str = [5.0 * (1 + 0.5 * pulse(t, (MIDNIGHT,), 0.35) + 0.9 * pulse(t, STRIKES, 0.8)) for t in ts]
bake(M["dial"].node_tree, 'nodes["DIALSTR"].outputs[0].default_value', dial_str)
bake(LT["bel"].data, "energy", [1400 * (1 + 4.0 * pulse(t, STRIKES, 0.7)) for t in ts])
bake(LT["flash"].data, "energy", [2500 * pulse(t, STRIKES, 0.25, 0.01) for t in ts])
# beam spot only matters inside; keep it off for the exterior shots
beam_E = LT["beam"].data.energy
keys = [(F(t0), beam_E if n in ("inside", "macro") else 0.0) for n, t0 in SHOTS]
bake_steps(LT["beam"].data, "energy", keys)
bake_steps(LT["rim"].data, "energy", [(F(t0), 1500.0 if n in ("inside", "macro") else 0.0) for n, t0 in SHOTS])
vol_keys = [(F(t0), 260.0 if n in EXTERIOR else 40.0) for n, t0 in SHOTS]
bake_steps(scene, "eevee.volumetric_end", vol_keys)
bake(scene.world.node_tree, 'nodes["CLOUDMAP"].inputs[1].default_value', [0.006 * t for t in ts], index=0)

for i, d in enumerate(DUST):
    r_ = random.Random(100 + i)
    sp = Vector((r_.uniform(-0.03, 0.03), r_.uniform(-0.02, 0.02), r_.uniform(-0.02, 0.02)))
    ph = r_.uniform(0, 6.28)
    for k in range(3):
        bake(d, "location", [sp[k] * t + 0.12 * math.sin(0.35 * t + ph + k) for t in ts], index=k)
for i, d in enumerate(FALL):
    r_ = random.Random(200 + i)
    v = r_.uniform(0.35, 0.9)
    dx = r_.uniform(-0.15, 0.15)
    t_rel = STRIKES[0] + r_.uniform(0.0, 0.15)
    bake(d, "location", [0.0 if t < t_rel else -(v * (t - t_rel) + 0.6 * (t - t_rel) ** 2) for t in ts], index=2)
    bake(d, "location", [0.0 if t < t_rel else dx * (t - t_rel) for t in ts], index=0)
    bake(d, "scale", [0.0 if t < t_rel - 0.01 else 1.0 for t in ts], index=0, interp="CONSTANT")
    bake(d, "scale", [0.0 if t < t_rel - 0.01 else 1.0 for t in ts], index=1, interp="CONSTANT")
    bake(d, "scale", [0.0 if t < t_rel - 0.01 else 1.0 for t in ts], index=2, interp="CONSTANT")
animate_birds(BIRDS)

# ---------------------------------------------------------------- cameras + cuts
bpy.context.view_layer.update()
esc_w = G["GEsc"]["piv"].matrix_world.translation.copy()
print("[scene] escape wheel at", tuple(round(c, 2) for c in esc_w), "anchor", tuple(round(c, 2) for c in ANCHOR.matrix_world.translation),
      "pendulum", tuple(round(c, 2) for c in PEND.matrix_world.translation))
dial_c = Vector((0, GLASS_Y, DIAL_Z))
CAMS = {}
CAMS["est"] = make_camera("CamEst", [(0.0, (-24, -96, 8), (0, 0, 37)), (7.0, (-8.5, -52, 31), (0, -2, 47.5), "io")],
                          30, 11, (0, -5, 46), (0.0, 7.0), roll_fn=lambda t: 0.035 - 0.005 * t)
CAMS["face"] = make_camera("CamFace", [(7.0, (5.2, -22.5, 43.4), (0, GLASS_Y, 46.2)),
                                       (12.0, (0.95, -10.6, 45.62), (0.2, GLASS_Y, 45.9), "soft"),
                                       (12.5, (0.6, -4.7, 45.2), (0.45, 4.0, 44.8), "accel")],
                           38, 5.6, dial_c, (7.0, 12.5))
CAMS["inside"] = make_camera("CamInside", [(12.5, (2.9, 4.4, 42.3), (-0.3, GLASS_Y, 46.6)),
                                           (18.0, (-1.5, 4.2, 42.8), (0.5, GLASS_Y, 46.2), "io")],
                             22, 5.6, (0, GLASS_Y, DIAL_Z), (12.5, 18.0))
mpos0 = esc_w + Vector((1.5, 2.4, 1.0))
mpos1 = esc_w + Vector((1.15, 1.95, 0.9))
CAMS["macro"] = make_camera("CamMacro", [(18.0, tuple(mpos0), tuple(esc_w + Vector((0, 0, 0.62)))),
                                         (20.5, tuple(mpos1), tuple(esc_w + Vector((-0.05, 0, 0.66))), "io")],
                            50, 2.8, tuple(esc_w + Vector((0, 0, 0.7))), (18.0, 20.5))
CAMS["count"] = make_camera("CamCount", [(20.5, (-5.6, -17.5, 37.4), (0, GLASS_Y, 46.6)),
                                         (22.75, (-4.2, -14.2, 38.6), (0, GLASS_Y, 46.4), "out")],
                            34, 6.3, dial_c, (20.5, 22.75), shake_fn=lambda t: shake(t, (MIDNIGHT,), 0.006, 0.25))
CAMS["bell"] = make_camera("CamBell", [(22.75, (3.4, -3.1, 53.1), (0.9, -0.1, 55.9)),
                                       (25.0, (3.6, -3.4, 52.95), (0.8, -0.1, 56.2), "io")],
                           20, 2.8, (1.3, -0.3, 55.0), (22.75, 25.0), shake_fn=lambda t: shake(t, (STRIKES[0],), 0.03, 0.45))
CAMS["finale"] = make_camera("CamFinale", [(25.0, (13.0, -38.0, 47.5), (-2, 0, 55.0)),
                                           (30.0, (36.0, -104.0, 61.0), (-6, 0, 54.0), "out")],
                             26, 11, (0, -5, 50), (25.0, 30.0), shake_fn=lambda t: shake(t, STRIKES, 0.01, 0.5))
scene.timeline_markers.clear()
for n, t0 in SHOTS:
    mk = scene.timeline_markers.new(n, frame=F(t0))
    mk.camera = CAMS[n]
scene.camera = CAMS["est"]
print(f"[scene] built in {time.time() - t_build:.1f} s; objects={len(bpy.data.objects)}")

# ---------------------------------------------------------------- outputs
frames_dir = BUILD / "frames"
frames_dir.mkdir(exist_ok=True)
scene.render.filepath = str(frames_dir) + "/f_"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.compression = 30
scene.render.use_overwrite = False
scene.render.use_placeholder = True

stills = arg("--stills")
if stills and stills is not True:
    sd = BUILD / "stills"
    sd.mkdir(exist_ok=True)
    for tt in [float(s) for s in stills.split(",")]:
        f = F(tt)
        scene.frame_set(f)
        # pick the camera the markers would choose
        cam_name = [n for n, t0 in SHOTS if t0 <= tt][-1]
        scene.camera = CAMS[cam_name]
        scene.render.filepath = str(sd / f"still_{tt:05.2f}.png")
        scene.render.use_overwrite = True
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[still] t={tt:.2f} ({cam_name}) {time.time() - t0:.1f} s")
    scene.render.filepath = str(frames_dir) + "/f_"
    scene.render.use_overwrite = False
    scene.camera = CAMS["est"]

if arg("--save"):
    bpy.ops.wm.save_as_mainfile(filepath=str(BUILD / "clock.blend"))
    print("[scene] saved", BUILD / "clock.blend")
