"""
core/gsplat_viewer.py
---------------------
Conversion entre GaussianData et objets Cinema 4D natifs.
Compatible C4D 2025 (CPython 3.11).

Fix juillet 2026 v10 :
  - Suppression IsPerPointMode() inexistant dans C4D 2025
  - c4d_to_gaussians lit les couleurs depuis VertexColorTag correctement
  - Fallback vers couleur moyenne si tag absent
"""

try:
    import c4d
    INSIDE_C4D = True
except ImportError:
    INSIDE_C4D = False


def _has_data(arr) -> bool:
    if arr is None:
        return False
    try:
        return len(arr) > 0
    except Exception:
        return False


GS_ATTR_SOURCE_PATH = 10001
GS_ATTR_N_GAUSSIANS = 10002
GS_ATTR_HAS_COLORS  = 10003
GS_ATTR_HAS_OPACITY = 10004
GS_ATTR_HAS_SCALES  = 10005
GS_ATTR_VERSION     = 10006

GS_VIEWPORT_LIMIT = 50000


def gaussians_to_c4d(doc, gd, max_points=None, name="Gaussian Splatting"):
    if not INSIDE_C4D:
        raise EnvironmentError("Doit etre appele depuis Cinema 4D.")
    if gd.n == 0:
        raise ValueError("GaussianData vide.")

    import numpy as _np

    n_total   = gd.n
    n_view    = min(n_total, max_points if max_points else GS_VIEWPORT_LIMIT)
    name_view = f"{name} [{n_view:,}/{n_total:,}pts]" if n_view < n_total else name

    step    = max(1, n_total // n_view)
    indices = list(range(0, n_total, step))[:n_view]
    n       = len(indices)

    xyz     = gd.xyz
    opacity = gd.opacity if _has_data(gd.opacity) else None

    # Conversion SH → RGB [0, 1]
    if _has_data(gd.colors_sh):
        colors_arr = _np.array(gd.colors_sh, dtype=_np.float32)
        colors_rgb = _np.clip(0.5 + 0.282095 * colors_arr, 0.0, 1.0)
    elif _has_data(gd.colors_rgb):
        colors_rgb = _np.clip(_np.array(gd.colors_rgb, dtype=_np.float32), 0.0, 1.0)
    else:
        colors_rgb = _np.full((n_total, 3), 0.5, dtype=_np.float32)

    # Normalisation échelle
    pts_arr     = _np.array([xyz[i] for i in indices], dtype=_np.float32)
    _mn         = pts_arr.min(axis=0)
    _mx         = pts_arr.max(axis=0)
    _scene_size = float((_mx - _mn).max())
    _scale      = 200.0 / _scene_size if _scene_size > 0.001 else 1.0

    # SplineObject
    spl = c4d.SplineObject(n, c4d.SPLINETYPE_LINEAR)
    if spl is None:
        raise RuntimeError("Impossible de creer le SplineObject.")
    spl.SetName(name_view)

    all_pts = [
        c4d.Vector(
            float(xyz[i][0]) * _scale,
            float(xyz[i][1]) * _scale,
            float(xyz[i][2]) * _scale,
        )
        for i in indices
    ]
    spl.SetAllPoints(all_pts)

    # Couleur moyenne
    sample    = colors_rgb[indices[::max(1, n // 2000)]]
    avg       = sample.mean(axis=0)
    avg_color = c4d.Vector(float(avg[0]), float(avg[1]), float(avg[2]))
    spl[c4d.ID_BASEOBJECT_COLOR]    = avg_color
    spl[c4d.ID_BASEOBJECT_USECOLOR] = 2
    print(f"[Viewer] Couleur moyenne RGB : {avg[0]:.2f} {avg[1]:.2f} {avg[2]:.2f}")

    # VertexColorTag
    try:
        tag = c4d.VertexColorTag(n)
        if tag is not None:
            tag.SetPerPointMode(True)
            data = tag.GetDataAddressW()
            for ii, idx in enumerate(indices):
                cv    = colors_rgb[idx]
                alpha = float(opacity[idx]) if opacity is not None else 1.0
                c4d.VertexColorTag.SetPoint(
                    data, None, None, ii,
                    c4d.Vector4d(float(cv[0]), float(cv[1]), float(cv[2]), alpha)
                )
            spl.InsertTag(tag)
    except Exception as e:
        print(f"[Viewer] VertexColorTag warning : {e}")

    # Métadonnées GS
    bc = spl.GetDataInstance()
    bc.SetString(GS_ATTR_SOURCE_PATH, gd.source_path if hasattr(gd, 'source_path') else "")
    bc.SetInt32(GS_ATTR_N_GAUSSIANS,  n_total)
    bc.SetBool(GS_ATTR_HAS_COLORS,    _has_data(gd.colors_rgb) or _has_data(gd.colors_sh))
    bc.SetBool(GS_ATTR_HAS_OPACITY,   _has_data(gd.opacity))
    bc.SetBool(GS_ATTR_HAS_SCALES,    _has_data(gd.scales))
    bc.SetString(GS_ATTR_VERSION,     "2.0.0")

    spl.Message(c4d.MSG_UPDATE)
    return spl


def c4d_to_gaussians(obj):
    """
    Extrait une GaussianData depuis un objet C4D (SplineObject ou PointObject).
    Compatible C4D 2025 — sans IsPerPointMode().
    """
    if not INSIDE_C4D:
        raise EnvironmentError("Doit etre appele depuis Cinema 4D.")

    from core.ply_io import GaussianData
    import numpy as _np

    # Accepter SplineObject ou PointObject directement
    if not isinstance(obj, c4d.PointObject):
        raise TypeError(f"PointObject attendu, recu : {type(obj)}")

    n = obj.GetPointCount()
    if n == 0:
        raise ValueError("L'objet ne contient aucun point.")

    gd             = GaussianData()
    gd.n           = n
    bc             = obj.GetDataInstance()
    gd.source_path = bc.GetString(GS_ATTR_SOURCE_PATH) or ""

    # Positions
    gd.xyz = []
    for i in range(n):
        pt = obj.GetPoint(i)
        gd.xyz.append((pt.x, pt.y, pt.z))

    # Couleurs depuis VertexColorTag
    # Fix C4D 2025 : pas d'IsPerPointMode() — on essaie directement
    tag = obj.GetTag(c4d.Tvertexcolor)
    if tag is not None:
        try:
            data = tag.GetDataAddressR()
            gd.colors_rgb, gd.colors_sh, gd.opacity = [], [], []
            for i in range(n):
                try:
                    col = c4d.VertexColorTag.GetPoint(data, None, None, i)
                    r   = float(col.x)
                    g   = float(col.y)
                    b   = float(col.z)
                    a   = float(col.w)
                except Exception:
                    r, g, b, a = 0.5, 0.5, 0.5, 1.0
                gd.colors_rgb.append((r, g, b))
                gd.colors_sh.append((
                    (r - 0.5) / 0.282095,
                    (g - 0.5) / 0.282095,
                    (b - 0.5) / 0.282095,
                ))
                gd.opacity.append(a)
        except Exception as e:
            print(f"[Viewer] VertexColorTag lecture echouee ({e}) — fallback couleur objet")
            tag = None

    if tag is None:
        # Fallback : lire la couleur moyenne de l'objet
        obj_color = obj[c4d.ID_BASEOBJECT_COLOR]
        r = float(obj_color.x) if obj_color else 0.5
        g = float(obj_color.y) if obj_color else 0.5
        b = float(obj_color.z) if obj_color else 0.5
        gd.colors_rgb = [(r, g, b)] * n
        gd.colors_sh  = [((r-0.5)/0.282095, (g-0.5)/0.282095, (b-0.5)/0.282095)] * n
        gd.opacity     = [1.0] * n

    gd.scales    = [(0.01, 0.01, 0.01)] * n
    gd.rotations = [(1.0, 0.0, 0.0, 0.0)] * n
    return gd


def get_selected_gaussian_obj(doc):
    if not INSIDE_C4D:
        return None
    sel = doc.GetActiveObjects(c4d.GETACTIVEOBJECTFLAGS_CHILDREN)
    for obj in sel:
        bc = obj.GetDataInstance()
        if bc.GetInt32(GS_ATTR_N_GAUSSIANS) > 0:
            return obj
    return None


def get_gs_info(obj):
    if not INSIDE_C4D:
        return {}
    bc = obj.GetDataInstance()
    return {
        "source_path": bc.GetString(GS_ATTR_SOURCE_PATH),
        "n_gaussians": bc.GetInt32(GS_ATTR_N_GAUSSIANS),
        "has_colors":  bc.GetBool(GS_ATTR_HAS_COLORS),
        "has_opacity": bc.GetBool(GS_ATTR_HAS_OPACITY),
        "has_scales":  bc.GetBool(GS_ATTR_HAS_SCALES),
        "version":     bc.GetString(GS_ATTR_VERSION),
        "point_count": obj.GetPointCount() if hasattr(obj, "GetPointCount") else 0,
    }