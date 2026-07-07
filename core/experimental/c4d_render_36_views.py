"""
c4d_render_36_views.py — version 3 (C4D 2025)
-----------------------------------------------
Stratégie : rig animé (Null + Caméra) sur 36 frames
Le script crée le rig et configure le rendu.
L'utilisateur lance le rendu depuis C4D normalement.
"""

import c4d
import os
import math
import json

# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
N_VIEWS       = 36
RENDER_W      = 512
RENDER_H      = 512
FOV_DEG       = 40.0
DIST_FACTOR   = 3.5
HEIGHT_FACTOR = 0.3
FPS           = 1   # 1 frame = 1 vue → frame 0 = 0°, frame 35 = 350°


# ─────────────────────────────────────────
# Bounding box récursive
# ─────────────────────────────────────────
def get_world_bbox(obj):
    INF = float('inf')
    mn  = [INF, INF, INF]
    mx  = [-INF, -INF, -INF]

    def recurse(o):
        if isinstance(o, c4d.PointObject):
            mg = o.GetMg()
            for pt in o.GetAllPoints():
                wp = mg * pt
                mn[0] = min(mn[0], wp.x); mx[0] = max(mx[0], wp.x)
                mn[1] = min(mn[1], wp.y); mx[1] = max(mx[1], wp.y)
                mn[2] = min(mn[2], wp.z); mx[2] = max(mx[2], wp.z)
        child = o.GetDown()
        while child:
            recurse(child)
            child = child.GetNext()

    recurse(obj)

    if mn[0] == INF:
        pos = obj.GetAbsPos()
        return pos, pos, pos, 1.0

    center = c4d.Vector(
        (mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2
    )
    size = max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
    return c4d.Vector(*mn), c4d.Vector(*mx), center, size


# ─────────────────────────────────────────
# Matrice look-at
# ─────────────────────────────────────────
def look_at_matrix(cam_pos, target):
    fwd   = (target - cam_pos).GetNormalized()
    up    = c4d.Vector(0, 1, 0)
    right = fwd.Cross(up).GetNormalized()
    if right.GetLength() < 0.001:
        up    = c4d.Vector(0, 0, 1)
        right = fwd.Cross(up).GetNormalized()
    up2   = right.Cross(fwd).GetNormalized()
    mat   = c4d.Matrix()
    mat.v1  = right
    mat.v2  = up2
    mat.v3  = -fwd
    mat.off = cam_pos
    return mat


def matrix_to_nerf(m):
    return [
        [m.v1.x, m.v2.x, m.v3.x, m.off.x],
        [m.v1.y, m.v2.y, m.v3.y, m.off.y],
        [m.v1.z, m.v2.z, m.v3.z, m.off.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ─────────────────────────────────────────
# Ajouter une keyframe sur un track
# ─────────────────────────────────────────
CINTERP_LINEAR = getattr(c4d, 'CINTERP_LINEAR', 3)

def add_keyframe(obj, desc_id, frame, value, fps):
    track = obj.FindCTrack(desc_id)
    if track is None:
        track = c4d.CTrack(obj, desc_id)
        obj.InsertTrackSorted(track)
    curve = track.GetCurve()
    key   = curve.AddKey(c4d.BaseTime(frame, fps))["key"]
    key.SetValue(curve, value)
    # CINTERP_LINEAR = 3 en C4D 2025 (constante supprimée du module)
    CINTERP_LINEAR = getattr(c4d, 'CINTERP_LINEAR', 3)
    key.SetInterpolation(curve, CINTERP_LINEAR)


# ─────────────────────────────────────────
# Script principal
# ─────────────────────────────────────────
def main():
    doc = c4d.documents.GetActiveDocument()

    # 1. Objet sélectionné
    obj = doc.GetActiveObject()
    if obj is None:
        c4d.gui.MessageDialog(
            "Aucun objet sélectionné.\n"
            "Clique sur l'objet dans la hiérarchie puis relance."
        )
        return

    print(f"[GS Render] Objet : {obj.GetName()}")

    # 2. Bounding box
    mn, mx, center, size = get_world_bbox(obj)
    if size < 0.001:
        c4d.gui.MessageDialog("L'objet semble vide (taille ≈ 0).")
        return

    distance = size * DIST_FACTOR
    height   = size * HEIGHT_FACTOR
    print(f"[GS Render] Centre: {center} | Taille: {size:.3f}")
    print(f"[GS Render] Distance caméra: {distance:.3f}")

    # 3. Dossier de sortie
    output_dir = c4d.storage.LoadDialog(
        type  = c4d.FILESELECTTYPE_ANYTHING,
        title = "Dossier de sortie pour les renders",
        flags = c4d.FILESELECT_DIRECTORY,
    )
    if not output_dir:
        return

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 4. Créer le Null pivot au centre de l'objet
    null = c4d.BaseObject(c4d.Onull)
    null.SetName("GS_Pivot")
    null.SetAbsPos(center)
    doc.InsertObject(null)

    # 5. Créer la caméra comme enfant du Null
    cam = c4d.BaseObject(c4d.Ocamera)
    cam.SetName("GS_Camera_36views")
    cam[c4d.CAMERAOBJECT_FOV] = math.radians(FOV_DEG)
    # Position locale : décalée sur Z + légèrement en hauteur
    cam.SetRelPos(c4d.Vector(0, height, -distance))
    # Inclinaison pour regarder vers le centre
    tilt = math.atan2(height, distance)
    cam.SetRelRot(c4d.Vector(tilt, 0, 0))
    cam.InsertUnder(null)

    # 6. Configurer le document (FPS + plage de frames)
    doc.SetFps(FPS)
    doc.SetMinTime(c4d.BaseTime(0, FPS))
    doc.SetMaxTime(c4d.BaseTime(N_VIEWS - 1, FPS))
    doc.SetLoopMinTime(c4d.BaseTime(0, FPS))
    doc.SetLoopMaxTime(c4d.BaseTime(N_VIEWS - 1, FPS))

    # 7. Animer la rotation du Null (Y) sur 36 frames
    rot_desc = c4d.DescID(
        c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0),
        c4d.DescLevel(c4d.VECTOR_Y),
    )
    for i in range(N_VIEWS + 1):
        angle = math.radians(i * 360.0 / N_VIEWS)
        add_keyframe(null, rot_desc, i, angle, FPS)

    print(f"[GS Render] {N_VIEWS} keyframes créées sur GS_Pivot")

    # 8. Configurer le rendu
    rd = doc.GetActiveRenderData()
    rd[c4d.RDATA_XRES]      = RENDER_W
    rd[c4d.RDATA_YRES]      = RENDER_H
    rd[c4d.RDATA_FRAMEFROM] = c4d.BaseTime(0, FPS)
    rd[c4d.RDATA_FRAMETO]   = c4d.BaseTime(N_VIEWS - 1, FPS)
    rd[c4d.RDATA_FORMAT]    = c4d.FILTER_PNG
    rd[c4d.RDATA_PATH]      = os.path.join(images_dir, "####")

    # 9. Générer transforms.json (positions caméra exactes)
    fl = (RENDER_W / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
    transforms = {
        "camera_angle_x": math.radians(FOV_DEG),
        "fl_x": fl, "fl_y": fl,
        "cx": RENDER_W / 2.0, "cy": RENDER_H / 2.0,
        "w": RENDER_W, "h": RENDER_H,
        "frames": [],
    }

    for i in range(N_VIEWS):
        angle_rad = math.radians(i * 360.0 / N_VIEWS)
        cam_pos = c4d.Vector(
            center.x + distance * math.sin(angle_rad),
            center.y + height,
            center.z + distance * math.cos(angle_rad),
        )
        mat = look_at_matrix(cam_pos, center)
        transforms["frames"].append({
            "file_path": f"./images/{i:04d}",
            "transform_matrix": matrix_to_nerf(mat),
        })

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=2)

    c4d.EventAdd()

    # 10. Message final avec instructions
    msg = (
        f"✅ Rig créé : GS_Pivot + GS_Camera_36views\n\n"
        f"ÉTAPES SUIVANTES dans C4D :\n\n"
        f"1. Dans la hiérarchie → sélectionne 'GS_Camera_36views'\n"
        f"2. Clic droit → 'Utiliser comme caméra de scène'\n"
        f"   (ou Menu Caméra → Utiliser comme caméra de rendu)\n\n"
        f"3. Rendu → Rendu dans la Visionneuse d'images\n"
        f"   → C4D rend les 36 frames automatiquement\n\n"
        f"4. Les images sont dans :\n"
        f"   {images_dir}\n\n"
        f"5. Upload le dossier sur Google Drive :\n"
        f"   {output_dir}"
    )
    c4d.gui.MessageDialog(msg)
    print(f"[GS Render] transforms.json → {output_dir}")
    print("[GS Render] Rig prêt — lance le rendu depuis C4D")


if __name__ == "__main__":
    main()