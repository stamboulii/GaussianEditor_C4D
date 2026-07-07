"""
core/actions/merge_save.py
---------------------------
Action POST /merge_and_save

Stratégie :
  1. Lire tous les PLY sources (espace TripoSplat ~0.809 unités)
  2. Normaliser chaque objet à TARGET_SIZE = 0.809 (espace naturel TripoSplat)
  3. Convertir les offsets C4D (espace 200 unités) → espace PLY
  4. Sauvegarder avec vrais SH
"""

import os
import traceback
import numpy as np

from core.gs_config import get_out_path

# Taille cible = taille naturelle des PLY TripoSplat
# (validée : guitare PLY original = 0.809, s'affiche bien dans SuperSplat)
TARGET_SIZE = 0.809


def action_merge_and_save(objects: list, out_path: str = "") -> dict:
    if not objects:
        return {"ok": False, "error": "Aucun objet fourni"}

    if not out_path:
        out_path = get_out_path("merged.ply")

    def _to_wsl(p):
        if not p or p.startswith("/"):
            return p
        p = p.replace("\\", "/")
        if len(p) >= 2 and p[1] == ":":
            return f"/mnt/{p[0].lower()}/{p[2:].lstrip('/')}"
        return p

    try:
        # ── Étape 1 : lire tous les PLY ──────────────────────
        ply_data = []
        for obj_info in objects:
            ply_path = _to_wsl(obj_info.get("ply_path", ""))
            if not os.path.isfile(ply_path):
                print(f"[MergeSave] Introuvable : {ply_path}")
                continue

            with open(ply_path, "rb") as f:
                props, n = [], 0
                while True:
                    line = f.readline().decode("utf-8", errors="replace").strip()
                    if line.startswith("element vertex"):
                        n = int(line.split()[-1])
                    elif line.startswith("property float"):
                        props.append(line.split()[-1])
                    elif line == "end_header":
                        break
                data = np.frombuffer(
                    f.read(n * len(props) * 4), dtype="<f4"
                ).reshape(n, len(props)).copy()

            pidx = {name: i for i, name in enumerate(props)}
            xyz  = data[:, [pidx["x"], pidx["y"], pidx["z"]]].copy()
            size = float((xyz.max(0) - xyz.min(0)).max())

            ply_data.append({
                "name":   os.path.basename(ply_path),
                "data":   data,
                "pidx":   pidx,
                "xyz":    xyz,
                "size":   size,
                "center": xyz.mean(0),
                "n":      n,
                "info":   obj_info,
            })
            print(f"[MergeSave] Lu : {os.path.basename(ply_path)} "
                  f"({n:,} pts, size={size:.4f})")

        if not ply_data:
            return {"ok": False, "error": "Aucun PLY valide"}

        # ── Étape 2 : appliquer transformations ──────────────
        ref_props = [
            "x", "y", "z",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3"
        ]

        all_rows = []
        for pd in ply_data:
            info       = pd["info"]
            xyz_orig   = pd["xyz"]
            data       = pd["data"]
            pidx       = pd["pidx"]
            n          = pd["n"]
            orig_center = pd["center"]
            orig_size   = pd["size"]

            # Position C4D (mg.off) — en unités C4D (~200 unités)
            dx = float(info.get("dx", 0.0))
            dy = float(info.get("dy", 0.0))
            dz = float(info.get("dz", 0.0))
            user_scale = float(info.get("scale", 1.0))

            # Facteur de normalisation : ramener cet objet à TARGET_SIZE
            obj_norm = TARGET_SIZE / orig_size if orig_size > 1e-6 else 1.0

            # Centrer + normaliser à TARGET_SIZE
            xyz = (xyz_orig - orig_center) * obj_norm * user_scale

            # Convertir offset C4D → espace PLY
            # C4D normalise à 200 unités, PLY fait TARGET_SIZE unités
            # 1 unite C4D = TARGET_SIZE / 200 unites PLY
            c4d_to_ply = TARGET_SIZE / 200.0
            xyz[:, 0] += dx * c4d_to_ply
            xyz[:, 1] += dy * c4d_to_ply
            xyz[:, 2] += dz * c4d_to_ply

            row = np.zeros((n, len(ref_props)), dtype=np.float32)
            row[:, 0] = xyz[:, 0]
            row[:, 1] = xyz[:, 1]
            row[:, 2] = xyz[:, 2]

            # Couleurs SH originales (vrais SH bruts)
            for j, prop in enumerate(["f_dc_0","f_dc_1","f_dc_2"], start=3):
                if prop in pidx:
                    row[:, j] = data[:, pidx[prop]]

            # Opacité (logit-space)
            if "opacity" in pidx:
                row[:, 6] = data[:, pidx["opacity"]]
            else:
                row[:, 6] = 2.0

            # Scales — adapter à la nouvelle échelle
            log_scale_adj = np.log(obj_norm * user_scale)
            for j, prop in enumerate(["scale_0","scale_1","scale_2"], start=7):
                if prop in pidx:
                    row[:, j] = data[:, pidx[prop]] + log_scale_adj
                else:
                    row[:, j] = np.log(0.01) + log_scale_adj

            # Rotations
            for j, prop in enumerate(["rot_0","rot_1","rot_2","rot_3"], start=10):
                if prop in pidx:
                    row[:, j] = data[:, pidx[prop]]
            if "rot_0" not in pidx:
                row[:, 10] = 1.0

            all_rows.append(row)
            print(f"[MergeSave] {pd['name']} : obj_norm={obj_norm:.4f} "
                  f"off_ply=({dx*c4d_to_ply:.3f},{dy*c4d_to_ply:.3f},{dz*c4d_to_ply:.3f})")

        # ── Étape 3 : fusionner et sauvegarder ───────────────
        merged  = np.concatenate(all_rows, axis=0)
        n_total = len(merged)

        xyz_out = merged[:, :3]
        print(f"[MergeSave] Total : {n_total:,} gaussiens")
        print(f"[MergeSave] X:[{xyz_out[:,0].min():.3f},{xyz_out[:,0].max():.3f}] "
              f"Y:[{xyz_out[:,1].min():.3f},{xyz_out[:,1].max():.3f}] "
              f"Z:[{xyz_out[:,2].min():.3f},{xyz_out[:,2].max():.3f}]")

        header  = "ply\nformat binary_little_endian 1.0\n"
        header += "comment GaussianEditor C4D merged\n"
        header += f"element vertex {n_total}\n"
        for p in ref_props:
            header += f"property float {p}\n"
        header += "end_header\n"

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(header.encode("utf-8"))
            f.write(merged.astype("<f4").tobytes())

        print(f"[MergeSave] Sauvegardé : {out_path}")

        return {
            "ok":       True,
            "action":   "merge_and_save",
            "ply_path": os.path.abspath(out_path),
            "n":        n_total,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}