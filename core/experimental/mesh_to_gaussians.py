"""
core/mesh_to_gaussians.py
--------------------------
Convertit un mesh 3D (.obj, .glb, .ply) en gaussiens 3D
prêts à fusionner dans une scène Gaussian Splatting.

Pipeline :
  mesh .obj
      ↓
  trimesh.sample_surface(n_points)
      → XYZ sur la surface du mesh
      → couleurs interpolées depuis vertex colors
      ↓
  Adapter scale + position à la scène existante
      ↓
  dict gaussiens (xyz, colors_sh, opacity, scales, rotations)

Usage :
    gaussians = mesh_to_gaussians(
        mesh_path  = "pillow.obj",
        n_points   = 30000,
        scene      = _scene,
        prompt     = "a red pillow on the right",
        log_fn     = print,
    )
"""

import os
import traceback
from typing import Optional, Callable

import numpy as np


def mesh_to_gaussians(
    mesh_path:  str,
    scene:      Optional[dict] = None,
    prompt:     str = "",
    position:   Optional[dict] = None,
    log_fn:     Optional[Callable] = None,
    image_path: str = "",
) -> dict:
    """
    Convertit un mesh 3D en gaussiens prêts pour la fusion.
    Le nombre de gaussiens est calculé automatiquement
    selon la surface réelle du mesh et l'échelle de la scène.

    Args:
        mesh_path : chemin vers le fichier .obj / .glb / .ply
        scene     : dict de la scène existante (pour adapter scale/position)
        prompt    : prompt original (pour déduire la position si position=None)
        position  : {"x": float, "y": float, "z": float} depuis Null C4D
                    ou None → calculée automatiquement depuis le prompt
        log_fn    : callback logs

    Returns:
        dict avec xyz, colors_sh, opacity, scales, rotations, n
    """
    def log(msg):
        if log_fn: log_fn(msg)
        else: print(msg)

    import trimesh

    # ------------------------------------------------------------------
    # Chargement mesh
    # ------------------------------------------------------------------
    log(f"[Mesh2GS] Chargement : {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # Scène multi-mesh → fusionner en un seul mesh
        meshes = [g for g in mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Aucun mesh trouvé dans la scène")
        mesh = trimesh.util.concatenate(meshes)

    log(f"[Mesh2GS] Mesh brut : {len(mesh.vertices):,} vertices, "
        f"{len(mesh.faces):,} faces")

    # ------------------------------------------------------------------
    # Filtrage des composantes connexes
    # Shap-E produit parfois des fragments détachés (surtout sur image_to_mesh)
    # On garde uniquement le plus grand composant = l'objet principal
    # ------------------------------------------------------------------
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Trier par nombre de vertices décroissant
            components_sorted = sorted(
                components, key=lambda m: len(m.vertices), reverse=True
            )
            main_mesh   = components_sorted[0]
            n_discarded = sum(len(m.vertices) for m in components_sorted[1:])
            pct_kept    = 100 * len(main_mesh.vertices) / len(mesh.vertices)

            log(f"[Mesh2GS] {len(components)} composantes détectées")
            log(f"[Mesh2GS] Composante principale : {len(main_mesh.vertices):,} vertices "
                f"({pct_kept:.0f}% du mesh total)")
            log(f"[Mesh2GS] Fragments supprimés : {n_discarded:,} vertices "
                f"({len(components)-1} fragments)")

            # Seuil de sécurité : si la composante principale fait < 30%
            # du mesh total, c'est suspect — on garde tout
            if pct_kept < 30:
                log("[Mesh2GS] Composante principale trop petite — mesh conservé entier")
            else:
                mesh = main_mesh
        else:
            log("[Mesh2GS] Mesh déjà connexe — aucun filtrage nécessaire")
    except Exception as e:
        log(f"[Mesh2GS] Filtrage composantes échoué ({e}) — mesh conservé tel quel")

    # ------------------------------------------------------------------
    # Calcul adaptatif du nombre de gaussiens
    # Basé sur la surface réelle du mesh
    # ------------------------------------------------------------------
    try:
        surface = float(mesh.area)
        if scene is not None and "xyz" in scene:
            scene_size_norm = float(np.linalg.norm(
                scene["xyz"].max(0) - scene["xyz"].min(0)
            ))
            densite = 30000 / max(scene_size_norm, 0.1)
        else:
            densite = 5000

        n_points = int(surface * densite)
        n_points = max(5000, min(n_points, 200000))
        log(f"[Mesh2GS] n_points adaptatif : {n_points:,} "
            f"(surface={surface:.3f}, densité={densite:.0f})")
    except Exception as e:
        n_points = 30000
        log(f"[Mesh2GS] Calcul adaptatif échoué ({e}) → {n_points:,} par défaut")

    # ------------------------------------------------------------------
    # Échantillonnage surface
    # ------------------------------------------------------------------
    log(f"[Mesh2GS] Mesh final : {len(mesh.vertices):,} vertices, "
        f"{len(mesh.faces):,} faces")
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)
    log(f"[Mesh2GS] {len(points):,} points échantillonnés")

    # ------------------------------------------------------------------
    # Couleurs — interpolées depuis vertex colors ou texture
    # ------------------------------------------------------------------
    if image_path and os.path.isfile(image_path):
        colors_rgb = _get_colors_from_image(points, image_path, log)
    else:
        colors_rgb = _get_colors(mesh, points, face_idx, log)

    # ------------------------------------------------------------------
    # Adapter à la scène existante
    # ------------------------------------------------------------------
    if scene is not None and "xyz" in scene:
        xyz_ref = scene["xyz"]

        # 1. Normaliser le mesh dans [-1, 1] puis adapter à la scène
        pts_min = points.min(axis=0)
        pts_max = points.max(axis=0)
        pts_range = pts_max - pts_min
        pts_range[pts_range < 1e-8] = 1.0

        # Taille cible = 25% de la scène
        scene_size  = xyz_ref.max(axis=0) - xyz_ref.min(axis=0)
        target_size = scene_size * 0.25
        scale_factor = (target_size / pts_range).min()

        # Centrer le mesh
        pts_center = (pts_min + pts_max) / 2
        points = (points - pts_center) * scale_factor

        # 2. Positionner selon position explicite ou prompt
        if position is not None:
            pos_c4d = np.array([
                float(position.get("x", 0)),
                float(position.get("y", 0)),
                float(position.get("z", 0)),
            ], dtype=np.float32)

            gs_min    = xyz_ref.min(axis=0)
            gs_max    = xyz_ref.max(axis=0)
            gs_size   = gs_max - gs_min
            gs_center = (gs_min + gs_max) / 2

            # gsplat_viewer.py applique : _scale = 200.0 / scene_size
            # scene_size = max de la bbox GS
            # → pour convertir C4D → GS : diviser par ce même facteur
            scene_size_gs = float(gs_size.max())
            c4d_scale     = 200.0 / scene_size_gs if scene_size_gs > 0 else 1.0
            factor        = 1.0 / c4d_scale  # = scene_size_gs / 200.0

            obj_center = pos_c4d * factor
            obj_center = obj_center.astype(np.float32)

            log(f"[Mesh2GS] C4D scale={c4d_scale:.2f} | GS factor={factor:.5f}")
            log(f"[Mesh2GS] C4D{pos_c4d.round(2)} → GS{obj_center.round(3)}")

            # Sécurité : rester dans la scène + marge 60%
            margin     = gs_size * 0.6
            obj_center = np.clip(obj_center, gs_min - margin, gs_max + margin)
            log(f"[Mesh2GS] Position finale : {obj_center.round(3)}")

        else:
            # Position automatique depuis le prompt
            obj_center = _compute_position(xyz_ref, scene_size, prompt, log)
        points += obj_center

        log(f"[Mesh2GS] Scale factor : {scale_factor:.4f}")
        log(f"[Mesh2GS] Position     : {obj_center.round(3)}")

        # 3. Scale des gaussiens = médiane de la scène
        per_gs = scene["scales"].mean(axis=1) if "scales" in scene else None
        if per_gs is not None:
            scale_val = float(np.percentile(per_gs, 75))
            scale_val = max(min(scale_val, 0.02), 0.001)
        else:
            scale_val = 0.005

    else:
        # Pas de scène de référence → valeurs par défaut
        scale_val = 0.005
        log("[Mesh2GS] Pas de scène de référence — scale par défaut")

    log(f"[Mesh2GS] Scale gaussiens : {scale_val:.5f}")

    # ------------------------------------------------------------------
    # Assemblage dict gaussiens
    # ------------------------------------------------------------------
    n = len(points)
    colors_sh = (colors_rgb - 0.5) / 0.282095

    gaussians = {
        "xyz":       points,
        "colors_sh": colors_sh.astype(np.float32),
        "opacity":   np.full(n, 0.9, dtype=np.float32),
        "scales":    np.full((n, 3), scale_val, dtype=np.float32),
        "rotations": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "n":         n,
    }

    log(f"[Mesh2GS] {n:,} gaussiens créés | "
        f"couleur moy : R={colors_rgb[:,0].mean():.2f} "
        f"G={colors_rgb[:,1].mean():.2f} "
        f"B={colors_rgb[:,2].mean():.2f}")

    return gaussians


def _get_colors_from_image(points, image_path, log) -> np.ndarray:
    """Projette l image originale sur les points 3D (vue frontale)."""
    try:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGBA"))
        x = points[:, 0]
        y = points[:, 1]
        x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-8)
        y_norm = 1.0 - (y - y.min()) / max(y.max() - y.min(), 1e-8)
        h, w = img.shape[:2]
        px = np.clip((x_norm * w).astype(int), 0, w-1)
        py = np.clip((y_norm * h).astype(int), 0, h-1)
        colors = img[py, px, :3] / 255.0
        alpha  = img[py, px, 3]
        colors[alpha < 10] = 0.5
        log(f"[Mesh2GS] Couleurs depuis image originale")
        return colors.astype(np.float32)
    except Exception as e:
        log(f"[Mesh2GS] Projection image echouee ({e})")
        return np.full((len(points), 3), 0.7, dtype=np.float32)

def _get_colors(mesh, points, face_idx, log) -> np.ndarray:
    """
    Extrait les couleurs RGB [0,1] pour chaque point échantillonné.
    Essaie dans l'ordre :
      1. Vertex colors
      2. Texture (UV mapping)
      3. Couleur uniforme grise
    """
    import trimesh

    n = len(points)

    # Vertex colors
    try:
        if hasattr(mesh.visual, "vertex_colors") and \
           mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors
            if vc.shape[0] == len(mesh.vertices) and vc.shape[1] >= 3:
                # Interpoler depuis les vertex colors des faces
                faces = mesh.faces[face_idx]
                # Moyenne des 3 vertices de chaque face
                c = vc[faces].mean(axis=1)[:, :3] / 255.0
                log(f"[Mesh2GS] Couleurs depuis vertex colors")
                return np.clip(c, 0, 1).astype(np.float32)
    except Exception as e:
        log(f"[Mesh2GS] Vertex colors échoué ({e})")

    # Texture via UV
    try:
        if hasattr(mesh.visual, "to_color"):
            color_mesh = mesh.visual.to_color()
            if color_mesh is not None and hasattr(color_mesh, "vertex_colors"):
                vc = color_mesh.vertex_colors
                if vc is not None and vc.shape[0] == len(mesh.vertices):
                    faces = mesh.faces[face_idx]
                    c = vc[faces].mean(axis=1)[:, :3] / 255.0
                    log("[Mesh2GS] Couleurs depuis texture UV")
                    return np.clip(c, 0, 1).astype(np.float32)
    except Exception as e:
        log(f"[Mesh2GS] Texture UV échoué ({e})")

    # Fallback : couleur grise uniforme
    log("[Mesh2GS] Couleur uniforme grise (pas de texture)")
    return np.full((n, 3), 0.7, dtype=np.float32)


def _compute_position(xyz_ref: np.ndarray, scene_size: np.ndarray,
                      prompt: str, log) -> np.ndarray:
    """
    Calcule la position de l'objet dans la scène selon le prompt.
    Mots-clés supportés : left/gauche, right/droite, front/avant,
                          behind/arrière, on/sur/top/above
    """
    scene_center = xyz_ref.mean(axis=0)
    p = prompt.lower()

    if any(w in p for w in ["left", "gauche"]):
        pos = scene_center + np.array([-scene_size[0] * 0.55, 0, 0])
        log("[Mesh2GS] Position : gauche")
    elif any(w in p for w in ["right", "droite"]):
        pos = scene_center + np.array([scene_size[0] * 0.55, 0, 0])
        log("[Mesh2GS] Position : droite")
    elif any(w in p for w in ["front", "avant", "devant"]):
        pos = scene_center + np.array([0, 0, scene_size[2] * 0.55])
        log("[Mesh2GS] Position : devant")
    elif any(w in p for w in ["behind", "back", "arriere", "derriere"]):
        pos = scene_center + np.array([0, 0, -scene_size[2] * 0.55])
        log("[Mesh2GS] Position : derrière")
    elif any(w in p for w in ["on", "sur", "top", "above", "dessus"]):
        pos = scene_center + np.array([0, scene_size[1] * 0.4, 0])
        log("[Mesh2GS] Position : dessus")
    else:
        # Défaut : à droite de la scène
        pos = scene_center + np.array([scene_size[0] * 0.55, 0, 0])
        log("[Mesh2GS] Position : droite (défaut)")

    return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\MSI\Documents\C4D\ui_result\a_red_pillow.obj"

    print(f"=== Test mesh_to_gaussians : {mesh_path} ===")
    g = mesh_to_gaussians(mesh_path, n_points=1000)
    print(f"N gaussiens : {g['n']}")
    print(f"XYZ range   : {g['xyz'].min(0).round(3)} → {g['xyz'].max(0).round(3)}")
    print(f"Scale       : {g['scales'].mean():.5f}")
    print(f"Couleur moy : {(0.5 + 0.282095*g['colors_sh']).clip(0,1).mean(0).round(2)}")
    print("=== OK ===")