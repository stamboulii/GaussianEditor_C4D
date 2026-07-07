"""
core/ply_io.py
--------------
Lecture et écriture de fichiers .ply au format Gaussian Splatting (3DGS).

Le format .ply 3DGS est binaire little-endian. Chaque vertex contient :
  - x, y, z           : position (float32)
  - nx, ny, nz         : normales (float32, souvent = 0)
  - f_dc_0/1/2        : couleur SH degré 0 (float32)
  - f_rest_0..44      : coefficients SH degrés 1–3 (float32, optionnels)
  - opacity           : avant sigmoid (float32)
  - scale_0/1/2       : avant exp (float32)
  - rot_0/1/2/3       : quaternion (float32)

Compatible C4D 2025 (CPython 3.11) — aucune dépendance externe requise.
numpy est utilisé s'il est disponible pour la performance ; sinon fallback struct.
"""

import struct
import os
from typing import Dict, List, Tuple, Optional

# numpy est optionnel (pas embarqué dans C4D) — on le tente via le Python externe
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class GaussianData:
    """
    Conteneur pour tous les attributs d'une scène Gaussian Splatting.
    Tous les champs sont des listes Python (ou np.ndarray si numpy disponible).
    """
    def __init__(self):
        self.n            = 0          # nombre de gaussiens
        self.xyz          = []         # [(x,y,z), ...]  float
        self.colors_sh    = []         # [(r,g,b), ...]  coefficients DC (avant correction)
        self.colors_rgb   = []         # [(r,g,b), ...]  couleurs 0-1 après correction SH
        self.opacity      = []         # [float, ...]    après sigmoid
        self.scales       = []         # [(sx,sy,sz), ...] après exp
        self.rotations    = []         # [(w,x,y,z), ...] quaternion normalisé
        self.extra_props  = {}         # propriétés supplémentaires non mappées
        self.source_path  = ""


# ---------------------------------------------------------------------------
# Parsing du header PLY
# ---------------------------------------------------------------------------

def _parse_header(f) -> Tuple[int, List[str], str]:
    """
    Lit le header ASCII d'un fichier PLY.
    Retourne : (n_vertices, [noms des propriétés], format)
    """
    properties = []
    n_vertices = 0
    fmt = "binary_little_endian"

    while True:
        raw = f.readline()
        # Gérer les fins de ligne Windows/Unix et l'encodage
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            line = raw.decode("latin-1").strip()

        if line == "end_header":
            break
        elif line.startswith("format"):
            parts = line.split()
            fmt = parts[1] if len(parts) > 1 else fmt
        elif line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property float"):
            properties.append(line.split()[-1])
        elif line.startswith("property double"):
            # Certains exports utilisent double
            properties.append(line.split()[-1])

    return n_vertices, properties, fmt


# ---------------------------------------------------------------------------
# Lecture rapide avec numpy (chemin préféré)
# ---------------------------------------------------------------------------

def _load_numpy(f, n_vertices: int, properties: List[str]) -> Dict:
    """Lecture vectorisée avec numpy — O(n) en une seule lecture mémoire."""
    n_props = len(properties)
    raw = f.read(n_vertices * n_props * 4)  # 4 bytes par float32
    data = np.frombuffer(raw, dtype="<f4").reshape(n_vertices, n_props)

    idx = {name: i for i, name in enumerate(properties)}

    def col(name):
        return data[:, idx[name]] if name in idx else None

    def cols(*names):
        found = [data[:, idx[n]] for n in names if n in idx]
        if not found:
            return np.zeros((n_vertices, len(names)), dtype=np.float32)
        return np.stack(found, axis=1)

    # Correction SH : rgb = 0.5 + 0.282095 * f_dc
    sh_dc = cols("f_dc_0", "f_dc_1", "f_dc_2")
    colors_rgb = np.clip(0.5 + 0.282095 * sh_dc, 0.0, 1.0)

    # opacity : sigmoid(raw)
    raw_opacity = col("opacity")
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity)) if raw_opacity is not None else np.ones(n_vertices)

    # scales : exp(raw)
    raw_scales = cols("scale_0", "scale_1", "scale_2")
    scales = np.exp(raw_scales) if raw_scales.shape[1] == 3 else np.ones((n_vertices, 3))

    # rotations : quaternion (w,x,y,z) — normalisation pour la robustesse
    raw_rots = cols("rot_0", "rot_1", "rot_2", "rot_3")
    if raw_rots.shape[1] == 4:
        norms = np.linalg.norm(raw_rots, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        rotations = raw_rots / norms
    else:
        rotations = np.tile([1, 0, 0, 0], (n_vertices, 1)).astype(np.float32)

    return {
        "xyz":        cols("x", "y", "z"),
        "colors_sh":  sh_dc,
        "colors_rgb": colors_rgb,
        "opacity":    opacity,
        "scales":     scales,
        "rotations":  rotations,
        "raw":        data,
        "properties": properties,
    }


# ---------------------------------------------------------------------------
# Lecture fallback sans numpy (pure Python, plus lente)
# ---------------------------------------------------------------------------

def _load_pure_python(f, n_vertices: int, properties: List[str]) -> Dict:
    """Fallback pur Python — utilisé si numpy absent (cas C4D natif)."""
    import math

    n_props = len(properties)
    fmt_str = f"<{n_props}f"
    size = struct.calcsize(fmt_str)

    idx = {name: i for i, name in enumerate(properties)}

    xyz, colors_sh, colors_rgb, opacity_list, scales, rotations = [], [], [], [], [], []

    for _ in range(n_vertices):
        row = struct.unpack(fmt_str, f.read(size))

        x = row[idx.get("x", 0)]
        y = row[idx.get("y", 1)]
        z = row[idx.get("z", 2)]
        xyz.append((x, y, z))

        r = row[idx["f_dc_0"]] if "f_dc_0" in idx else 0.0
        g = row[idx["f_dc_1"]] if "f_dc_1" in idx else 0.0
        b = row[idx["f_dc_2"]] if "f_dc_2" in idx else 0.0
        colors_sh.append((r, g, b))

        # Correction SH
        cr = max(0.0, min(1.0, 0.5 + 0.282095 * r))
        cg = max(0.0, min(1.0, 0.5 + 0.282095 * g))
        cb = max(0.0, min(1.0, 0.5 + 0.282095 * b))
        colors_rgb.append((cr, cg, cb))

        # Opacity sigmoid
        raw_o = row[idx["opacity"]] if "opacity" in idx else 0.0
        op_val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw_o))))
        opacity_list.append(op_val)

        # Scales exp
        s0 = math.exp(row[idx["scale_0"]]) if "scale_0" in idx else 1.0
        s1 = math.exp(row[idx["scale_1"]]) if "scale_1" in idx else 1.0
        s2 = math.exp(row[idx["scale_2"]]) if "scale_2" in idx else 1.0
        scales.append((s0, s1, s2))

        # Rotations (quaternion)
        w = row[idx["rot_0"]] if "rot_0" in idx else 1.0
        rx = row[idx["rot_1"]] if "rot_1" in idx else 0.0
        ry = row[idx["rot_2"]] if "rot_2" in idx else 0.0
        rz = row[idx["rot_3"]] if "rot_3" in idx else 0.0
        norm = math.sqrt(w*w + rx*rx + ry*ry + rz*rz)
        if norm < 1e-8:
            norm = 1.0
        rotations.append((w/norm, rx/norm, ry/norm, rz/norm))

    return {
        "xyz":        xyz,
        "colors_sh":  colors_sh,
        "colors_rgb": colors_rgb,
        "opacity":    opacity_list,
        "scales":     scales,
        "rotations":  rotations,
        "properties": properties,
    }


# ---------------------------------------------------------------------------
# API publique : parse_gaussian_ply()
# ---------------------------------------------------------------------------

def parse_gaussian_ply(path: str) -> GaussianData:
    """
    Parse un fichier .ply au format Gaussian Splatting.

    Args:
        path: Chemin absolu vers le fichier .ply

    Returns:
        GaussianData avec tous les attributs remplis

    Raises:
        FileNotFoundError: si le fichier n'existe pas
        ValueError: si le format PLY n'est pas reconnu
        RuntimeError: si le fichier est corrompu
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    file_size = os.path.getsize(path)
    if file_size < 100:
        raise ValueError(f"Fichier trop petit pour être un PLY valide ({file_size} bytes)")

    gd = GaussianData()
    gd.source_path = path

    with open(path, "rb") as f:
        # Vérifier magic PLY
        magic = f.read(4)
        if magic != b"ply\n" and magic != b"ply\r":
            raise ValueError(f"Pas un fichier PLY valide (magic: {magic})")
        f.seek(0)

        n_vertices, properties, fmt = _parse_header(f)

        if n_vertices == 0:
            raise ValueError("Le fichier PLY ne contient aucun vertex.")

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(
                f"Format PLY non supporté : '{fmt}'. "
                "Seul binary_little_endian est supporté pour le Gaussian Splatting."
            )

        # Vérification que les propriétés essentielles sont présentes
        required = {"x", "y", "z"}
        missing = required - set(properties)
        if missing:
            raise ValueError(f"Propriétés PLY manquantes : {missing}")

        gd.n = n_vertices

        # Choisir le parseur selon la disponibilité de numpy
        if HAS_NUMPY:
            result = _load_numpy(f, n_vertices, properties)
        else:
            result = _load_pure_python(f, n_vertices, properties)

    gd.xyz        = result["xyz"]
    gd.colors_sh  = result["colors_sh"]
    gd.colors_rgb = result["colors_rgb"]
    gd.opacity    = result["opacity"]
    gd.scales     = result["scales"]
    gd.rotations  = result["rotations"]

    return gd


# ---------------------------------------------------------------------------
# API publique : write_gaussian_ply()
# ---------------------------------------------------------------------------

def write_gaussian_ply(path: str, gd: GaussianData) -> None:
    """
    Écrit une GaussianData dans un fichier .ply binaire little-endian.
    Reconstruit les attributs depuis les valeurs décodées (réencoding SH, log, logit).

    Args:
        path: Chemin de sortie (sera créé ou écrasé)
        gd:   GaussianData à sérialiser
    """
    import math

    n = gd.n
    if n == 0:
        raise ValueError("GaussianData vide, rien à écrire.")

    # Propriétés qu'on va écrire (ordre standard 3DGS)
    props = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]

    # Construction du header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment GaussianEditor C4D Plugin",
        f"element vertex {n}",
    ]
    for p in props:
        header_lines.append(f"property float {p}")
    header_lines.append("end_header")
    header_lines.append("")  # newline final

    header_bytes = "\n".join(header_lines).encode("utf-8")

    # Helpers pour ré-encoder
    def safe_logit(x):
        x = max(1e-6, min(1.0 - 1e-6, x))
        return math.log(x / (1.0 - x))

    def safe_log(x):
        return math.log(max(1e-8, x))

    def get_xyz(i):
        v = gd.xyz[i]
        return (v[0], v[1], v[2]) if hasattr(v, "__len__") else (float(v), 0.0, 0.0)

    def get_sh(i):
        # Si colors_sh disponibles, les utiliser directement
        sh_available = (gd.colors_sh is not None and
                        hasattr(gd.colors_sh, '__len__') and
                        len(gd.colors_sh) > 0)
        if sh_available:
            c = gd.colors_sh[i]
            return (float(c[0]), float(c[1]), float(c[2]))
        # Sinon inverser la correction SH depuis colors_rgb
        rgb = gd.colors_rgb[i]
        return ((float(rgb[0]) - 0.5) / 0.282095,
                (float(rgb[1]) - 0.5) / 0.282095,
                (float(rgb[2]) - 0.5) / 0.282095)

    def get_opacity(i):
        return safe_logit(float(gd.opacity[i]))

    def get_scale(i):
        s = gd.scales[i]
        return (safe_log(s[0]), safe_log(s[1]), safe_log(s[2]))

    def get_rot(i):
        r = gd.rotations[i]
        return (r[0], r[1], r[2], r[3])

    fmt_str = "<17f"  # 17 propriétés float32

    with open(path, "wb") as f:
        f.write(header_bytes)
        for i in range(n):
            xyz  = get_xyz(i)
            sh   = get_sh(i)
            op   = get_opacity(i)
            sc   = get_scale(i)
            rot  = get_rot(i)
            row = (
                xyz[0], xyz[1], xyz[2],
                0.0, 0.0, 0.0,          # normales = 0
                sh[0], sh[1], sh[2],
                op,
                sc[0], sc[1], sc[2],
                rot[0], rot[1], rot[2], rot[3],
            )
            f.write(struct.pack(fmt_str, *row))


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def ply_info(path: str) -> dict:
    """
    Retourne des métadonnées sur un fichier PLY sans charger les données.
    Utile pour afficher des infos dans l'UI sans overhead.
    """
    if not os.path.isfile(path):
        return {"error": f"Fichier introuvable : {path}"}

    try:
        with open(path, "rb") as f:
            n_vertices, properties, fmt = _parse_header(f)
        return {
            "n_vertices":  n_vertices,
            "properties":  properties,
            "format":      fmt,
            "file_size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
            "has_colors":  "f_dc_0" in properties,
            "has_opacity": "opacity" in properties,
            "has_scales":  "scale_0" in properties,
            "has_rotations": "rot_0" in properties,
        }
    except Exception as e:
        return {"error": str(e)}