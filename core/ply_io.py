"""
core/ply_io.py
--------------
Lecture et écriture de fichiers PLY Gaussian Splatting.
Version sans numpy — utilise uniquement struct et le module standard Python.
Compatible avec le Python interne de Cinema 4D (pas de dépendances externes).
"""

import struct
import os


class GaussianData:
    """Conteneur pour les données Gaussian Splatting."""

    def __init__(self):
        self.n           = 0
        self.source_path = ""
        self.xyz         = []   # list of (x, y, z)
        self.colors_sh   = []   # list of (f_dc_0, f_dc_1, f_dc_2)
        self.colors_rgb  = []   # list of (r, g, b) in [0,1]
        self.opacity     = []   # list of float in [0,1]
        self.scales      = []   # list of (s0, s1, s2)
        self.rotations   = []   # list of (r0, r1, r2, r3)


def parse_gaussian_ply(path: str) -> GaussianData:
    """
    Lit un fichier PLY Gaussian Splatting sans numpy.
    Utilise struct pour parser les données binaires.
    """
    gd = GaussianData()
    gd.source_path = path

    with open(path, "rb") as f:
        # Lire le header
        props   = []
        n_verts = 0
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break

        gd.n = n_verts
        n_props = len(props)
        pidx    = {name: i for i, name in enumerate(props)}

        # Lire toutes les données binaires
        raw = f.read(n_verts * n_props * 4)

    # Parser avec struct
    fmt    = f"<{n_props}f"
    stride = n_props * 4

    xyz_xi = pidx.get("x", -1)
    xyz_yi = pidx.get("y", -1)
    xyz_zi = pidx.get("z", -1)

    sh0i = pidx.get("f_dc_0", -1)
    sh1i = pidx.get("f_dc_1", -1)
    sh2i = pidx.get("f_dc_2", -1)

    opi  = pidx.get("opacity", -1)

    sc0i = pidx.get("scale_0", -1)
    sc1i = pidx.get("scale_1", -1)
    sc2i = pidx.get("scale_2", -1)

    rt0i = pidx.get("rot_0", -1)
    rt1i = pidx.get("rot_1", -1)
    rt2i = pidx.get("rot_2", -1)
    rt3i = pidx.get("rot_3", -1)

    SH_C0 = 0.282095

    for i in range(n_verts):
        row = struct.unpack_from(fmt, raw, i * stride)

        # XYZ
        x = row[xyz_xi] if xyz_xi >= 0 else 0.0
        y = row[xyz_yi] if xyz_yi >= 0 else 0.0
        z = row[xyz_zi] if xyz_zi >= 0 else 0.0
        gd.xyz.append((x, y, z))

        # Couleurs SH
        sh0 = row[sh0i] if sh0i >= 0 else 0.0
        sh1 = row[sh1i] if sh1i >= 0 else 0.0
        sh2 = row[sh2i] if sh2i >= 0 else 0.0
        gd.colors_sh.append((sh0, sh1, sh2))

        # RGB depuis SH
        r = max(0.0, min(1.0, 0.5 + SH_C0 * sh0))
        g = max(0.0, min(1.0, 0.5 + SH_C0 * sh1))
        b = max(0.0, min(1.0, 0.5 + SH_C0 * sh2))
        gd.colors_rgb.append((r, g, b))

        # Opacité (logit → sigmoid)
        if opi >= 0:
            logit = row[opi]
            try:
                op = 1.0 / (1.0 + (2.718281828 ** (-logit)))
            except OverflowError:
                op = 0.0 if logit < 0 else 1.0
        else:
            op = 0.9
        gd.opacity.append(op)

        # Scales (log → exp)
        if sc0i >= 0:
            try:
                s0 = 2.718281828 ** row[sc0i]
                s1 = 2.718281828 ** row[sc1i]
                s2 = 2.718281828 ** row[sc2i]
            except OverflowError:
                s0 = s1 = s2 = 0.01
        else:
            s0 = s1 = s2 = 0.01
        gd.scales.append((s0, s1, s2))

        # Rotations
        r0 = row[rt0i] if rt0i >= 0 else 1.0
        r1 = row[rt1i] if rt1i >= 0 else 0.0
        r2 = row[rt2i] if rt2i >= 0 else 0.0
        r3 = row[rt3i] if rt3i >= 0 else 0.0
        gd.rotations.append((r0, r1, r2, r3))

    return gd


def write_gaussian_ply(path: str, gd: GaussianData) -> None:
    """
    Écrit un fichier PLY Gaussian Splatting sans numpy.
    """
    props = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]

    n = gd.n
    import math

    header  = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    SH_C0 = 0.282095

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        fmt = "<14f"
        for i in range(n):
            x, y, z   = gd.xyz[i]
            sh0, sh1, sh2 = gd.colors_sh[i] if gd.colors_sh else (0.0, 0.0, 0.0)
            op = gd.opacity[i] if gd.opacity else 0.9

            # sigmoid → logit
            op = max(1e-6, min(1 - 1e-6, op))
            logit = math.log(op / (1.0 - op))

            s0, s1, s2 = gd.scales[i] if gd.scales else (0.01, 0.01, 0.01)
            # exp → log
            ls0 = math.log(max(1e-8, s0))
            ls1 = math.log(max(1e-8, s1))
            ls2 = math.log(max(1e-8, s2))

            r0, r1, r2, r3 = gd.rotations[i] if gd.rotations else (1.0, 0.0, 0.0, 0.0)

            f.write(struct.pack(fmt,
                x, y, z,
                sh0, sh1, sh2,
                logit,
                ls0, ls1, ls2,
                r0, r1, r2, r3,
            ))