"""
core/dreamgaussian_client.py
-----------------------------
Integration DreamGaussian dans GaussianEditor C4D.
"""

import os
import sys
import time
import subprocess
import shutil
import numpy as np
from typing import Optional, Callable

DREAMGAUSSIAN_DIR = os.path.expanduser("~/dreamgaussian")


class DreamGaussianClient:

    def __init__(self, log_fn: Optional[Callable] = None):
        self.log_fn = log_fn or print
        self.dg_dir = DREAMGAUSSIAN_DIR

    def _log(self, msg):
        self.log_fn(msg)

    def generate(self, image_path: str, save_name: str = "dg_object",
                 iters: int = 1000, elevation: float = 0) -> str:
        t0 = time.time()
        self._log(f"[DG] Generation depuis : {image_path}")
        self._log(f"[DG] Iters : {iters} | Elevation : {elevation}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        data_dir = os.path.join(self.dg_dir, "data")
        logs_dir = os.path.join(self.dg_dir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        img_name = os.path.splitext(os.path.basename(image_path))[0]
        dst_path = os.path.join(data_dir, f"{img_name}.png")

        if os.path.abspath(image_path) != os.path.abspath(dst_path):
            shutil.copy(image_path, dst_path)

        rgba_path = os.path.join(data_dir, f"{img_name}_rgba.png")
        if not self._has_transparent_bg(image_path):
            self._log("[DG] Suppression fond...")
            self._remove_background(dst_path, rgba_path)
        else:
            if os.path.abspath(image_path) != os.path.abspath(rgba_path):
                shutil.copy(image_path, rgba_path)
            self._log("[DG] Image deja avec fond transparent")

        self._log("[DG] Lancement training...")
        ply_path = self._run_dreamgaussian(rgba_path, save_name, iters, elevation)

        elapsed = time.time() - t0
        self._log(f"[DG] Termine en {elapsed:.0f}s -> {ply_path}")
        return ply_path

    def _has_transparent_bg(self, image_path: str) -> bool:
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGBA")
            arr = np.array(img)
            return (arr[:,:,3] < 10).sum() > 100
        except Exception:
            return False

    def _remove_background(self, src: str, dst: str, size: int = 256):
        from PIL import Image
        img = Image.open(src).convert("RGBA")
        data = np.array(img)
        r = data[:,:,0].astype(float)
        g = data[:,:,1].astype(float)
        b = data[:,:,2].astype(float)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        brightness = max_rgb / 255.0
        bg_mask = (saturation < 0.25) & (brightness > 0.55)
        data[:,:,3] = np.where(bg_mask, 0, 255)
        mask = data[:,:,3] > 0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any():
            rmin, rmax = np.where(rows)[0][[0,-1]]
            cmin, cmax = np.where(cols)[0][[0,-1]]
            data = data[rmin:rmax+1, cmin:cmax+1]
        Image.fromarray(data).resize((size, size), Image.LANCZOS).save(dst)
        self._log(f"[DG] RGBA : {dst}")

    def _run_dreamgaussian(self, rgba_path, save_name, iters, elevation):
        cmd = [
            sys.executable,
            os.path.join(self.dg_dir, "main.py"),
            "--config", os.path.join(self.dg_dir, "configs", "image.yaml"),
            f"input={rgba_path}",
            f"save_path={save_name}",
            f"iters={iters}",
            f"elevation={elevation}",
        ]
        env = os.environ.copy()
        env["CUDA_HOME"] = "/usr/local/cuda-13.2"
        env["PATH"] = f"/usr/local/cuda-13.2/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            "/usr/local/cuda-13.2/lib64:"
            "/home/mansour/miniconda3/lib/python3.13/site-packages/torch/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"

        proc = subprocess.Popen(
            cmd, cwd=self.dg_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._log(f"[DG] {line}")
        proc.wait()

        ply_path = os.path.join(self.dg_dir, "logs", f"{save_name}_model.ply")
        if not os.path.exists(ply_path):
            raise RuntimeError(f"PLY non genere : {ply_path}")
        return ply_path


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/dreamgaussian/data/anya_rgba.png")
    client = DreamGaussianClient()
    ply = client.generate(image, save_name="test_client", iters=500)
    print(f"PLY : {ply}")
