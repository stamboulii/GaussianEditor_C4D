# GaussianEditor C4D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Cinema 4D](https://img.shields.io/badge/Cinema%204D-2025-red.svg)](https://maxon.net)
[![WSL2](https://img.shields.io/badge/WSL2-Ubuntu-orange.svg)](https://docs.microsoft.com/en-us/windows/wsl/)

Plugin Cinema 4D pour générer, composer et éditer des **Gaussian Splats 3D** directement depuis une image, en utilisant [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat).

---

## Démonstration

| Image source | Résultat 3D |
|---|---|
| Photo canapé | Gaussian Splat 3D en ~20 min |
| Photo guitare | Ajout à une scène existante |

---

## Fonctionnalités

- **Générer** un objet 3D depuis une image (JPG, PNG, WEBP)
- **Ajouter** des objets à une scène existante (placement automatique)
- **Segmenter** des gaussiens par texte (LangSAM)
- **Supprimer / Crop** une sélection
- **Undo** — annuler la dernière action
- **Sauver** — fusionner plusieurs objets en un seul PLY avec positions C4D
- **Export .splat** — format léger compatible viewers web

---

## Prérequis

| Composant | Version |
|---|---|
| Windows | 10 / 11 |
| WSL2 | Ubuntu 22.04 / 24.04 |
| Cinema 4D | 2025+ |
| GPU NVIDIA | RTX, 4GB VRAM minimum |
| CUDA | 11.8+ |

---

## Installation

```bash
# 1. Cloner le repo dans WSL2
git clone https://github.com/stamboulii/GaussianEditor_C4D
cd GaussianEditor_C4D

# 2. Lancer l'installation automatique
chmod +x install.sh
./install.sh
```

Le script installe automatiquement :
- Miniconda + environnement `gs_c4d`
- PyTorch CUDA
- gsplat (avec MAX_JOBS adapté à ta machine)
- Dépendances Python
- TripoSplat + poids (~3.8 GB)
- Plugin dans Cinema 4D
- Raccourci `Demarrer_GaussianEditor.bat` sur le Bureau Windows

Pour les détails, voir [INSTALLATION.md](INSTALLATION.md).

---

## Utilisation

### Démarrer le serveur

Double-clic sur **`Demarrer_GaussianEditor.bat`** sur le Bureau Windows.

Ou depuis WSL2 :
```bash
conda activate gs_c4d
cd /chemin/vers/GaussianEditor_C4D
python -m core.server
```

### Dans Cinema 4D

1. Ouvrir le plugin **GaussianEditor for C4D**
2. Entrer le chemin Python `gs_c4d` et cliquer **Démarrer**
3. Sélectionner une image → cliquer **Générer** (~20 min sur RTX 2050)
4. L'objet apparaît automatiquement dans le viewport C4D

### Workflow multi-objets

```
1. Importer ou générer objet 1 (ex: piano.ply)
2. Sélectionner image objet 2 → Générer
   → L'objet 2 est ajouté automatiquement à côté
3. Déplacer/redimensionner dans C4D (touches E et T)
4. Cliquer Sauver → PLY fusionné avec vraies couleurs
```

---

## Architecture

```
core/
├── server.py          ← Serveur HTTP (port 8086)
├── gs_config.py       ← Configuration centralisée
├── actions/           ← Endpoints HTTP (/generate, /trace, /save...)
├── models/            ← Lazy loading TripoSplat, LangSAM, IP2P
├── scene/             ← Gestion état scène + PLY I/O
├── utils/             ← Conversion SH↔RGB, rendu ortho
└── experimental/      ← Outils dormants (Kaggle, DreamGaussian...)
ui/
└── main_dialog.py     ← Interface Cinema 4D
```

---

## Endpoints API

| Endpoint | Description |
|---|---|
| `GET /health` | Statut du serveur |
| `POST /generate` | Image → TripoSplat → nouvelle scène |
| `POST /generate_and_add` | Image → TripoSplat → ajout à la scène |
| `POST /load` | Charger un PLY existant |
| `POST /trace` | Segmentation 3D par texte |
| `POST /edit` | Édition guidée par texte |
| `POST /delete` | Supprimer la sélection |
| `POST /crop` | Garder uniquement la sélection |
| `POST /undo` | Annuler la dernière action |
| `POST /save` | Sauvegarder la scène |
| `POST /merge_and_save` | Fusionner plusieurs PLY avec transformations C4D |
| `POST /export_splat` | Export format .splat |

---

## Matériel testé

| GPU | VRAM | Temps génération |
|---|---|---|
| RTX 2050 | 4 GB | ~20 min |
| RTX 3070 | 8 GB | ~8 min (estimé) |
| RTX 4090 | 24 GB | ~2 min (estimé) |

---

## Licence

MIT — voir [LICENSE](LICENSE)

---

## Crédits

- [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) — VAST-AI Research
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) — Segmentation sémantique
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Rasterisation Gaussian Splatting