#!/bin/bash
# =============================================================
# install.sh — GaussianEditor C4D
# Installation complete depuis WSL2 Ubuntu
#
# Usage :
#   git clone https://github.com/TON_USER/GaussianEditor_C4D
#   cd GaussianEditor_C4D
#   chmod +x install.sh
#   ./install.sh
# =============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "${GREEN}OK  $1${NC}"; }
warn() { echo -e "${YELLOW}WARN $1${NC}"; }
err()  { echo -e "${RED}ERR  $1${NC}"; exit 1; }
info() { echo -e "     $1"; }
step() { echo -e "\n${BLUE}-- $1${NC}"; }

echo ""
echo "=================================================="
echo "   GaussianEditor C4D -- Installation complete"
echo "=================================================="
echo ""

# Verifier WSL2
if ! grep -qi "microsoft" /proc/version 2>/dev/null; then
    warn "Ce script est concu pour WSL2 Ubuntu."
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "Repo : $REPO_DIR"

# Detecter utilisateur Windows
WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r' || \
           ls /mnt/c/Users/ | grep -v "Public\|Default\|All Users" | head -1)
info "Utilisateur Windows : $WIN_USER"

# =============================================================
# Detection Cinema 4D
# =============================================================

_detect_c4d_plugins() {
    local base="/mnt/c/Users/$WIN_USER/AppData/Roaming/Maxon"
    [ ! -d "$base" ] && echo "" && return

    mapfile -t DIRS < <(find "$base" -maxdepth 2 -name "plugins" -type d 2>/dev/null)

    [ ${#DIRS[@]} -eq 0 ] && echo "" && return
    [ ${#DIRS[@]} -eq 1 ] && echo "${DIRS[0]}" && return

    # Plusieurs versions : prendre la plus recente
    local newest="" newest_date=0
    for dir in "${DIRS[@]}"; do
        local date
        date=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
        if [ "$date" -gt "$newest_date" ]; then
            newest_date=$date
            newest=$dir
        fi
    done

    warn "Plusieurs versions C4D detectees :"
    for dir in "${DIRS[@]}"; do
        local name
        name=$(basename "$(dirname "$dir")")
        if [ "$dir" = "$newest" ]; then
            info "  -> $name  <- selectionnee (plus recente)"
        else
            info "     $name"
        fi
    done

    echo "$newest"
}

C4D_PLUGIN_DIR=$(_detect_c4d_plugins)

if [ -n "$C4D_PLUGIN_DIR" ]; then
    C4D_VERSION=$(basename "$(dirname "$C4D_PLUGIN_DIR")")
    ok "Cinema 4D : $C4D_VERSION"
    info "Plugins : $C4D_PLUGIN_DIR"
else
    warn "Cinema 4D non detecte automatiquement."
    echo ""
    echo "     Entrez le chemin du dossier plugins C4D"
    echo "     Exemple : /mnt/c/Users/$WIN_USER/AppData/Roaming/Maxon/Maxon Cinema 4D 2025_XXX/plugins"
    read -p "     Chemin (ou Entree pour ignorer) : " C4D_PLUGIN_DIR
fi

echo ""

# =============================================================
# Etape 1 -- GPU / CUDA
# =============================================================
step "Etape 1 : Verification GPU"

command -v nvidia-smi &>/dev/null || err "nvidia-smi introuvable. Installez les drivers NVIDIA cote Windows."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
VRAM_GB=$((VRAM_MB / 1024))

ok "GPU : $GPU_NAME (${VRAM_GB} GB VRAM)"
[ "$VRAM_GB" -lt 4 ] && err "VRAM insuffisante (${VRAM_GB} GB). Minimum requis : 4 GB."

# =============================================================
# Etape 2 -- Miniconda
# =============================================================
step "Etape 2 : Miniconda"

if command -v conda &>/dev/null; then
    ok "Conda deja installe : $(conda --version)"
else
    info "Telechargement Miniconda..."
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    ok "Miniconda installe"
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true

# =============================================================
# Etape 3 -- Environnement conda gs_c4d
# =============================================================
step "Etape 3 : Environnement conda gs_c4d"

if conda env list | grep -q "^gs_c4d "; then
    ok "Environnement gs_c4d existe deja"
else
    info "Creation de l'environnement gs_c4d (Python 3.11)..."
    conda create -n gs_c4d python=3.11 -y
    ok "Environnement cree"
fi

conda activate gs_c4d

# =============================================================
# Etape 4 -- PyTorch CUDA
# =============================================================
step "Etape 4 : PyTorch CUDA"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "PyTorch CUDA deja installe : $(python -c 'import torch; print(torch.__version__)')"
else
    info "Installation PyTorch avec support CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    python -c "import torch; assert torch.cuda.is_available()" || \
        err "PyTorch CUDA non fonctionnel. Verifiez vos drivers NVIDIA."
    ok "PyTorch CUDA installe"
fi

# =============================================================
# Etape 5 -- gsplat
# =============================================================
step "Etape 5 : gsplat"

if python -c "import gsplat" 2>/dev/null; then
    ok "gsplat deja installe"
else
    info "Lancement install_gsplat.sh..."
    chmod +x "$REPO_DIR/install_gsplat.sh"
    bash "$REPO_DIR/install_gsplat.sh"
    python -c "import gsplat" 2>/dev/null && ok "gsplat installe" || err "gsplat installation echouee"
fi

# =============================================================
# Etape 6 -- Dependances Python
# =============================================================
step "Etape 6 : Dependances Python"

pip install -r "$REPO_DIR/requirements.txt" -q
ok "Dependances installees"

# =============================================================
# Etape 7 -- TripoSplat
# =============================================================
step "Etape 7 : TripoSplat (clone + poids ~3.8 GB)"

TRIPOSPLAT_DIR="$HOME/TripoSplat"

if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    info "Clonage TripoSplat..."
    git clone https://github.com/VAST-AI-Research/TripoSplat.git "$TRIPOSPLAT_DIR"
    ok "TripoSplat clone"
else
    ok "TripoSplat deja clone"
fi

mkdir -p "$TRIPOSPLAT_DIR/ckpts/diffusion_models"
mkdir -p "$TRIPOSPLAT_DIR/ckpts/vae"
mkdir -p "$TRIPOSPLAT_DIR/ckpts/clip_vision"
mkdir -p "$TRIPOSPLAT_DIR/ckpts/background_removal"

HF_BASE="https://huggingface.co/VAST-AI/TripoSplat/resolve/main"

declare -A WEIGHTS=(
    ["ckpts/diffusion_models/triposplat_fp16.safetensors"]="diffusion_models/triposplat_fp16.safetensors"
    ["ckpts/vae/triposplat_vae_decoder_fp16.safetensors"]="vae/triposplat_vae_decoder_fp16.safetensors"
    ["ckpts/vae/flux2-vae.safetensors"]="vae/flux2-vae.safetensors"
    ["ckpts/background_removal/birefnet.safetensors"]="background_removal/birefnet.safetensors"
    ["ckpts/clip_vision/dino_v3_vit_h.safetensors"]="clip_vision/dino_v3_vit_h.safetensors"
)

for LOCAL_PATH in "${!WEIGHTS[@]}"; do
    FULL_PATH="$TRIPOSPLAT_DIR/$LOCAL_PATH"
    if [ -f "$FULL_PATH" ] && [ -s "$FULL_PATH" ]; then
        ok "$(basename $FULL_PATH) deja present"
    else
        info "Telechargement $(basename $FULL_PATH)..."
        wget -c "$HF_BASE/${WEIGHTS[$LOCAL_PATH]}" \
             -O "$FULL_PATH" --progress=bar:force 2>&1
        ok "$(basename $FULL_PATH) telecharge"
    fi
done

# =============================================================
# Etape 8 -- Installer le plugin dans C4D
# =============================================================
step "Etape 8 : Installation plugin C4D"

if [ -n "$C4D_PLUGIN_DIR" ]; then
    DEST="$C4D_PLUGIN_DIR/GaussianEditor_C4D"
    mkdir -p "$DEST"
    rsync -a --exclude='.git' --exclude='__pycache__' \
          --exclude='*.pyc' --exclude='.env' --exclude='*.bak' \
          "$REPO_DIR/" "$DEST/"
    ok "Plugin installe : $DEST"
else
    warn "Plugin non installe -- copiez manuellement dans C4D/plugins/"
fi

# =============================================================
# Etape 9 -- Raccourci Bureau Windows
# =============================================================
step "Etape 8b : Sauvegarde chemin Python"

PYTHON_PATH=$(conda run -n gs_c4d which python 2>/dev/null || echo "")
if [ -n "$PYTHON_PATH" ]; then
    # Convertir chemin WSL -> Windows
    WIN_PYTHON=$(echo "$PYTHON_PATH" | sed 's|/mnt/\([a-z]\)|\U\1:|' | sed 's|/|\\\\|g')
    # Sauvegarder dans le plugin pour que C4D le lise automatiquement
    if [ -n "$C4D_PLUGIN_DIR" ]; then
        echo "$WIN_PYTHON" > "$C4D_PLUGIN_DIR/GaussianEditor_C4D/python_path.txt"
        ok "Chemin Python sauvegarde : $WIN_PYTHON"
    fi
    echo "$WIN_PYTHON" > "$REPO_DIR/python_path.txt"
fi

step "Etape 9 : Raccourci Bureau Windows"

DESKTOP="/mnt/c/Users/$WIN_USER/Desktop"
PLUGIN_WSL="${C4D_PLUGIN_DIR:+$C4D_PLUGIN_DIR/GaussianEditor_C4D}"
PLUGIN_WSL="${PLUGIN_WSL:-$REPO_DIR}"
PYTHON_PATH=$(conda run -n gs_c4d which python 2>/dev/null || echo "python")

if [ -d "$DESKTOP" ]; then
    cat > "$DESKTOP/Demarrer_GaussianEditor.bat" << BATEOF
@echo off
title GaussianEditor C4D - Serveur TripoSplat
echo.
echo  ================================================
echo  GaussianEditor C4D  -  Serveur actif
echo  Ne pas fermer cette fenetre !
echo  ================================================
echo.
wsl.exe -e bash -ic "conda activate gs_c4d && cd '$PLUGIN_WSL' && python -m core.server"
pause
BATEOF
    ok "Raccourci cree : Bureau\\Demarrer_GaussianEditor.bat"
else
    warn "Bureau Windows non trouve -- creez le raccourci manuellement"
fi

# =============================================================
# Etape 10 -- Verification finale
# =============================================================
step "Etape 10 : Verification"

ERRORS=0

python -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from core.gs_config import detect_device, recommended_num_gaussians
device, vram = detect_device()
print(f'  Device : {device} | VRAM : {vram:.1f} GB')
print(f'  Gaussians recommandes : {recommended_num_gaussians(vram)}')
" && ok "gs_config.py OK" || { warn "gs_config.py KO"; ERRORS=$((ERRORS+1)); }

python -c "
import sys
sys.path.insert(0, '$HOME/TripoSplat')
from triposplat import TripoSplatPipeline
" && ok "TripoSplat importable" || { warn "TripoSplat KO"; ERRORS=$((ERRORS+1)); }

python -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from core.server import main
" && ok "server.py importable" || { warn "server.py KO"; ERRORS=$((ERRORS+1)); }

MISSING_WEIGHTS=0
for LOCAL_PATH in "${!WEIGHTS[@]}"; do
    [ ! -s "$TRIPOSPLAT_DIR/$LOCAL_PATH" ] && MISSING_WEIGHTS=$((MISSING_WEIGHTS+1))
done
[ $MISSING_WEIGHTS -eq 0 ] && ok "5/5 poids TripoSplat presents" || \
    { warn "$MISSING_WEIGHTS poids manquants -- relancez install.sh"; ERRORS=$((ERRORS+1)); }

# =============================================================
# Resume final
# =============================================================
echo ""
echo "=================================================="
if [ $ERRORS -eq 0 ]; then
    echo "   Installation reussie !"
else
    echo "   Installation avec $ERRORS erreur(s)"
fi
echo "=================================================="
echo ""
echo "  Pour utiliser GaussianEditor C4D :"
echo ""
echo "  1. Double-cliquez sur le Bureau Windows :"
echo "     Demarrer_GaussianEditor.bat"
echo ""
echo "  2. Ouvrez Cinema 4D"
echo ""
echo "  3. Dans le plugin, entrez le chemin Python :"
WIN_PYTHON=$(echo "$PYTHON_PATH" | sed 's|/mnt/\([a-z]\)|\U\1:|' | sed 's|/|\\\\|g')
echo "     $WIN_PYTHON"
echo ""
echo "  4. Cliquez Demarrer -> selectionnez une image -> Generer"
echo ""
echo "  Documentation : INSTALLATION.md"
echo ""