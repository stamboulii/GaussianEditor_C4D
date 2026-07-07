@echo off
title GaussianEditor C4D - Setup
chcp 65001 >nul

echo.
echo  ================================================
echo  GaussianEditor C4D -- Installation Windows
echo  ================================================
echo.

:: ── Étape 1 : Vérifier WSL2 ─────────────────────
echo  [1/4] Verification WSL2...
wsl --status >nul 2>&1
if %errorlevel% neq 0 (
    echo  WSL2 non installe. Installation en cours...
    echo  Un redemarrage sera necessaire.
    echo.
    wsl --install
    echo.
    echo  ================================================
    echo  Redemarrez votre PC puis relancez ce fichier.
    echo  ================================================
    pause
    exit /b
)
echo  [OK] WSL2 installe

:: ── Étape 2 : Vérifier Ubuntu ───────────────────
echo  [2/4] Verification Ubuntu...
wsl -d Ubuntu-24.04 echo ok >nul 2>&1
if %errorlevel% neq 0 (
    wsl -d Ubuntu echo ok >nul 2>&1
    if %errorlevel% neq 0 (
        echo  Installation Ubuntu 24.04...
        wsl --install -d Ubuntu-24.04
        echo.
        echo  Creez votre utilisateur Ubuntu puis relancez ce fichier.
        pause
        exit /b
    )
)
echo  [OK] Ubuntu disponible

:: ── Étape 3 : Vérifier NVIDIA ───────────────────
echo  [3/4] Verification GPU NVIDIA...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ATTENTION] Drivers NVIDIA non detectes.
    echo  Installez-les depuis : https://www.nvidia.com/drivers
    echo  puis relancez ce fichier.
    echo.
    pause
    exit /b
)
echo  [OK] GPU NVIDIA detecte

:: ── Étape 4 : Cloner et installer ───────────────
echo  [4/4] Installation GaussianEditor C4D...
echo.

:: Vérifier si déjà cloné
wsl -e bash -ic "[ -d ~/GaussianEditor_C4D ]" >nul 2>&1
if %errorlevel% == 0 (
    echo  Repo deja present, mise a jour...
    wsl -e bash -ic "cd ~/GaussianEditor_C4D && git pull"
) else (
    echo  Clonage du repo...
    wsl -e bash -ic "cd ~ && git clone https://github.com/stamboulii/GaussianEditor_C4D"
)

echo.
echo  Lancement de l'installation automatique...
echo  Cette etape prend 45 a 90 minutes selon votre connexion.
echo  Ne fermez pas cette fenetre.
echo.

wsl -e bash -ic "cd ~/GaussianEditor_C4D && chmod +x install.sh && ./install.sh"

echo.
echo  ================================================
echo  Installation terminee !
echo.
echo  Pour utiliser GaussianEditor C4D :
echo  1. Double-cliquez sur Demarrer_GaussianEditor.bat
echo     sur votre Bureau Windows
echo  2. Ouvrez Cinema 4D
echo  3. Dans le plugin cliquez Demarrer
echo  ================================================
pause