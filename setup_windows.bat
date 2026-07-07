@echo off
title GaussianEditor C4D - Setup Windows
echo.
echo  ================================================
echo  GaussianEditor C4D -- Setup Windows
echo  ================================================
echo.

:: Verifier si WSL2 est installe
wsl --status >nul 2>&1
if %errorlevel% == 0 (
    echo  [OK] WSL2 deja installe
    goto :check_ubuntu
)

echo  [1/3] Installation WSL2...
echo  Cette operation necessite un redemarrage.
echo.
wsl --install
echo.
echo  Redemarrez votre PC puis relancez ce script.
pause
exit

:check_ubuntu
:: Verifier Ubuntu
wsl -d Ubuntu-24.04 echo "ok" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [2/3] Installation Ubuntu 24.04...
    wsl --install -d Ubuntu-24.04
    echo.
    echo  Creez votre nom d'utilisateur et mot de passe Ubuntu
    echo  puis relancez ce script.
    pause
    exit
)

echo  [OK] Ubuntu 24.04 disponible
echo.

:: Verifier NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ATTENTION] Drivers NVIDIA non detectes.
    echo  Installez les depuis : https://www.nvidia.com/drivers
    echo  puis relancez ce script.
    echo.
    pause
    exit
)

echo  [OK] GPU NVIDIA detecte
echo.

:: Cloner et installer
echo  [3/3] Installation GaussianEditor C4D...
echo.
echo  Dans la fenetre WSL2 qui va s'ouvrir :
echo  Les commandes seront executees automatiquement.
echo.

:: Lancer l'installation dans WSL2
wsl -e bash -ic "cd ~ && git clone https://github.com/TON_USER/GaussianEditor_C4D && cd GaussianEditor_C4D && chmod +x install.sh && ./install.sh"

echo.
echo  ================================================
echo  Installation terminee !
echo  Double-cliquez sur Demarrer_GaussianEditor.bat
echo  ================================================
pause