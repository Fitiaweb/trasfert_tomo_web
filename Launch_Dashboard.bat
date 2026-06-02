@echo off
:: Force le script a se positionner dans le bon dossier
cd /d "%~dp0"
echo ===================================================
echo   LANCEMENT DU DASHBOARD SUIVI DOSES TOMO
echo ===================================================
echo.

:: Lancement direct avec le Python de ton environnement (ignore le Microsoft Store)
echo Demarrage du serveur web...
".venv\Scripts\python.exe" -m streamlit run Transfer_impact.py

:: Securite
echo.
echo Le serveur s'est arrete.
pause