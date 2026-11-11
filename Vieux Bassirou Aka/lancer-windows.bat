@echo off
chcp 65001 >nul
echo === AgriVision : lancement du chatbot ===
where ollama >nul 2>nul || (echo Ollama n'est pas installe. Telechargez-le sur https://ollama.com/download & pause & exit /b)
taskkill /F /IM "ollama app.exe" >nul 2>nul
taskkill /F /IM ollama.exe >nul 2>nul
set OLLAMA_ORIGINS=*
start "Ollama" /min ollama serve
timeout /t 4 /nobreak >nul
echo Telechargement du modele (une seule fois, patientez)...
ollama pull llama3.2
start "" "%~dp0agrivision.html"
echo.
echo Pret. Le chatbot est dans la page qui vient de s'ouvrir.
echo Laissez cette fenetre ouverte pendant que vous l'utilisez.
pause
