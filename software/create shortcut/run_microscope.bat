@echo off
REM Startet die Mikroskop-Steuerung ueber das PySide6-Environment.
REM Diese Datei wird vom Desktop-Shortcut aufgerufen.

CALL "%USERPROFILE%\miniconda3\Scripts\activate.bat" pyside6

cd /d "%~dp0"

python main.py

REM Falls ein Fehler auftritt, Fenster offen halten, damit man ihn lesen kann.
if errorlevel 1 (
    echo.
    echo Es ist ein Fehler aufgetreten.
    pause
)
