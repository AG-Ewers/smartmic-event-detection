# create_shortcut.ps1
# Einmalig ausführen, um einen Desktop-Shortcut fuer die Mikroskop-Steuerung anzulegen.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetBat = Join-Path $ScriptDir "run_microscope.bat"
$IconFile = Join-Path $ScriptDir "Icon.ico"

$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Mikroskop-Steuerung.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath = $TargetBat
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.Description = "Mikroskop-Steuerung starten"
$Shortcut.WindowStyle = 1

# Eigenes Icon verwenden
if (Test-Path $IconFile) {
    $Shortcut.IconLocation = "$IconFile,0"
}

$Shortcut.Save()

Write-Host "Shortcut erstellt: $ShortcutPath"