"""Zentrale Konfiguration und Konstanten für die Mikroskop-Steuerung.

Diese Datei enthält ausschließlich Konstanten. Keine Logik, keine
Funktionen, keine Klassen.
"""

# --- FluidNC / GRBL Verbindung ---
GRBL_PORT: str = "COM6"
GRBL_BAUDRATE: int = 115200
GRBL_TIMEOUT: float = 2.0

# --- Bewegungssteuerung ---
JOG_FEEDRATE: int = 1000  # mm/min für Jog-Bewegungen
DEFAULT_FEEDRATE = 500
MAX_FEEDRATE = 5000

# --- Laser ---
LASER_MAX_S: int = 1000  # maximaler GRBL/FluidNC S-Wert (siehe Laser-Config)

# --- Kamera ---
STREAM_INTERVAL_MS: int = 70   # Aktualisierungsintervall Live-Bild in ms für Qt QTimer
LIVE_FPS = 15

# Belichtungszeit in Mikrosekunden. CameraManager fragt den Wertebereich
# nicht von der Hardware ab, daher hier feste Grenzen fuer die Slider.
# Bei Bedarf an die tatsaechlichen Grenzen der Kamera anpassen.
EXPOSURE_MIN_US: float = 200.0
EXPOSURE_MAX_US: float = 500000.0
EXPOSURE_DEFAULT_US: float = 40000.0

# Gain in dB, gleiche Einschraenkung wie bei der Belichtungszeit.
GAIN_MIN_DB: float = 0.0
GAIN_MAX_DB: float = 24.0
GAIN_DEFAULT_DB: float = 0.0

# --- Speicherort für Snapshots ---
IMAGE_DIR: str = "images"
IMAGE_EXTENSION: str = "png"  # spaeter auf "tiff" umstellen