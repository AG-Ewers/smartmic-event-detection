"""PySide6-Oberfläche für die Mikroskop-Steuerung.

Ersetzt app.py (Streamlit) durch eine native Qt-Desktop-Anwendung.
Enthält ausschließlich UI-Code. Sämtliche Hardware-Logik liegt weiterhin
in camera.py, stage.py, laser.py und autofocus.py - unverändert.

Start:
    python main.py
"""

from __future__ import annotations

import base64
import contextlib
import csv
import io
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autofocus import autofocus_sweep
from calibration import (
    CalibrationResult,
    load_calibration,
    run_calibration,
    save_calibration,
)
from camera import CameraManager
from event_acquisition import AnalysisResult, run_event_analysis
from laser import LaserController
from settings import (
    EXPOSURE_DEFAULT_US,
    EXPOSURE_MAX_US,
    EXPOSURE_MIN_US,
    GAIN_DEFAULT_DB,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    IMAGE_DIR,
    IMAGE_EXTENSION,
    STREAM_INTERVAL_MS,
)
from stage import StageController
from utils import save_image, sharpness_score, timestamped_filename

STEP_SIZE_OPTIONS_UM = [1, 10, 100, 1000, 10000]

# Kleine Dreieck-Icons (12x7px, hellgrau, transparenter Hintergrund) für die
# Spinbox-Pfeile im Dark Mode. Ein reiner CSS-Border-Dreieck-Trick rendert in
# Qt/Windows nicht zuverlässig als Dreieck (nur als flacher Strich), und
# eingebettete data:-URIs werden von Qt-Stylesheets stillschweigend
# ignoriert - Qt braucht dafür einen echten Dateipfad. Deshalb werden diese
# winzigen PNGs beim Start einmalig aus den Base64-Daten unten ins
# Temp-Verzeichnis geschrieben (main.py bleibt trotzdem eine einzelne Datei).
_SPINBOX_ARROW_UP_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAwAAAAHCAYAAAA8sqwkAAAANklEQVR4nGNgwAIePHjwH5s4"
    "AwMDAxMuxbg0MWFTjM8mDBsIOY8JlwQuTUyEFKNrYiRGMTIAAAehJ+FKw0SeAAAAAElFTkSu"
    "QmCC"
)
_SPINBOX_ARROW_DOWN_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAwAAAAHCAYAAAA8sqwkAAAAOElEQVR4nGN88ODBfwYSAJOC"
    "ggIjsYoVFBQYmWAMYhQzMDAwMKEL4FOMooEYxRga0CWJ9h++kAMAlhgKsyO6/ioAAAAASUVO"
    "RK5CYII="
)


def _write_spinbox_arrow_icons() -> tuple[str, str]:
    """Schreibt die Pfeil-Icons einmalig ins Temp-Verzeichnis und gibt ihre
    Pfade zurück (als Posix-Pfade, wie von Qt-Stylesheets benötigt)."""
    icon_dir = Path(tempfile.gettempdir()) / "mikroskop_ui_icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    up_path = icon_dir / "spinbox_arrow_up.png"
    down_path = icon_dir / "spinbox_arrow_down.png"

    up_path.write_bytes(base64.b64decode(_SPINBOX_ARROW_UP_PNG_B64))
    down_path.write_bytes(base64.b64decode(_SPINBOX_ARROW_DOWN_PNG_B64))

    return up_path.as_posix(), down_path.as_posix()


_SPINBOX_ARROW_UP_PATH, _SPINBOX_ARROW_DOWN_PATH = _write_spinbox_arrow_icons()

# Einfaches Dark-Mode-Stylesheet - bewusst schlicht gehalten, verändert nur
# Farben der bestehenden nativen Widgets, kein Redesign von Layout/Struktur.
DARK_STYLESHEET = f"""
QWidget {{
    background-color: #2b2b2b;
    color: #e0e0e0;
}}
QGroupBox {{
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #e0e0e0;
}}
QPushButton {{
    background-color: #3c3f41;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 4px 10px;
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: #4b4f52;
}}
QPushButton:checked {{
    background-color: #6c8dff;
    color: #ffffff;
    border: 1px solid #8ab4f8;
}}
QPushButton:checked:hover {{
    background-color: #7d9bff;
}}
QPushButton:disabled {{
    color: #808080;
}}
QLabel {{
    background: transparent;
}}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit {{
    background-color: #3c3f41;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 3px;
    selection-background-color: #5a8dee;
}}
/* Ohne diese expliziten Subcontrol-Regeln berechnet der native
   Windows-Stil die Klickfläche des "Pfeil hoch"-Buttons an Spinboxen
   nach dem Setzen eines eigenen Rahmens (siehe oben) teils falsch/auf
   nahezu 0px, wodurch er nicht mehr anklickbar ist - "Pfeil runter"
   bleibt davon meist unberührt. Deshalb hier beide Buttons explizit
   mit fester Größe/Position definieren, inkl. eigener Pfeil-Icons (ein
   reiner CSS-Dreieck-Trick rendert in Qt/Windows nicht zuverlässig). */
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    height: 11px;
    border-left: 1px solid #555555;
    border-bottom: 1px solid #555555;
    border-top-right-radius: 3px;
    background-color: #3c3f41;
}}
QAbstractSpinBox::up-button:hover {{
    background-color: #4b4f52;
}}
QAbstractSpinBox::up-button:pressed {{
    background-color: #2b2b2b;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    height: 11px;
    border-left: 1px solid #555555;
    border-bottom-right-radius: 3px;
    background-color: #3c3f41;
}}
QAbstractSpinBox::down-button:hover {{
    background-color: #4b4f52;
}}
QAbstractSpinBox::down-button:pressed {{
    background-color: #2b2b2b;
}}
QAbstractSpinBox::up-arrow {{
    image: url({_SPINBOX_ARROW_UP_PATH});
    width: 8px;
    height: 5px;
}}
QAbstractSpinBox::down-arrow {{
    image: url({_SPINBOX_ARROW_DOWN_PATH});
    width: 8px;
    height: 5px;
}}
QComboBox QAbstractItemView {{
    background-color: #3c3f41;
    color: #e0e0e0;
    selection-background-color: #5a8dee;
}}
QRadioButton, QCheckBox {{
    background: transparent;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 8px;
    border: 2px solid #7a7f83;
    background-color: #2b2b2b;
}}
QRadioButton::indicator:hover {{
    border: 2px solid #8ab4f8;
}}
QRadioButton::indicator:checked {{
    background-color: #6c8dff;
    border: 2px solid #8ab4f8;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 2px solid #7a7f83;
    background-color: #2b2b2b;
}}
QCheckBox::indicator:hover {{
    border: 2px solid #8ab4f8;
}}
QCheckBox::indicator:checked {{
    background-color: #6c8dff;
    border: 2px solid #8ab4f8;
}}
QSlider::groove:horizontal {{
    background: #555555;
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #9aa0a6;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QProgressBar {{
    background-color: #3c3f41;
    border: 1px solid #555555;
    border-radius: 3px;
    text-align: center;
    color: #e0e0e0;
}}
QProgressBar::chunk {{
    background-color: #5a8dee;
}}
QScrollBar:vertical {{
    background: #2b2b2b;
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: #555555;
    border-radius: 6px;
    min-height: 24px;
}}
"""


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Konvertiert ein Graustufen-NumPy-Array in ein QPixmap für die Anzeige."""
    if img.ndim == 3:
        img = img[:, :, 0]
    img = np.ascontiguousarray(img)
    height, width = img.shape
    qimage = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimage.copy())


def um_to_mm(value_um: float) -> float:
    """Rechnet Mikrometer in Millimeter um."""
    return value_um / 1000.0


def mm_to_um(value_mm: float) -> float:
    """Rechnet Millimeter in Mikrometer um."""
    return value_mm * 1000.0


# ---------------------------------------------------------------------------
# Autofokus-Worker (läuft in eigenem Thread, damit die GUI nicht einfriert)
# ---------------------------------------------------------------------------


class CameraBridge:
    """Stellt sicher, dass JEDER Zugriff auf die vmbpy-Kamera (get_frame,
    set_exposure, get_exposure, ...) IMMER im GUI-Thread läuft.

    vmbpy/VmbSystem ist an den Thread gebunden, der die Verbindung geöffnet
    hat (das ist bei uns der GUI-Thread, über den "Verbinden"-Button). Ruft
    ein Worker-Thread (Autofokus, Kalibrierung, Event-Analyse) irgendeine
    Kamera-Methode direkt auf, kann das den internen SDK-Kontext
    korrumpieren ("outside of with context"-Fehler, teils erst beim
    nächsten Verbindungsversuch sichtbar - siehe die Kalibrierungs-Odyssee
    weiter oben in diesem Projekt).

    Lösung: Worker-Threads fragen hier an (call_from_worker) und WARTEN
    (blockierend), bis der GUI-Thread (via poll_pending(), regelmäßig aus
    dem Live-View-Timer aufgerufen) den eigentlichen Aufruf tatsächlich
    ausgeführt hat.
    """

    def __init__(self, camera: CameraManager) -> None:
        self.camera = camera
        self._event = threading.Event()
        self._result: Any = None
        self._pending_call: Optional[Tuple[str, tuple]] = None
        self._lock = threading.Lock()

    def call_from_worker(self, method_name: str, *args: Any, timeout_s: float = 5.0) -> Any:
        """Von einem Worker-Thread aufrufen, z.B.
        bridge.call_from_worker('set_exposure', 100000.0). Blockiert bis
        das Ergebnis da ist (oder das Timeout erreicht ist, z.B. wenn der
        GUI-Thread gerade selbst hängt - dann wird None zurückgegeben)."""
        with self._lock:
            self._event.clear()
            self._pending_call = (method_name, args)
        got = self._event.wait(timeout=timeout_s)
        if not got:
            return None
        return self._result

    def get_frame_from_worker(self, timeout_s: float = 5.0) -> Optional[np.ndarray]:
        """Bequemer Alias für den häufigsten Fall."""
        return self.call_from_worker("get_frame", timeout_s=timeout_s)

    def poll_pending(self) -> None:
        """Im GUI-Thread regelmäßig aufrufen (hier: aus dem Live-View-Timer).
        Führt den angefragten Kamera-Aufruf aus, falls gerade eine
        Worker-Anfrage vorliegt."""
        with self._lock:
            pending = self._pending_call
        if pending is None:
            return
        method_name, args = pending
        method = getattr(self.camera, method_name)
        result = method(*args)
        with self._lock:
            self._result = result
            self._pending_call = None
        self._event.set()


class ThreadSafeCameraProxy:
    """Duck-typed Ersatz für CameraManager für die Übergabe an Worker-
    Threads (AutofocusWorker, CalibrationWorker, EventAnalysisWorker):
    alle Kamera-Zugriffe laufen über die CameraBridge im GUI-Thread."""

    def __init__(self, bridge: CameraBridge) -> None:
        self._bridge = bridge

    def get_frame(self) -> Optional[np.ndarray]:
        return self._bridge.get_frame_from_worker()

    def set_exposure(self, exposure_us: float) -> None:
        self._bridge.call_from_worker("set_exposure", exposure_us)

    def get_exposure(self) -> Optional[float]:
        return self._bridge.call_from_worker("get_exposure")

    def set_roi(self, x: float, y: float, width: float, height: float) -> bool:
        result = self._bridge.call_from_worker("set_roi", x, y, width, height)
        return bool(result)

    def reset_roi(self) -> bool:
        result = self._bridge.call_from_worker("reset_roi")
        return bool(result)

    @property
    def last_error(self) -> Optional[str]:
        return self._bridge.camera.last_error

    @property
    def is_connected(self) -> bool:
        return self._bridge.camera.is_connected


class StageConnectWorker(QThread):
    """Verbindet die Stage in einem eigenen Thread.

    stage.connect() kann - je nach Hardware/Port - mehrere Sekunden dauern
    oder sogar hängen bleiben (z.B. wenn kein Gerät antwortet). Lief das
    bisher direkt im GUI-Thread, fror die gesamte Oberfläche ein und Windows
    zeigte "Keine Rückmeldung" an. Hier läuft der Verbindungsversuch im
    Hintergrund, die GUI bleibt bedienbar.
    """

    finished_connect = Signal(bool, str)  # (erfolgreich, Fehlertext)

    def __init__(self, stage: StageController, port: str) -> None:
        super().__init__()
        self.stage = stage
        self.port = port

    def run(self) -> None:
        try:
            if self.port:
                self.stage.set_port(self.port)
            self.stage.connect()
            success = self.stage.is_connected
            error_text = "" if success else (self.stage.last_error or "Keine Verbindung")
        except Exception as exc:  # noqa: BLE001 - Verbindungsfehler sollen die GUI nicht crashen
            success = False
            error_text = str(exc)
        self.finished_connect.emit(success, error_text)


class AutofocusWorker(QThread):
    """Führt autofocus_sweep() in einem separaten Thread aus.

    autofocus_sweep() ist eine blockierende, synchrone Funktion (mehrere
    time.sleep()-Aufrufe). Würde sie direkt im GUI-Thread laufen, würde die
    gesamte Oberfläche für die Dauer des Sweeps einfrieren.
    """

    progress = Signal(int, int, float, float)
    finished_with_result = Signal(object)

    def __init__(
        self,
        stage: StageController,
        camera: CameraManager,
        range_mm: float,
        step_mm: float,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.camera = camera
        self.range_mm = range_mm
        self.step_mm = step_mm

    def run(self) -> None:
        """Führt den Sweep aus und meldet Fortschritt/Ergebnis über Signale."""

        def on_progress(i: int, n: int, offset: float, score: float) -> None:
            self.progress.emit(i, n, offset, score)

        best_offset, results = autofocus_sweep(
            self.stage,
            self.camera,
            self.range_mm,
            self.step_mm,
            progress_callback=on_progress,
        )
        self.finished_with_result.emit((best_offset, results))


class CalibrationWorker(QThread):
    """Führt run_calibration() in einem separaten Thread aus.

    Gleiche Begründung wie bei AutofocusWorker: mehrere blockierende
    Bewegungs-/Bildaufnahme-Schritte hintereinander würden sonst die
    gesamte Oberfläche für die Dauer der Kalibrierung einfrieren.
    """

    progress = Signal(str)
    finished_with_result = Signal(object)  # CalibrationResult bei Erfolg, sonst Exception

    def __init__(
        self,
        stage: StageController,
        camera: CameraManager,
        step_um: float,
        n_repeats: int = 3,
        laser: Optional[LaserController] = None,
        laser_percent: float = 50.0,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.camera = camera
        self.step_um = step_um
        self.n_repeats = n_repeats
        self.laser = laser
        self.laser_percent = laser_percent

    def run(self) -> None:
        def on_progress(msg: str) -> None:
            self.progress.emit(msg)

        try:
            result = run_calibration(
                self.stage,
                self.camera,
                step_um=self.step_um,
                n_repeats=self.n_repeats,
                laser=self.laser,
                laser_percent=self.laser_percent,
                progress_callback=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - Fehler sollen die GUI nicht crashen
            self.finished_with_result.emit(exc)
            return

        self.finished_with_result.emit(result)


class EventAnalysisWorker(QThread):
    """Führt run_event_analysis() in einem separaten Thread aus.

    Gleiche Begründung wie bei AutofocusWorker/CalibrationWorker: mehrere
    blockierende Bewegungs-/Bildaufnahme-Schritte hintereinander würden
    sonst die gesamte Oberfläche für die Dauer der Analyse+Aufnahmeserie
    einfrieren.
    """

    progress = Signal(str)
    finished_with_result = Signal(object)  # AnalysisResult bei Erfolg, sonst Exception

    def __init__(
        self,
        stage: StageController,
        camera: CameraManager,
        image_dir: str,
        z_stack_range_mm: float,
        z_stack_step_mm: float,
        exposure_presets_us: tuple[float, ...],
        laser: Optional[LaserController] = None,
        name_prefix: Optional[str] = None,
        run_index: Optional[int] = None,
        extensions: tuple[str, ...] = ("tif",),
        use_hardware_roi: bool = False,
        roi_margin_px: float = 50.0,
        save_no_hit_images: bool = False,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.camera = camera
        self.image_dir = image_dir
        self.z_stack_range_mm = z_stack_range_mm
        self.z_stack_step_mm = z_stack_step_mm
        self.exposure_presets_us = exposure_presets_us
        self.extensions = extensions
        self.laser = laser
        self.name_prefix = name_prefix
        self.run_index = run_index
        self.use_hardware_roi = use_hardware_roi
        self.roi_margin_px = roi_margin_px
        self.save_no_hit_images = save_no_hit_images

    def run(self) -> None:
        def on_progress(msg: str) -> None:
            self.progress.emit(msg)

        try:
            result = run_event_analysis(
                self.stage,
                self.camera,
                self.image_dir,
                z_stack_range_mm=self.z_stack_range_mm,
                z_stack_step_mm=self.z_stack_step_mm,
                exposure_presets_us=self.exposure_presets_us,
                laser=self.laser,
                name_prefix=self.name_prefix,
                run_index=self.run_index,
                extensions=self.extensions,
                use_hardware_roi=self.use_hardware_roi,
                roi_margin_px=self.roi_margin_px,
                save_no_hit_images=self.save_no_hit_images,
                progress_callback=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - Fehler sollen die GUI nicht crashen
            self.finished_with_result.emit(exc)
            return

        self.finished_with_result.emit(result)


# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Hauptfenster der Mikroskop-Steuerung."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mikroskop-Steuerung")
        self.resize(1400, 900)
        # Damit Pfeiltasten auch dann ankommen, wenn kein Eingabefeld
        # (Spinbox etc.) gerade den Fokus haelt - z.B. nach Klick auf das
        # Live-Bild oder eine Gruppierung.
        self.setFocusPolicy(Qt.StrongFocus)

        self.camera = CameraManager()
        self.camera_bridge = CameraBridge(self.camera)
        self.stage = StageController()
        self.laser = LaserController(self.stage)

        self.last_image: Optional[np.ndarray] = None
        self.step_size_um: Dict[str, float] = {"X": 10.0, "Y": 10.0, "Z": 10.0}
        self.axis_step_groups: Dict[str, QButtonGroup] = {}
        self.axis_step_radios: Dict[str, Dict[int, QRadioButton]] = {}
        self.axis_custom_step_spinboxes: Dict[str, QDoubleSpinBox] = {}
        self.position_spinboxes: Dict[str, QDoubleSpinBox] = {}
        self.autofocus_worker: Optional[AutofocusWorker] = None
        # Logischer Nullpunkt fuer Koordinaten-Tracking (Maschinen-Position
        # zum Zeitpunkt des letzten Drucks auf "Koordinaten zuruecksetzen").
        # None = noch nicht gesetzt.
        self.coord_zero_offset: Optional[Tuple[float, float, float]] = None
        self.coord_log_path: Path = Path(IMAGE_DIR) / "koordinaten_log.csv"
        self.calibration_worker: Optional[CalibrationWorker] = None
        self.analysis_worker: Optional[EventAnalysisWorker] = None
        self.event_counter: int = 1
        # Beim Start vorhandene Kalibrierung laden, falls schon einmal
        # durchgeführt - muss dann nicht bei jedem Programmstart wiederholt
        # werden. UI-Anzeige dazu wird erst weiter unten aufgebaut, daher
        # hier nur den Wert laden; das Label wird in _build_calibration_group
        # bzw. direkt danach im __init__ aktualisiert.
        self.calibration_result: Optional[CalibrationResult] = load_calibration()
        self._cal_prior_laser_on: Optional[bool] = None
        self.stage_connect_worker: Optional[StageConnectWorker] = None
        self.stage_connect_timeout_timer: Optional[QTimer] = None
        self._stage_connect_timed_out = False

        # Speichert Fenster-/Anzeigeeinstellungen (z.B. Dark Mode) zwischen
        # Programmstarts (Windows: Registry, Linux/Mac: Konfigdatei).
        self.app_settings = QSettings("MikroskopSteuerung", "MikroskopApp")

        self._build_ui()

        # Zuletzt gespeicherte Dark-Mode-Einstellung wiederherstellen.
        # Falls noch nie gespeichert (erster Start): Standardmäßig dunkel.
        saved_dark_mode = self.app_settings.value("dark_mode", True, type=bool)
        self.dark_mode_button.setChecked(saved_dark_mode)

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._update_live_view)
        self.live_timer.start(STREAM_INTERVAL_MS)

    # -----------------------------------------------------------------
    # UI-Aufbau
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        """Baut das komplette Widget-Layout auf."""
        central = QWidget()

        # In eine QScrollArea packen: falls das Fenster nicht hoch genug ist,
        # um alles (inkl. Skript-Bereich ganz unten) darzustellen, kann man
        # jetzt nach unten scrollen statt dass der untere Teil abgeschnitten
        # wird.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setWidget(central)
        self.setCentralWidget(scroll_area)

        outer_layout = QVBoxLayout(central)

        # Kleine Kopfzeile nur für den Dark-Mode-Umschalter - Rest des Layouts
        # bleibt unverändert.
        header_row = QHBoxLayout()
        header_row.addStretch(1)
        self.dark_mode_button = QPushButton("Dark Mode")
        self.dark_mode_button.setCheckable(True)
        self.dark_mode_button.toggled.connect(self._toggle_dark_mode)
        header_row.addWidget(self.dark_mode_button)
        outer_layout.addLayout(header_row)

        top_layout = QHBoxLayout()
        outer_layout.addLayout(top_layout)

        left_column = QVBoxLayout()
        top_layout.addLayout(left_column, stretch=2)
        left_column.addWidget(self._build_camera_group())

        right_column = QVBoxLayout()
        top_layout.addLayout(right_column, stretch=1)

        stage_group = self._build_stage_group()
        laser_group = self._build_laser_group()
        analysis_group = self._build_analysis_group()
        autofocus_group = self._build_autofocus_group()
        calibration_group = self._build_calibration_group()

        # Referenzen merken, damit wir nach dem ersten Layout-Durchlauf ihre
        # Maximalhöhe fixieren können (siehe _lock_side_panel_heights).
        self.stage_group = stage_group
        self.laser_group = laser_group
        self.autofocus_group = autofocus_group
        self.calibration_group = calibration_group
        self.analysis_group = analysis_group

        right_column.addWidget(stage_group)
        right_column.addWidget(laser_group)
        right_column.addWidget(analysis_group)
        right_column.addWidget(autofocus_group)
        right_column.addWidget(calibration_group)
        # Fängt den restlichen, ungenutzten Platz ab, statt die Panels darüber
        # zu strecken.
        right_column.addStretch(1)

        outer_layout.addWidget(self._build_script_group())

        # Fix: setSizePolicy(..., Maximum) allein reicht nicht - Qt berechnet
        # sizeHint() beim ersten Aufbau teils noch mit einer zu kleinen Breite,
        # wodurch die Stage Control bei Vollbild "zusammengestaucht" gerendert
        # wurde (und erst nach z.B. einem Stylesheet-Wechsel, der ein
        # komplettes Neu-Layout erzwingt, wieder korrekt aussah). Deshalb wird
        # die Maximalhöhe erst fixiert, nachdem die Fensterlayout einmal
        # vollständig durchgelaufen ist.
        QTimer.singleShot(0, self._lock_side_panel_heights)

    def _lock_side_panel_heights(self) -> None:
        """Fixiert die Höhe von Stage/Laser/Autofokus-Panel auf ihre tatsächlich
        benötigte Größe, nachdem das Layout einmal komplett berechnet wurde."""
        for grp in (
            self.stage_group, self.laser_group, self.autofocus_group,
            self.calibration_group, self.analysis_group,
        ):
            grp.setMaximumHeight(16_777_215)  # Qt-Maximum zurücksetzen, bevor neu gemessen wird
            grp.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            grp.updateGeometry()
            grp.setMaximumHeight(grp.sizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt-Methodenname
        """Bei jeder Fenstergrößenänderung (auch Wechsel in den Vollbildmodus)
        die Panelhöhen neu fixieren.

        Vorher wurde das nur einmal beim Start gemacht - Qt berechnet
        sizeHint() bei manchen Übergängen (z.B. Vollbild unter Windows) aber
        offenbar mit veralteten Metriken, wodurch die Stage Control
        "gestaucht" gerendert wurde, bis irgendein anderes Ereignis (z.B. ein
        Stylesheet-Wechsel) ein komplettes Neu-Layout erzwang. Jetzt wird
        nach jedem Resize neu gemessen, das ist günstig genug, um es bei
        jedem Resize-Event zu tun.
        """
        super().resizeEvent(event)
        QTimer.singleShot(0, self._lock_side_panel_heights)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt-Methodenname
        """Fängt zusätzlich Fensterstatus-Änderungen ab (z.B. Maximieren per
        Doppelklick auf die Titelleiste), falls dabei kein reguläres
        resizeEvent ausgelöst wird."""
        super().changeEvent(event)
        if event.type() == event.Type.WindowStateChange:
            QTimer.singleShot(0, self._lock_side_panel_heights)

    def _build_camera_group(self) -> QGroupBox:
        """Baut den Kamera-Bereich: Verbindung, Livebild, Belichtung/Gain."""
        group = QGroupBox("Kamera")
        layout = QVBoxLayout(group)

        conn_row = QHBoxLayout()
        self.camera_connect_button = QPushButton("Verbinden")
        self.camera_connect_button.setCheckable(True)
        self.camera_connect_button.toggled.connect(self._on_camera_toggle)
        conn_row.addWidget(self.camera_connect_button)

        self.camera_status_label = QLabel("Nicht verbunden")
        conn_row.addWidget(self.camera_status_label)
        conn_row.addStretch(1)
        layout.addLayout(conn_row)

        self.live_view_label = QLabel()
        self.live_view_label.setMinimumSize(640, 480)
        self.live_view_label.setAlignment(Qt.AlignCenter)
        self.live_view_label.setStyleSheet("background-color: black;")
        self.live_view_label.setText("Kein Bild")
        layout.addWidget(self.live_view_label)

        self.sharpness_label = QLabel("Schärfe-Score: -")
        layout.addWidget(self.sharpness_label)

        save_row = QHBoxLayout()
        self.save_image_button = QPushButton("Bild speichern")
        self.save_image_button.clicked.connect(self._on_save_image)
        save_row.addWidget(self.save_image_button)

        save_row.addWidget(QLabel("Format:"))
        self.format_button_group = QButtonGroup(self)
        self.format_radio_png = QRadioButton("PNG")
        self.format_radio_tif = QRadioButton("TIF")
        self.format_radio_both = QRadioButton("Beide")
        self.format_radio_png.setChecked(True)  # Standard: bisheriges Verhalten
        for rb in (self.format_radio_png, self.format_radio_tif, self.format_radio_both):
            self.format_button_group.addButton(rb)
            save_row.addWidget(rb)

        save_row.addStretch(1)
        layout.addLayout(save_row)

        settings_form = QFormLayout()

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(int(EXPOSURE_MIN_US))
        self.exposure_slider.setMaximum(int(EXPOSURE_MAX_US))
        self.exposure_slider.setValue(int(EXPOSURE_DEFAULT_US))
        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setMinimum(EXPOSURE_MIN_US)
        self.exposure_spinbox.setMaximum(EXPOSURE_MAX_US)
        self.exposure_spinbox.setValue(EXPOSURE_DEFAULT_US)
        self.exposure_slider.valueChanged.connect(self._on_exposure_slider_changed)
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_spinbox_changed)

        exposure_row = QHBoxLayout()
        exposure_row.addWidget(self.exposure_slider)
        exposure_row.addWidget(self.exposure_spinbox)
        settings_form.addRow("Belichtungszeit (µs):", exposure_row)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(int(GAIN_MIN_DB))
        self.gain_slider.setMaximum(int(GAIN_MAX_DB))
        self.gain_slider.setValue(int(GAIN_DEFAULT_DB))
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setMinimum(GAIN_MIN_DB)
        self.gain_spinbox.setMaximum(GAIN_MAX_DB)
        self.gain_spinbox.setValue(GAIN_DEFAULT_DB)
        self.gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self.gain_spinbox.valueChanged.connect(self._on_gain_spinbox_changed)

        gain_row = QHBoxLayout()
        gain_row.addWidget(self.gain_slider)
        gain_row.addWidget(self.gain_spinbox)
        settings_form.addRow("Gain (dB):", gain_row)

        self.event_name_input = QLineEdit()
        self.event_name_input.setPlaceholderText("z.B. HeLa_2026-09-16 (leer = Zeitstempel)")
        self.event_name_input.setToolTip(
            "Basisname für von 'Analysieren' gespeicherte Bilder. Läuft "
            "hoch als <Name>_0001_event, <Name>_0002_noevent, ... - Zahl "
            "VOR dem Label (mit führenden Nullen), damit die alphabetische "
            "Windows-Sortierung mit der zeitlichen Reihenfolge "
            "übereinstimmt. Leer lassen für den bisherigen "
            "zeitstempel-basierten Namen."
        )
        settings_form.addRow("Bildname (Analyse):", self.event_name_input)

        layout.addLayout(settings_form)

        return group

    def _build_stage_group(self) -> QGroupBox:
        """Baut den Stage-Bereich: Verbindung, Positionierung, Achsen, Global Speed."""
        group = QGroupBox("Stage Control")
        layout = QVBoxLayout(group)

        conn_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()
        conn_row.addWidget(self.port_combo)

        refresh_button = QPushButton("Aktualisieren")
        refresh_button.clicked.connect(self._refresh_ports)
        conn_row.addWidget(refresh_button)

        self.stage_connect_button = QPushButton("Verbinden")
        self.stage_connect_button.setCheckable(True)
        self.stage_connect_button.toggled.connect(self._on_stage_toggle)
        conn_row.addWidget(self.stage_connect_button)
        layout.addLayout(conn_row)

        self.stage_status_label = QLabel("Nicht verbunden")
        layout.addWidget(self.stage_status_label)

        pos_group = QGroupBox("Position Control (µm) - Wert eingeben und bestätigen")
        pos_layout = QGridLayout(pos_group)
        for col, axis in enumerate(["X", "Y", "Z"]):
            pos_layout.addWidget(QLabel(f"{axis}-Achse"), 0, col)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1_000_000.0, 1_000_000.0)
            spinbox.setDecimals(1)
            spinbox.setValue(0.0)
            self.position_spinboxes[axis] = spinbox
            pos_layout.addWidget(spinbox, 1, col)

        confirm_button = QPushButton("Bestätigen - Position anfahren")
        confirm_button.clicked.connect(self._on_move_to_confirmed)
        pos_layout.addWidget(confirm_button, 2, 0, 1, 3)

        reset_coords_button = QPushButton("Koordinaten zurücksetzen")
        reset_coords_button.setToolTip(
            "Setzt die aktuelle Position als neuen Nullpunkt (0,0,0) für die "
            "Koordinaten-Tabelle. Wirkt nur auf das Tracking/Logging, nicht "
            "auf die tatsächliche Maschinen-Position."
        )
        reset_coords_button.clicked.connect(self._on_reset_coordinates)
        pos_layout.addWidget(reset_coords_button, 3, 0, 1, 3)

        self.coord_zero_status_label = QLabel("Kein Nullpunkt gesetzt")
        pos_layout.addWidget(self.coord_zero_status_label, 4, 0, 1, 3)
        layout.addWidget(pos_group)

        axes_row = QHBoxLayout()
        for axis in ["X", "Y", "Z"]:
            axes_row.addWidget(self._build_axis_group(axis))
        layout.addLayout(axes_row)

        bottom_row = QHBoxLayout()
        speed_label = QLabel("Global Speed (mm/min):")
        self.global_speed_spinbox = QSpinBox()
        self.global_speed_spinbox.setRange(1, 100000)
        self.global_speed_spinbox.setValue(self.stage.jog_feedrate)
        self.global_speed_spinbox.valueChanged.connect(self._on_global_speed_changed)
        bottom_row.addWidget(speed_label)
        bottom_row.addWidget(self.global_speed_spinbox)
        bottom_row.addStretch(1)

        home_all_button = QPushButton("HOME ALL")
        home_all_button.clicked.connect(lambda: self.stage.home())
        bottom_row.addWidget(home_all_button)

        stop_all_button = QPushButton("STOP ALL")
        stop_all_button.setStyleSheet("background-color: #c0392b; color: white;")
        stop_all_button.clicked.connect(lambda: self.stage.soft_reset())
        bottom_row.addWidget(stop_all_button)

        layout.addLayout(bottom_row)

        return group

    def _build_axis_group(self, axis: str) -> QGroupBox:
        """Baut das Steuerungspanel für eine einzelne Achse (X, Y oder Z)."""
        group = QGroupBox(f"{axis} Achse")
        layout = QVBoxLayout(group)

        jog_row = QHBoxLayout()
        minus_button = QPushButton("-")
        minus_button.clicked.connect(lambda: self._on_jog(axis, negative=True))
        plus_button = QPushButton("+")
        plus_button.clicked.connect(lambda: self._on_jog(axis, negative=False))
        jog_row.addWidget(minus_button)
        jog_row.addWidget(plus_button)
        layout.addLayout(jog_row)

        home_button = QPushButton("HOME")
        home_button.clicked.connect(lambda: self.stage.home_axis(axis))
        layout.addWidget(home_button)

        stop_button = QPushButton("STOP")
        stop_button.setStyleSheet("color: #c0392b;")
        stop_button.clicked.connect(lambda: self.stage.jog_cancel())
        layout.addWidget(stop_button)

        layout.addWidget(QLabel("Step Size:"))
        button_group = QButtonGroup(group)
        radios: Dict[int, QRadioButton] = {}
        for option in STEP_SIZE_OPTIONS_UM:
            radio = QRadioButton(f"{option} µm")
            if option == int(self.step_size_um[axis]):
                radio.setChecked(True)
            button_group.addButton(radio)
            radios[option] = radio
            radio.toggled.connect(
                lambda checked, a=axis, o=option: self._on_step_size_changed(a, o, checked)
            )
            layout.addWidget(radio)

        custom_row = QHBoxLayout()
        custom_radio = QRadioButton("Custom:")
        button_group.addButton(custom_radio)
        radios[-1] = custom_radio  # -1 = Marker fuer "Custom" in axis_step_radios
        custom_spinbox = QDoubleSpinBox()
        custom_spinbox.setRange(0.1, 100_000.0)
        custom_spinbox.setDecimals(1)
        custom_spinbox.setSuffix(" µm")
        custom_spinbox.setValue(self.step_size_um[axis])
        custom_radio.toggled.connect(
            lambda checked, a=axis: self._on_custom_step_size_toggled(a, checked)
        )
        custom_spinbox.valueChanged.connect(
            lambda value, a=axis: self._on_custom_step_size_value_changed(a, value)
        )
        self.axis_custom_step_spinboxes[axis] = custom_spinbox
        custom_row.addWidget(custom_radio)
        custom_row.addWidget(custom_spinbox)
        layout.addLayout(custom_row)

        self.axis_step_groups[axis] = button_group
        self.axis_step_radios[axis] = radios

        return group

    def _build_laser_group(self) -> QGroupBox:
        """Baut den Laser-Bereich: Toggle und Intensität in Prozent."""
        group = QGroupBox("Laser")
        layout = QVBoxLayout(group)

        self.laser_toggle_button = QPushButton("Laser aktiv")
        self.laser_toggle_button.setCheckable(True)
        self.laser_toggle_button.toggled.connect(self._on_laser_toggle)
        layout.addWidget(self.laser_toggle_button)

        intensity_row = QHBoxLayout()
        self.laser_slider = QSlider(Qt.Horizontal)
        self.laser_slider.setRange(0, 100)
        self.laser_slider.setValue(20)
        self.laser_spinbox = QSpinBox()
        self.laser_spinbox.setRange(0, 100)
        self.laser_spinbox.setValue(20)
        self.laser_slider.valueChanged.connect(self._on_laser_slider_changed)
        self.laser_spinbox.valueChanged.connect(self._on_laser_spinbox_changed)
        intensity_row.addWidget(self.laser_slider)
        intensity_row.addWidget(self.laser_spinbox)
        intensity_row.addWidget(QLabel("%"))
        layout.addLayout(intensity_row)

        return group

    def _build_autofocus_group(self) -> QGroupBox:
        """Baut den Autofokus-Bereich."""
        group = QGroupBox("Autofokus")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.af_range_spinbox = QDoubleSpinBox()
        self.af_range_spinbox.setRange(0.001, 100.0)
        self.af_range_spinbox.setDecimals(3)
        self.af_range_spinbox.setValue(0.1)
        form.addRow("Bereich +/- (mm):", self.af_range_spinbox)

        self.af_step_spinbox = QDoubleSpinBox()
        self.af_step_spinbox.setRange(0.001, 10.0)
        self.af_step_spinbox.setDecimals(3)
        self.af_step_spinbox.setValue(0.01)
        form.addRow("Schrittweite (mm):", self.af_step_spinbox)
        layout.addLayout(form)

        self.af_start_button = QPushButton("Autofokus starten")
        self.af_start_button.clicked.connect(self._on_autofocus_start)
        layout.addWidget(self.af_start_button)

        self.af_progress_bar = QProgressBar()
        layout.addWidget(self.af_progress_bar)

        self.af_status_label = QLabel("")
        layout.addWidget(self.af_status_label)

        return group

    def _build_calibration_group(self) -> QGroupBox:
        """Baut den Bereich zur Pixel-zu-Stage-Kalibrierung."""
        group = QGroupBox("Pixel-Kalibrierung (für automatische Zentrierung)")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.cal_step_spinbox = QDoubleSpinBox()
        self.cal_step_spinbox.setRange(0.1, 2000.0)
        self.cal_step_spinbox.setDecimals(1)
        self.cal_step_spinbox.setValue(300.0)
        form.addRow("Test-Schrittweite (µm):", self.cal_step_spinbox)

        self.cal_repeats_spinbox = QSpinBox()
        self.cal_repeats_spinbox.setRange(1, 10)
        self.cal_repeats_spinbox.setValue(3)
        self.cal_repeats_spinbox.setToolTip(
            "Anzahl Wiederholungen je Achse. Der Median über mehrere "
            "Messungen ist robuster gegen einzelne Ausreißer (Backlash, "
            "Bleich-Schwankungen) als eine Einzelmessung."
        )
        form.addRow("Wiederholungen je Achse:", self.cal_repeats_spinbox)
        layout.addLayout(form)

        self.cal_laser_flash_checkbox = QCheckBox("Laser nur bei Aufnahme aktivieren (gegen Bleichen)")
        self.cal_laser_flash_checkbox.setChecked(True)
        self.cal_laser_flash_checkbox.setToolTip(
            "Schaltet den Laser nur für den kurzen Moment jeder Bildaufnahme "
            "ein statt durchgehend - reduziert kumulatives Ausbleichen des "
            "Fluorophors zwischen den zu vergleichenden Bildern."
        )
        layout.addWidget(self.cal_laser_flash_checkbox)

        self.cal_start_button = QPushButton("Kalibrieren")
        self.cal_start_button.setToolTip(
            "Fährt die Stage testweise in X und Y, um aus der beobachteten "
            "Bildverschiebung die Umrechnung Pixel <-> µm zu bestimmen. "
            "Am besten an einer Probe mit sichtbarer Struktur durchführen "
            "(nicht an einer leeren/strukturlosen Fläche)."
        )
        self.cal_start_button.clicked.connect(self._on_calibration_start)
        layout.addWidget(self.cal_start_button)

        self.cal_status_label = QLabel(self._format_calibration_status(self.calibration_result))
        self.cal_status_label.setWordWrap(True)
        layout.addWidget(self.cal_status_label)

        return group

    @staticmethod
    def _format_calibration_status(result: Optional[CalibrationResult]) -> str:
        """Formatiert eine CalibrationResult (oder None) als kurzen Statustext."""
        if result is None:
            return "Keine Kalibrierung vorhanden - bitte 'Kalibrieren' ausführen."
        um_x, um_y = result.um_per_pixel_estimate()
        rot = result.rotation_deg_estimate()
        return (
            f"Kalibriert am {result.timestamp} (Schritt {result.step_um:.1f} µm)\n"
            f"~{um_x:.4f} µm/px (X), ~{um_y:.4f} µm/px (Y), "
            f"Achsverdrehung ~{rot:.1f}°"
        )

    def _build_analysis_group(self) -> QGroupBox:
        """Baut den Bereich für die automatische Event-Analyse."""
        group = QGroupBox("Zellteilung analysieren")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.ea_zrange_spinbox = QDoubleSpinBox()
        self.ea_zrange_spinbox.setRange(0.001, 1.0)
        self.ea_zrange_spinbox.setDecimals(3)
        self.ea_zrange_spinbox.setValue(0.010)
        self.ea_zrange_spinbox.setToolTip(
            "Standard: ±10µm, deckt eine abgerundete mitotische NRK49F-"
            "Zelle (typ. 15-20µm Höhe) mit Marge ab. Setzt voraus, dass "
            "steps_per_mm für die Z-Achse per Messung kalibriert wurde "
            "(Gewindestange, siehe Konversation) - sonst stimmen die "
            "mm-Angaben nicht mit der Realität überein."
        )
        form.addRow("Z-Stack Bereich +/- (mm):", self.ea_zrange_spinbox)

        self.ea_zstep_spinbox = QDoubleSpinBox()
        self.ea_zstep_spinbox.setRange(0.0005, 0.5)
        self.ea_zstep_spinbox.setDecimals(4)
        self.ea_zstep_spinbox.setValue(0.002)
        form.addRow("Z-Stack Schrittweite (mm):", self.ea_zstep_spinbox)

        self.ea_roi_margin_spinbox = QSpinBox()
        self.ea_roi_margin_spinbox.setRange(0, 2000)
        self.ea_roi_margin_spinbox.setValue(50)
        self.ea_roi_margin_spinbox.setSuffix(" px")
        form.addRow("ROI-Rand:", self.ea_roi_margin_spinbox)
        layout.addLayout(form)

        self.ea_hardware_roi_checkbox = QCheckBox("Hardware-ROI nutzen (Ausschnitt statt volles Bild)")
        self.ea_hardware_roi_checkbox.setChecked(False)
        self.ea_hardware_roi_checkbox.setToolTip(
            "Kamera liest nach einem Treffer nur noch den erkannten "
            "Bereich (+ ROI-Rand) aus, statt des vollen Sensors - kein "
            "Zentrieren/Kalibrieren nötig, da Box- und ROI-Koordinaten "
            "im selben Pixel-System liegen. Aus lassen, um beim Testen "
            "immer das volle Bild zu sehen. Wird nach der Aufnahmeserie "
            "in jedem Fall automatisch zurückgesetzt."
        )
        layout.addWidget(self.ea_hardware_roi_checkbox)

        self.ea_save_no_hit_checkbox = QCheckBox("Auch Bild bei \"keine Zellteilung\" speichern")
        self.ea_save_no_hit_checkbox.setChecked(False)
        self.ea_save_no_hit_checkbox.setToolTip(
            "Speichert auch bei einem Nicht-Treffer das aufgenommene "
            "Referenzbild (roh + unveränderte Kopie), statt nur die "
            "Metadaten-Datei zu schreiben. Wichtig für einen "
            "systematischen Sweep, bei dem später ausgezählt werden soll, "
            "wie viele echte Events erkannt wurden. Treffer und "
            "Nicht-Treffer teilen sich einen gemeinsamen, fortlaufenden "
            "Zähler, damit die zeitliche Reihenfolge beim Sortieren nach "
            "Dateiname erhalten bleibt (z.B. 0002_event vor 0003_noevent)."
        )
        layout.addWidget(self.ea_save_no_hit_checkbox)

        self.ea_start_button = QPushButton("Analysieren")
        self.ea_start_button.setToolTip(
            "Prüft das aktuelle Sichtfeld auf Zellteilung. Bei Treffern: "
            "Z-Stack + Aufnahmen bei 50.000/100.000/200.000/300.000µs "
            "Belichtung, alles an der aktuellen Position - kein "
            "automatisches Anfahren/Zentrieren. Boxen werden fest ins "
            "Bild eingezeichnet."
        )
        self.ea_start_button.clicked.connect(self._on_analysis_start)
        layout.addWidget(self.ea_start_button)

        self.ea_status_label = QLabel("")
        self.ea_status_label.setWordWrap(True)
        layout.addWidget(self.ea_status_label)

        return group

    def _build_script_group(self) -> QGroupBox:
        """Baut den Bereich zum Ausführen eigener Python-Skripte."""
        group = QGroupBox("Eigenes Skript ausführen")
        layout = QVBoxLayout(group)

        layout.addWidget(
            QLabel(
                "Verfügbare Funktionen: jog(axis, dist_mm), set_laser(active, percent), "
                "home_all(), get_status(), get_camera_frame(), sharpness_score(img), "
                "run_autofocus(range_mm, step_mm)"
            )
        )

        self.script_editor = QPlainTextEdit()
        self.script_editor.setPlainText(
            "# Beispiel: 5x Bild aufnehmen und Schärfe ausgeben\n"
            "for i in range(5):\n"
            "    img = get_camera_frame()\n"
            "    if img is not None:\n"
            "        print(f\"Bild {i}: Schärfe = {sharpness_score(img):.2f}\")\n"
            "    jog(\"Z\", 0.01)\n"
        )
        self.script_editor.setMaximumHeight(150)
        layout.addWidget(self.script_editor)

        run_button = QPushButton("Skript ausführen")
        run_button.clicked.connect(self._on_run_script)
        layout.addWidget(run_button)

        self.script_output = QPlainTextEdit()
        self.script_output.setReadOnly(True)
        self.script_output.setMaximumHeight(150)
        layout.addWidget(self.script_output)

        return group

    def _toggle_dark_mode(self, checked: bool) -> None:
        """Schaltet zwischen Standard- und einfachem Dark-Mode-Stylesheet um."""
        if checked:
            self.setStyleSheet(DARK_STYLESHEET)
            self.dark_mode_button.setText("Light Mode")
        else:
            self.setStyleSheet("")
            self.dark_mode_button.setText("Dark Mode")
        # Einstellung merken, damit die nächste Session damit startet.
        self.app_settings.setValue("dark_mode", checked)
        # Stylesheet kann Innenabstände/Ränder leicht verändern - Panelhöhen
        # nach dem Wechsel neu fixieren.
        QTimer.singleShot(0, self._lock_side_panel_heights)

    # -----------------------------------------------------------------
    # Kamera-Callbacks
    # -----------------------------------------------------------------

    def _on_camera_toggle(self, checked: bool) -> None:
        """Verbindet/trennt die Kamera je nach Toggle-Zustand."""
        if checked:
            self.camera.connect()
        else:
            self.camera.disconnect()
        self._update_camera_status()

    def _update_camera_status(self) -> None:
        """Aktualisiert Statustext und Button-Beschriftung für die Kamera."""
        if self.camera.is_connected:
            name = self.camera.get_camera_name()
            self.camera_status_label.setText(f"Verbunden: {name}" if name else "Verbunden")
            self.camera_connect_button.setText("Trennen")
        else:
            error_text = f" ({self.camera.last_error})" if self.camera.last_error else ""
            self.camera_status_label.setText(f"Nicht verbunden{error_text}")
            self.camera_connect_button.setText("Verbinden")
            self.camera_connect_button.setChecked(False)

    def _update_live_view(self) -> None:
        """Wird periodisch vom QTimer aufgerufen: holt und zeigt das aktuelle Bild.

        Bedient nebenbei auch anstehende Bild-Anfragen von Worker-Threads
        (Autofokus, Kalibrierung) über die CameraBridge - deren
        get_frame()-Aufrufe müssen zwingend hier im GUI-Thread landen,
        siehe CameraBridge-Docstring.
        """
        self.camera_bridge.poll_pending()

        if not self.camera.is_connected:
            return

        img = self.camera.get_frame()
        if img is None:
            if not self.camera.is_connected:
                # get_frame() hat die Verbindung als tot erkannt (z.B. nach
                # Sleep/Wakeup) und automatisch getrennt - Status sofort
                # sichtbar machen, statt nur das letzte Bild stehen zu lassen.
                self._update_camera_status()
            return

        self.last_image = img
        pixmap = numpy_to_qpixmap(img)
        scaled = pixmap.scaled(
            self.live_view_label.width(),
            self.live_view_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.live_view_label.setPixmap(scaled)
        self.sharpness_label.setText(f"Schärfe-Score: {sharpness_score(img):.2f}")

    def _get_selected_extensions(self) -> tuple[str, ...]:
        """Liest die PNG/TIF/Beide-Auswahl aus - gemeinsam genutzt von
        manuellem Speichern UND der automatischen Analyse, damit beide
        garantiert dasselbe Format-Verhalten haben (nicht zweimal
        getrennt implementieren - das war der Bug von eben)."""
        if self.format_radio_both.isChecked():
            return ("png", "tif")
        if self.format_radio_tif.isChecked():
            return ("tif",)
        return ("png",)

    def _on_save_image(self) -> None:
        """Speichert das zuletzt angezeigte Bild (PNG/TIF/beide) und
        protokolliert die Koordinaten."""
        if self.last_image is None:
            QMessageBox.warning(self, "Kein Bild", "Es liegt aktuell kein Bild vor.")
            return

        extensions = self._get_selected_extensions()

        # Basis-Zeitstempel EINMAL erzeugen, damit z.B. bei "Beide" PNG und
        # TIF garantiert denselben Namen tragen (nur andere Endung) - nicht
        # zweimal timestamped_filename() aufrufen, da das zwei leicht
        # unterschiedliche Zeitstempel liefern könnte.
        base_filename = timestamped_filename(extension=IMAGE_EXTENSION)
        stem = Path(base_filename).stem

        saved_paths = []
        for ext in extensions:
            filename = f"{stem}.{ext}"
            path = save_image(self.last_image, IMAGE_DIR, filename)
            saved_paths.append(path)
            # Koordinaten pro gespeicherter Datei loggen (bei "Beide" also
            # zwei Zeilen - eine je Datei, damit jede Bilddatei in der
            # Tabelle nachvollziehbar ist).
            self._log_coordinates(filename)

        paths_text = "\n".join(str(p) for p in saved_paths)
        QMessageBox.information(self, "Gespeichert", f"Bild gespeichert als:\n{paths_text}")

    def _on_reset_coordinates(self) -> None:
        """Setzt die aktuelle Stage-Position als neuen (0,0,0)-Bezugspunkt.

        Wichtig: Das ist rein softwareseitiges Tracking auf Basis der von
        FluidNC gemeldeten Maschinen-Position (MPos). Es aendert NICHTS an
        der tatsächlichen G-Code-Koordinate auf dem Board (kein G92).
        """
        if not self.stage.is_connected:
            QMessageBox.warning(self, "Nicht verbunden", "Stage ist nicht verbunden.")
            return
        self.coord_zero_offset = self.stage.get_position()
        zx, zy, zz = self.coord_zero_offset
        self.coord_zero_status_label.setText(
            f"Nullpunkt gesetzt (Maschinen-Position X{zx:.3f} Y{zy:.3f} Z{zz:.3f})"
        )

    def _get_logical_position_um(self) -> Optional[Tuple[float, float, float]]:
        """Aktuelle Position relativ zum gesetzten Nullpunkt, in µm.

        Fragt jedes Mal die echte Maschinen-Position (MPos) ab, statt
        Bewegungen selbst aufzuaddieren - dadurch bleibt der Wert auch dann
        korrekt, wenn zwischendurch manuell gejogt wurde. Er wird ABER
        falsch, wenn die Stage Schritte verliert (Motor-Stall) - das kann
        FluidNC ohne Endschalter nicht erkennen. Siehe Thesis-Limitations.
        """
        if self.coord_zero_offset is None or not self.stage.is_connected:
            return None
        mx, my, mz = self.stage.get_position()
        zx, zy, zz = self.coord_zero_offset
        return (mm_to_um(mx - zx), mm_to_um(my - zy), mm_to_um(mz - zz))

    def _log_coordinates(self, filename: str) -> None:
        """Haengt einen Eintrag (Zeitstempel, Dateiname, Koordinaten) an die
        CSV-Tabelle an (Semikolon-getrennt, oeffnet direkt in Excel/DE-Locale).
        """
        pos_um = self._get_logical_position_um()
        if pos_um is None:
            x_um = y_um = z_um = ""
            note = "kein Nullpunkt gesetzt"
        else:
            x_um, y_um, z_um = (round(v, 2) for v in pos_um)
            note = ""

        file_exists = self.coord_log_path.exists()
        self.coord_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.coord_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                if not file_exists:
                    writer.writerow(
                        ["Zeitstempel", "Dateiname", "X (µm)", "Y (µm)", "Z (µm)", "Hinweis"]
                    )
                writer.writerow(
                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, x_um, y_um, z_um, note]
                )
        except PermissionError:
            # Haeufigste Ursache: Datei ist gerade in Excel o.ae. geoeffnet
            # und dadurch unter Windows exklusiv gesperrt. Bild ist zu diesem
            # Zeitpunkt bereits gespeichert - nur das Logging schlaegt fehl.
            QMessageBox.warning(
                self, "Koordinaten-Log gesperrt",
                f"Konnte den Eintrag nicht in\n{self.coord_log_path}\nschreiben "
                "(Datei ist vermutlich gerade in Excel o.ä. geöffnet). "
                "Das Bild wurde trotzdem gespeichert.",
            )
        except OSError as exc:
            QMessageBox.warning(
                self, "Koordinaten-Log fehlgeschlagen",
                f"Konnte den Eintrag nicht schreiben: {exc}\n"
                "Das Bild wurde trotzdem gespeichert.",
            )

    def _on_exposure_slider_changed(self, value: int) -> None:
        """Übernimmt einen neuen Slider-Wert für die Belichtungszeit."""
        if abs(self.exposure_spinbox.value() - value) > 0.5:
            self.exposure_spinbox.blockSignals(True)
            self.exposure_spinbox.setValue(float(value))
            self.exposure_spinbox.blockSignals(False)
        self.camera.set_exposure(float(value))

    def _on_exposure_spinbox_changed(self, value: float) -> None:
        """Übernimmt einen neuen Spinbox-Wert für die Belichtungszeit."""
        if abs(self.exposure_slider.value() - value) > 0.5:
            self.exposure_slider.blockSignals(True)
            self.exposure_slider.setValue(int(value))
            self.exposure_slider.blockSignals(False)
        self.camera.set_exposure(value)

    def _on_gain_slider_changed(self, value: int) -> None:
        """Übernimmt einen neuen Slider-Wert für den Gain."""
        if abs(self.gain_spinbox.value() - value) > 0.05:
            self.gain_spinbox.blockSignals(True)
            self.gain_spinbox.setValue(float(value))
            self.gain_spinbox.blockSignals(False)
        self.camera.set_gain(float(value))

    def _on_gain_spinbox_changed(self, value: float) -> None:
        """Übernimmt einen neuen Spinbox-Wert für den Gain."""
        if abs(self.gain_slider.value() - value) > 0.05:
            self.gain_slider.blockSignals(True)
            self.gain_slider.setValue(int(value))
            self.gain_slider.blockSignals(False)
        self.camera.set_gain(value)

    # -----------------------------------------------------------------
    # Stage-Callbacks
    # -----------------------------------------------------------------

    def _refresh_ports(self) -> None:
        """Aktualisiert die Liste verfügbarer serieller Ports im Dropdown."""
        current = self.port_combo.currentText() if self.port_combo.count() else None
        self.port_combo.clear()
        ports = StageController.available_ports()
        self.port_combo.addItems(ports)
        if current and current in ports:
            self.port_combo.setCurrentText(current)

    # Wie lange auf eine Verbindungsantwort gewartet wird, bevor die GUI
    # aufgibt und "Keine Verbindung möglich" meldet.
    # stage.py braucht im Normalfall bis zu ~5s (2s Boot-Wartezeit des
    # Boards + bis zu 3s Warten auf eine Antwort, siehe
    # CONNECT_RESPONSE_TIMEOUT_S in stage.py) - hier bewusst mit Marge
    # etwas darüber, damit die GUI stage.py nicht vorzeitig "überholt".
    STAGE_CONNECT_TIMEOUT_MS = 7000

    def _on_stage_toggle(self, checked: bool) -> None:
        """Verbindet/trennt die Stage über den gewählten Port (nicht-blockierend)."""
        if checked:
            # Verbindung läuft bereits -> nichts tun
            if self.stage_connect_worker is not None and self.stage_connect_worker.isRunning():
                return

            selected_port = self.port_combo.currentText()

            # Während des Verbindens: Button/Dropdown sperren, damit nicht doppelt
            # geklickt werden kann, und Status anzeigen.
            self.stage_connect_button.setEnabled(False)
            self.port_combo.setEnabled(False)
            self.stage_status_label.setText("Verbinde...")
            self._stage_connect_timed_out = False

            self.stage_connect_worker = StageConnectWorker(self.stage, selected_port)
            self.stage_connect_worker.finished_connect.connect(self._on_stage_connect_finished)
            self.stage_connect_worker.start()

            # Timeout: Wenn nach STAGE_CONNECT_TIMEOUT_MS keine Rückmeldung kam,
            # gibt die GUI auf und zeigt eine Meldung, statt endlos zu warten.
            self.stage_connect_timeout_timer = QTimer(self)
            self.stage_connect_timeout_timer.setSingleShot(True)
            self.stage_connect_timeout_timer.timeout.connect(self._on_stage_connect_timeout)
            self.stage_connect_timeout_timer.start(self.STAGE_CONNECT_TIMEOUT_MS)
        else:
            self.stage.disconnect()
            self._update_stage_status()

    def _on_stage_connect_timeout(self) -> None:
        """Wird aufgerufen, wenn der Verbindungsversuch nach STAGE_CONNECT_TIMEOUT_MS
        immer noch nicht zurückgekehrt ist. Die GUI gibt auf und meldet das dem
        Nutzer, auch wenn der Hintergrund-Thread technisch noch weiterläuft."""
        if self.stage_connect_worker is None or not self.stage_connect_worker.isRunning():
            return  # ist inzwischen schon fertig geworden, nichts zu tun

        self._stage_connect_timed_out = True

        self.stage_connect_button.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.stage_connect_button.blockSignals(True)
        self.stage_connect_button.setChecked(False)
        self.stage_connect_button.blockSignals(False)
        self.stage_status_label.setText("Keine Verbindung möglich (Zeitüberschreitung)")

        QMessageBox.warning(
            self,
            "Keine Verbindung",
            "Es konnte keine Verbindung zur Stage hergestellt werden "
            f"(keine Rückmeldung innerhalb von {self.STAGE_CONNECT_TIMEOUT_MS // 1000} Sekunden).",
        )

    def _on_stage_connect_finished(self, success: bool, error_text: str) -> None:
        """Wird aufgerufen, sobald der StageConnectWorker fertig ist (Erfolg oder Fehler)."""
        if self.stage_connect_timeout_timer is not None:
            self.stage_connect_timeout_timer.stop()
            self.stage_connect_timeout_timer = None

        if self._stage_connect_timed_out:
            # Die GUI hat bereits per Timeout aufgegeben und den Nutzer
            # informiert. Jetzt nur noch den tatsächlichen Zustand nachträglich
            # angleichen (z.B. falls die Verbindung doch noch zustande kam),
            # ohne erneut eine Meldung zu zeigen.
            self._stage_connect_timed_out = False
            self.stage_connect_button.blockSignals(True)
            self.stage_connect_button.setChecked(self.stage.is_connected)
            self.stage_connect_button.blockSignals(False)
            self._update_stage_status()
            return

        self.stage_connect_button.setEnabled(True)
        self.port_combo.setEnabled(True)

        if not success:
            # Button zurücksetzen, ohne erneut _on_stage_toggle über das
            # toggled-Signal auszulösen (würde sonst versuchen, "abzumelden").
            self.stage_connect_button.blockSignals(True)
            self.stage_connect_button.setChecked(False)
            self.stage_connect_button.blockSignals(False)
            QMessageBox.warning(
                self,
                "Keine Verbindung",
                f"Es konnte keine Verbindung zur Stage hergestellt werden.\n{error_text}".rstrip(),
            )

        self._update_stage_status()

    def _update_stage_status(self) -> None:
        """Aktualisiert Statustext und Button-Beschriftung für die Stage."""
        if self.stage.is_connected:
            self.stage_status_label.setText(f"Verbunden auf {self.stage.port}")
            self.stage_connect_button.setText("Trennen")
        else:
            error_text = f": {self.stage.last_error}" if self.stage.last_error else ""
            self.stage_status_label.setText(f"Nicht verbunden{error_text}")
            self.stage_connect_button.setText("Verbinden")
            self.stage_connect_button.setChecked(False)

    def _on_jog(self, axis: str, negative: bool) -> None:
        """Führt eine Jog-Bewegung mit der aktuell gewählten Schrittweite aus."""
        step_um = self.step_size_um[axis]
        distance_mm = um_to_mm(step_um)
        if negative:
            distance_mm = -distance_mm
        self.stage.jog(axis, distance_mm)

    def keyPressEvent(self, event) -> None:
        """WASD steuert X/Y mit der aktuell gewählten Schrittweite
        (dieselbe wie bei den +/- Buttons).

        Bewusst WASD statt Pfeiltasten: Pfeiltasten werden von Qt intern
        zur Fokus-Navigation zwischen Buttons/Radio-Buttons verwendet und
        von diesen Widgets bereits abgefangen (event.accept()), bevor sie
        hier ankommen - WASD hat diese Sonderbedeutung nicht.

        Ausnahme: Hat gerade ein Eingabefeld (z.B. eine Positions-Spinbox)
        den Fokus, wird nicht abgefangen, sondern normal durchgereicht.

        event.isAutoRepeat() wird ignoriert, damit gehaltene Tasten nicht
        eine Flut von Jog-Befehlen auslösen (ein Tastendruck = ein Jog,
        wie bei einem Klick auf den +/- Button).
        """
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QAbstractSpinBox, QLineEdit)):
            super().keyPressEvent(event)
            return

        key_axis_map = {
            Qt.Key_W: ("X", False),
            Qt.Key_S: ("X", True),
            Qt.Key_A: ("Y", True),
            Qt.Key_D: ("Y", False),
            Qt.Key_Q: ("Z", False),
            Qt.Key_E: ("Z", True),
        }
        mapping = key_axis_map.get(event.key())
        if mapping is not None and not event.isAutoRepeat():
            axis, negative = mapping
            self._on_jog(axis, negative)
            event.accept()
            return

        # Belichtungszeit-Presets: 1/2/3 -> 100000/200000/300000 µs.
        # Ueber die Spinbox gesetzt (nicht camera.set_exposure() direkt),
        # damit die Anzeige automatisch mitgeht (bestehende Verdrahtung
        # via _on_exposure_spinbox_changed).
        exposure_presets_us = {
            Qt.Key_1: 100_000.0,
            Qt.Key_2: 200_000.0,
            Qt.Key_3: 300_000.0,
        }
        exposure_us = exposure_presets_us.get(event.key())
        if exposure_us is not None and not event.isAutoRepeat():
            self.exposure_spinbox.setValue(exposure_us)
            event.accept()
            return

        # Bild aufnehmen: "C" statt Enter - Enter/Return wird von Qt oft
        # vom gerade fokussierten Button/Spinbox selbst verarbeitet (z.B.
        # bestaetigt es dort das Feld) und kaeme inkonsistent bei uns an.
        if event.key() == Qt.Key_C and not event.isAutoRepeat():
            self._on_save_image()
            event.accept()
            return

        super().keyPressEvent(event)

    def _on_step_size_changed(self, axis: str, option_um: int, checked: bool) -> None:
        """Übernimmt eine neu gewählte Schrittweite für eine Achse."""
        if checked:
            self.step_size_um[axis] = float(option_um)

    def _on_custom_step_size_toggled(self, axis: str, checked: bool) -> None:
        """Wird aktiviert, wenn "Custom" für eine Achse ausgewählt wird -
        übernimmt den aktuellen Wert aus dem zugehörigen Eingabefeld."""
        if checked:
            self.step_size_um[axis] = self.axis_custom_step_spinboxes[axis].value()

    def _on_custom_step_size_value_changed(self, axis: str, value: float) -> None:
        """Wird bei jeder Änderung des Custom-Eingabefelds aufgerufen - nur
        wirksam, wenn "Custom" für diese Achse auch tatsächlich ausgewählt
        ist (sonst tippt man ins Feld, ohne dass es etwas bewirkt)."""
        if self.axis_step_radios[axis][-1].isChecked():
            self.step_size_um[axis] = value

    def _on_move_to_confirmed(self) -> None:
        """Fährt X/Y/Z gemeinsam zur eingegebenen Zielposition (µm -> mm)."""
        x_um = self.position_spinboxes["X"].value()
        y_um = self.position_spinboxes["Y"].value()
        z_um = self.position_spinboxes["Z"].value()
        self.stage.move_to(
            x=um_to_mm(x_um),
            y=um_to_mm(y_um),
            z=um_to_mm(z_um),
        )

    def _on_global_speed_changed(self, value: int) -> None:
        """Übernimmt eine neue globale Vorschubgeschwindigkeit."""
        self.stage.jog_feedrate = value

    # -----------------------------------------------------------------
    # Laser-Callbacks
    # -----------------------------------------------------------------

    def _on_laser_toggle(self, checked: bool) -> None:
        """Schaltet den Laser über den Toggle-Button an oder aus."""
        percent = self.laser_slider.value()
        self.laser.set_laser(checked, percent)

    def _on_laser_slider_changed(self, value: int) -> None:
        """Übernimmt einen neuen Slider-Wert für die Laserintensität."""
        if self.laser_spinbox.value() != value:
            self.laser_spinbox.blockSignals(True)
            self.laser_spinbox.setValue(value)
            self.laser_spinbox.blockSignals(False)
        self._apply_laser_power(value)

    def _on_laser_spinbox_changed(self, value: int) -> None:
        """Übernimmt einen neuen Spinbox-Wert für die Laserintensität."""
        if self.laser_slider.value() != value:
            self.laser_slider.blockSignals(True)
            self.laser_slider.setValue(value)
            self.laser_slider.blockSignals(False)
        self._apply_laser_power(value)

    def _apply_laser_power(self, percent: int) -> None:
        """Wendet eine neue Intensität an - sendet nur G-Code, wenn der Laser aktiv ist."""
        if self.laser.is_on:
            self.laser.set_laser(True, percent)
        else:
            self.laser.set_power(percent)

    # -----------------------------------------------------------------
    # Autofokus-Callbacks
    # -----------------------------------------------------------------

    def _on_autofocus_start(self) -> None:
        """Startet den Autofokus-Sweep in einem separaten Thread."""
        if self.autofocus_worker is not None and self.autofocus_worker.isRunning():
            return

        range_mm = self.af_range_spinbox.value()
        step_mm = self.af_step_spinbox.value()

        self.af_start_button.setEnabled(False)
        self.af_progress_bar.setValue(0)
        self.af_status_label.setText("Läuft...")

        self.autofocus_worker = AutofocusWorker(
            self.stage, ThreadSafeCameraProxy(self.camera_bridge), range_mm, step_mm
        )
        self.autofocus_worker.progress.connect(self._on_autofocus_progress)
        self.autofocus_worker.finished_with_result.connect(self._on_autofocus_finished)
        self.autofocus_worker.start()

    def _on_autofocus_progress(self, i: int, n: int, offset: float, score: float) -> None:
        """Aktualisiert Fortschrittsbalken und Statustext während des Sweeps."""
        self.af_progress_bar.setMaximum(n)
        self.af_progress_bar.setValue(i)
        self.af_status_label.setText(f"[{i}/{n}] Offset {offset:+.3f} mm -> Schärfe {score:.2f}")

    def _on_autofocus_finished(self, result: Tuple[Optional[float], Any]) -> None:
        """Wird nach Abschluss des Sweeps aufgerufen."""
        best_offset, _results = result
        self.af_start_button.setEnabled(True)
        if best_offset is not None:
            self.af_status_label.setText(f"Bester Fokus bei Offset {best_offset:+.3f} mm")
        else:
            self.af_status_label.setText("Autofokus fehlgeschlagen - keine Bilder erhalten")

    # -----------------------------------------------------------------
    # Kalibrierungs-Callbacks
    # -----------------------------------------------------------------

    def _on_calibration_start(self) -> None:
        """Startet die Pixel-zu-Stage-Kalibrierung in einem Worker-Thread."""
        if not self.stage.is_connected:
            QMessageBox.warning(self, "Nicht verbunden", "Stage ist nicht verbunden.")
            return
        if not self.camera.is_connected:
            QMessageBox.warning(self, "Nicht verbunden", "Kamera ist nicht verbunden.")
            return

        step_um = self.cal_step_spinbox.value()
        n_repeats = self.cal_repeats_spinbox.value()
        laser_for_flash = self.laser if self.cal_laser_flash_checkbox.isChecked() else None
        laser_percent = self.laser_spinbox.value()

        # Falls der Laser im Blitz-Modus zwischendurch aus-/angeschaltet
        # wird: Zustand von VOR der Kalibrierung merken, um ihn danach
        # wiederherzustellen (sonst zeigt der "Laser aktiv"-Button einen
        # Zustand, der nicht mehr zur echten Hardware passt).
        self._cal_prior_laser_on = self.laser.is_on if laser_for_flash is not None else None

        self.cal_start_button.setEnabled(False)
        self.cal_status_label.setText("Kalibrierung läuft...")

        self.calibration_worker = CalibrationWorker(
            self.stage,
            ThreadSafeCameraProxy(self.camera_bridge),
            step_um,
            n_repeats=n_repeats,
            laser=laser_for_flash,
            laser_percent=laser_percent,
        )
        self.calibration_worker.progress.connect(self._on_calibration_progress)
        self.calibration_worker.finished_with_result.connect(self._on_calibration_finished)
        self.calibration_worker.start()

    def _on_calibration_progress(self, msg: str) -> None:
        self.cal_status_label.setText(msg)

    def _on_calibration_finished(self, result: object) -> None:
        """Wird nach Abschluss der Kalibrierung aufgerufen.

        result ist entweder ein CalibrationResult (Erfolg) oder eine
        Exception (Fehlschlag - z.B. Verbindung waehrend der Kalibrierung
        verloren, oder kein Bild aufnehmbar).
        """
        self.cal_start_button.setEnabled(True)

        # Laser-Zustand von vor der Kalibrierung wiederherstellen, falls im
        # Blitz-Modus kalibriert wurde - unabhaengig davon, ob die
        # Kalibrierung erfolgreich war oder fehlgeschlagen ist. Hardware
        # wird explizit gesetzt (nicht nur der Button), da setChecked()
        # kein Signal ausloest, wenn sich der angezeigte Zustand nicht
        # aendert - der Button wurde waehrend der Kalibrierung ja nie
        # umgeschaltet, nur die Hardware direkt.
        if self._cal_prior_laser_on is not None:
            prior_on = self._cal_prior_laser_on
            self._cal_prior_laser_on = None
            if prior_on:
                self.laser.set_laser(True, self.laser_spinbox.value())
            else:
                self.laser.set_laser(False)
            self.laser_toggle_button.blockSignals(True)
            self.laser_toggle_button.setChecked(prior_on)
            self.laser_toggle_button.blockSignals(False)

        if isinstance(result, Exception):
            self.cal_status_label.setText(
                f"Kalibrierung fehlgeschlagen: {result}\n"
                f"{self._format_calibration_status(self.calibration_result)}"
            )
            return

        self.calibration_result = result
        save_calibration(result)
        self.cal_status_label.setText(self._format_calibration_status(result))

    # -----------------------------------------------------------------
    # Event-Analyse-Callbacks
    # -----------------------------------------------------------------

    def _on_analysis_start(self) -> None:
        """Startet die Event-Analyse (+ ggf. Aufnahmeserie) in einem Worker-Thread."""
        if not self.stage.is_connected:
            QMessageBox.warning(self, "Nicht verbunden", "Stage ist nicht verbunden.")
            return
        if not self.camera.is_connected:
            QMessageBox.warning(self, "Nicht verbunden", "Kamera ist nicht verbunden.")
            return

        z_range = self.ea_zrange_spinbox.value()
        z_step = self.ea_zstep_spinbox.value()
        # Dieselben Presets wie die Tasten 1/2/3 (siehe keyPressEvent) -
        # bewusst dieselben Werte, damit "Analysieren" dieselben
        # Belichtungszeiten liefert, die man auch manuell gewohnt ist.
        exposure_presets_us = (50_000.0, 100_000.0, 200_000.0, 300_000.0)

        custom_name = self.event_name_input.text().strip()
        name_prefix = custom_name or None
        extensions = self._get_selected_extensions()
        use_hardware_roi = self.ea_hardware_roi_checkbox.isChecked()
        roi_margin_px = float(self.ea_roi_margin_spinbox.value())
        save_no_hit_images = self.ea_save_no_hit_checkbox.isChecked()

        self.ea_start_button.setEnabled(False)
        self.ea_status_label.setText("Analyse läuft...")

        self.analysis_worker = EventAnalysisWorker(
            self.stage,
            ThreadSafeCameraProxy(self.camera_bridge),
            str(IMAGE_DIR),
            z_stack_range_mm=z_range,
            z_stack_step_mm=z_step,
            exposure_presets_us=exposure_presets_us,
            laser=self.laser,
            name_prefix=name_prefix,
            run_index=self.event_counter,
            extensions=extensions,
            use_hardware_roi=use_hardware_roi,
            roi_margin_px=roi_margin_px,
            save_no_hit_images=save_no_hit_images,
        )
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.finished_with_result.connect(self._on_analysis_finished)
        self.analysis_worker.start()

    def _on_analysis_progress(self, msg: str) -> None:
        self.ea_status_label.setText(msg)

    def _on_analysis_finished(self, result: object) -> None:
        """Wird nach Abschluss der Analyse (+ ggf. Aufnahmeserie) aufgerufen.

        result ist entweder ein AnalysisResult oder eine Exception
        (Fehlschlag - z.B. Verbindung waehrend der Serie verloren).
        """
        self.ea_start_button.setEnabled(True)

        if isinstance(result, Exception):
            self.ea_status_label.setText(f"Analyse fehlgeschlagen: {result}")
            return

        if not result.found:
            self.ea_status_label.setText(
                f"Keine Zellteilung erkannt.\nMetadaten: {result.metadata_path}"
            )
            # Zaehler auch bei Nicht-Treffern hochzaehlen, aber NUR wenn
            # dafuer tatsaechlich ein Bild gespeichert wurde (Toggle an) -
            # so bleibt die Nummerierung luecklos zu dem, was wirklich auf
            # der Platte liegt, waehrend sie bei ausgeschaltetem Toggle wie
            # bisher nur echte Treffer zaehlt.
            if self.ea_save_no_hit_checkbox.isChecked():
                self.event_counter += 1
                for frame in result.saved_frames:
                    self._log_coordinates(frame.filename_raw)
                    self._log_coordinates(frame.filename_annotated)
            return

        self.ea_status_label.setText(
            f"{len(result.detections)} Event(s) erkannt, "
            f"{len(result.saved_frames)} Bild-Paare (roh + Boxen) gespeichert.\n"
            f"Metadaten: {result.metadata_path}"
        )
        # Gemeinsamer Zaehler mit dem Nicht-Treffer-Fall oben - siehe
        # Docstring von run_event_analysis (name_prefix/run_index) fuer
        # die Begruendung (zeitliche Reihenfolge beim Sortieren).
        self.event_counter += 1
        # Jede gespeicherte Aufnahme der Serie in der bestehenden
        # Koordinaten-Tabelle protokollieren (gleiche Funktion wie beim
        # manuellen "Bild speichern") - Rohbild UND annotierte Version
        # bekommen je eine eigene Zeile, damit beide nachvollziehbar sind.
        for frame in result.saved_frames:
            self._log_coordinates(frame.filename_raw)
            self._log_coordinates(frame.filename_annotated)

    # -----------------------------------------------------------------
    # Skript-Ausführung
    # -----------------------------------------------------------------

    def _on_run_script(self) -> None:
        """Führt den im Editor eingegebenen Python-Code aus."""
        script_code = self.script_editor.toPlainText()
        output_buffer = io.StringIO()

        def run_autofocus(range_mm: float, step_mm: float):
            """Wrapper um autofocus_sweep mit bereits gebundenen Abhängigkeiten."""
            return autofocus_sweep(self.stage, self.camera, range_mm, step_mm)

        exec_globals: Dict[str, Any] = {
            "jog": self.stage.jog,
            "set_laser": self.laser.set_laser,
            "home_all": self.stage.home,
            "get_status": self.stage.status,
            "send_gcode": self.stage.send_gcode,
            "get_camera_frame": self.camera.get_frame,
            "sharpness_score": sharpness_score,
            "run_autofocus": run_autofocus,
            "np": np,
        }

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(script_code, exec_globals)  # noqa: S102 - bewusstes Feature
            self.script_output.setPlainText(output_buffer.getvalue() or "(kein Output)")
        except Exception as exc:  # noqa: BLE001 - Nutzerskript kann beliebig fehlschlagen
            self.script_output.setPlainText(f"Fehler beim Ausführen: {exc}")

    # -----------------------------------------------------------------
    # Sauberes Beenden
    # -----------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt-Methodenname
        """Trennt Kamera und Stage sauber, bevor das Fenster geschlossen wird."""
        self.live_timer.stop()
        if self.autofocus_worker is not None and self.autofocus_worker.isRunning():
            self.autofocus_worker.wait(2000)
        if self.stage_connect_worker is not None and self.stage_connect_worker.isRunning():
            # Nicht endlos warten, falls der Verbindungsversuch hängt (z.B. kein
            # Gerät antwortet) - das Fenster soll trotzdem schließbar bleiben.
            self.stage_connect_worker.wait(500)
        self.camera.disconnect()
        self.stage.disconnect()
        event.accept()


def main() -> None:
    """Startet die Qt-Anwendung."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()