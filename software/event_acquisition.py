"""Analyse-Workflow für den "Analysieren"-Button.

Prüft das aktuelle Sichtfeld auf Zellteilungs-Events (event_detector.py).
Bei einem Treffer läuft direkt an der aktuellen Position eine
Aufnahmeserie: ein Z-Stack (eine repräsentative Belichtungszeit) und
zusätzlich Einzelbilder bei mehreren Belichtungs-Presets auf der
Ausgangs-Fokusebene. Kein automatisches Zentrieren/Anfahren - das erkannte
Sichtfeld wird komplett so dokumentiert, wie es gerade steht.

Optional (save_no_hit_images) wird auch bei einem Nicht-Treffer das
Referenzbild gespeichert - wichtig z.B. für einen systematischen Sweep,
bei dem später ausgezählt werden soll, wie viele echte Events erkannt
wurden. Treffer und Nicht-Treffer teilen sich dafür EINEN gemeinsamen,
fortlaufenden Zähler (statt zwei getrennter), damit "event2" und
"noevent3" beim Sortieren nach Dateiname in der richtigen zeitlichen
Reihenfolge bleiben.

Von jedem Frame werden ZWEI Versionen gespeichert: das unveränderte
Rohbild UND eine Kopie mit fest eingezeichneten Boxen - nie nur die
annotierte Version, damit das Original für die Thesis/spätere Analyse
nicht verloren geht. Beide landen in nach Datum sortierten
Unterordnern (raw/, boxed/, metadata/) statt alles in einen Ordner zu
mischen.

Zu jedem Durchlauf (Treffer oder nicht) wird eine Metadaten-Datei
(JSON) geschrieben: Modell, Schwellenwerte, Boxen, ROI, Belichtung,
Z-Position, hit/no-hit, Zeitstempel, Verlauf der Statusmeldungen.

StageController und CameraManager werden als Abhängigkeiten übergeben,
nicht selbst erzeugt (gleiches Prinzip wie autofocus.py/calibration.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from camera import CameraManager
from event_detector import MODEL_PATH, detect_events
from laser import LaserController
from stage import StageController
from utils import save_image, timestamped_filename

ProgressCallback = Callable[[str], None]

BOX_COLOR_BGR = (156, 93, 214)  # BGR - Lila
BOX_THICKNESS = 2


@dataclass
class SavedFrame:
    """Eine gespeicherte Aufnahme - Rohbild UND annotierte Kopie, nie nur
    eine der beiden. kind ist z.B. "zstack", "exposure" oder "reference"
    (Referenzbild bei einem gespeicherten Nicht-Treffer)."""

    kind: str
    z_offset_mm: float
    exposure_us: Optional[float]
    filename_raw: str
    path_raw: str
    filename_annotated: str
    path_annotated: str


@dataclass
class AnalysisResult:
    detections: List[dict] = field(default_factory=list)
    saved_frames: List[SavedFrame] = field(default_factory=list)
    metadata_path: Optional[str] = None

    @property
    def found(self) -> bool:
        return len(self.detections) > 0


def _draw_boxes(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Zeichnet die erkannten Boxen fest in eine Kopie des Bildes ein.
    Mit einer leeren detections-Liste einfach nur Farbkonvertierung ohne
    Boxen (fuer das Referenzbild eines gespeicherten Nicht-Treffers)."""
    annotated = image.copy()
    if annotated.ndim == 2 or (annotated.ndim == 3 and annotated.shape[2] == 1):
        annotated = cv2.cvtColor(annotated.squeeze(), cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = (int(round(det[k])) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR_BGR, BOX_THICKNESS)
        label_y = y1 - 8 if y1 - 8 > 10 else y1 + 18
        cv2.putText(
            annotated, f"{det['confidence']:.2f}", (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR_BGR, 2,
        )
    return annotated


def run_event_analysis(
    stage: StageController,
    camera: CameraManager,
    image_dir: str,
    conf: float = 0.15,
    iou: float = 0.5,
    z_stack_range_mm: float = 0.010,
    z_stack_step_mm: float = 0.002,
    z_stack_exposure_us: float = 100_000.0,
    exposure_presets_us: tuple[float, ...] = (50_000.0, 100_000.0, 200_000.0, 300_000.0),
    settle_s: float = 0.3,
    laser: Optional[LaserController] = None,
    name_prefix: Optional[str] = None,
    run_index: Optional[int] = None,
    extensions: tuple[str, ...] = ("tif",),
    use_hardware_roi: bool = False,
    roi_margin_px: float = 50.0,
    save_no_hit_images: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    """Analysiert das aktuelle Sichtfeld und nimmt bei einem Treffer eine
    Aufnahmeserie auf - Z-Stack + mehrere Belichtungen, alles an der
    aktuellen Position (kein Zentrieren/Anfahren).

    Args:
        stage: Verbundene StageController-Instanz.
        camera: Verbundene CameraManager-Instanz.
        image_dir: Basis-Zielverzeichnis (wie IMAGE_DIR). Darunter werden
            automatisch <image_dir>/<Datum>/{raw,boxed,metadata}/ angelegt.
        conf, iou: Werden 1:1 an detect_events() durchgereicht.
        z_stack_range_mm, z_stack_step_mm, z_stack_exposure_us,
        exposure_presets_us, settle_s: siehe vorherige Version - unveraendert.
        laser: Optionale LaserController-Instanz - wird waehrend der
            Modell-Auswertung kurz ausgeschaltet und danach in JEDEM Fall
            (Treffer oder nicht) auf den vorherigen Zustand zurueckgesetzt.
        name_prefix: Eigener Basisname (z.B. "HeLa_2026-09-17"), OHNE
            Nummer/Suffix - die werden hier automatisch angehaengt, als
            "_0003_event" bzw. "_0004_noevent" (Zahl VOR dem Label, mit
            fuehrenden Nullen - siehe run_index). Ohne Angabe wird ein
            zeitstempel-basierter Name verwendet.
        run_index: Laufende Nummer fuer den "_NNNN_event"/"_NNNN_noevent"-
            Suffix. WICHTIG: Treffer und Nicht-Treffer sollten sich EINEN
            gemeinsamen, fortlaufenden Zaehler teilen (nicht zwei
            getrennte), UND die Zahl muss vor dem Label stehen - sonst
            sortiert z.B. Windows alphabetisch "event3" vor "noevent4"
            vor "event5" (falsche Reihenfolge), statt nach der echten
            zeitlichen Abfolge. Das Hochzaehlen selbst passiert beim
            Aufrufer (main.py).
        extensions: In welchem/welchen Format(en) jedes Bild gespeichert
            wird, z.B. ("png",), ("tif",) oder ("png", "tif") fuer beide.
        use_hardware_roi, roi_margin_px: siehe vorherige Version -
            unveraendert (nur bei einem Treffer relevant).
        save_no_hit_images: Wenn True, wird auch bei "keine Zellteilung
            erkannt" das aufgenommene Referenzbild gespeichert (roh +
            eine unveraenderte "boxed"-Kopie ohne Boxen, fuer konsistente
            Ordnerstruktur) - wichtig fuer einen systematischen Sweep, bei
            dem spaeter ausgezaehlt werden soll, wie viele echte Events
            erkannt wurden. Ohne dieses Flag wird bei einem Nicht-Treffer
            weiterhin nur die Metadaten-Datei geschrieben, kein Bild.
        progress_callback: Optionale Funktion für Status-Updates.

    Returns:
        AnalysisResult. Bei .found == False wurde ggf. (siehe
        save_no_hit_images) trotzdem ein Bild gespeichert.

    Raises:
        RuntimeError: Wenn Stage oder Kamera nicht verbunden sind, oder ein
            Bild nicht aufgenommen werden konnte.
    """
    if not stage.is_connected:
        raise RuntimeError("Stage ist nicht verbunden")
    if not camera.is_connected:
        raise RuntimeError("Kamera ist nicht verbunden")

    log_lines: List[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        if progress_callback:
            progress_callback(msg)

    def _get_frame_or_raise() -> np.ndarray:
        img = camera.get_frame()
        if img is None:
            raise RuntimeError(f"Konnte kein Bild aufnehmen ({camera.last_error})")
        return img

    now = datetime.now()
    run_timestamp = now.isoformat(timespec="seconds")
    date_str = now.strftime("%Y-%m-%d")

    base_out_dir = Path(image_dir) / date_str
    raw_dir = base_out_dir / "raw"
    boxed_dir = base_out_dir / "boxed"
    metadata_dir = base_out_dir / "metadata"
    for d in (raw_dir, boxed_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _build_stem(hit: bool) -> str:
        if name_prefix:
            label = "event" if hit else "noevent"
            idx = run_index if run_index is not None else 0
            # Zahl VOR dem Label, mit fuehrenden Nullen - sonst sortiert
            # Windows alphabetisch nach dem Label zuerst ("event3" vor
            # "noevent4" vor "event5", statt in echter Reihenfolge) und
            # unaufgefuellte Zahlen sortieren als Text falsch (10 vor 3).
            return f"{name_prefix}_{idx:04d}_{label}"
        prefix = "event" if hit else "noevent"
        return Path(timestamped_filename(prefix=prefix, extension="tif")).stem

    def _write_metadata(stem: str, extra: dict) -> str:
        metadata = {
            "timestamp": run_timestamp,
            "model_path": MODEL_PATH,
            "conf_threshold": conf,
            "iou_threshold": iou,
            **extra,
            "log": log_lines,
        }
        metadata_path = metadata_dir / f"{stem}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(metadata_path)

    _log("Nehme Bild für Analyse auf...")
    reference_img = _get_frame_or_raise()

    # Laser waehrend der Modell-Auswertung aus - dafuer wird kein Bild
    # aufgenommen, und es spart unnoetige Belichtung. In jedem Fall
    # (Treffer oder nicht) danach wieder auf den vorherigen Zustand
    # zuruecksetzen.
    prior_laser_on = laser.is_on if laser is not None else None
    prior_laser_percent = laser.power_percent if laser is not None else None
    if laser is not None:
        laser.set_laser(False)

    try:
        _log("Analysiere auf Zellteilung...")
        detections = detect_events(reference_img, conf=conf, iou=iou)
    finally:
        if laser is not None and prior_laser_on:
            laser.set_laser(True, prior_laser_percent)

    if not detections:
        _log("Keine Zellteilung erkannt.")
        base_stem = _build_stem(hit=False)
        saved_frames: List[SavedFrame] = []

        if save_no_hit_images:
            annotated = _draw_boxes(reference_img, [])
            for ext in extensions:
                raw_filename = f"{base_stem}_raw.{ext}"
                boxed_filename = f"{base_stem}_boxed.{ext}"
                raw_path = save_image(reference_img, str(raw_dir), raw_filename)
                boxed_path = save_image(annotated, str(boxed_dir), boxed_filename)
                saved_frames.append(SavedFrame(
                    "reference", 0.0, None,
                    raw_filename, raw_path, boxed_filename, boxed_path,
                ))

        metadata_path = _write_metadata(base_stem, {
            "hit": False,
            "detections": [],
            "frames": [asdict(f) for f in saved_frames],
        })
        return AnalysisResult(detections=[], saved_frames=saved_frames, metadata_path=metadata_path)

    _log(f"{len(detections)} Treffer gefunden - starte Aufnahmeserie.")
    base_stem = _build_stem(hit=True)

    original_exposure_us = camera.get_exposure()
    saved_frames = []

    # Optional: Hardware-ROI auf die Vereinigung aller gefundenen Boxen
    # (+ Rand) setzen. Box-Koordinaten fuer die spaeter gespeicherten
    # (dann bereits zugeschnittenen) Bilder entsprechend verschieben.
    draw_detections = detections
    roi_active = False
    roi_info: Optional[dict] = None
    if use_hardware_roi and hasattr(camera, "set_roi"):
        roi_x = max(0.0, min(d["x1"] for d in detections) - roi_margin_px)
        roi_y = max(0.0, min(d["y1"] for d in detections) - roi_margin_px)
        roi_x2 = max(d["x2"] for d in detections) + roi_margin_px
        roi_y2 = max(d["y2"] for d in detections) + roi_margin_px
        roi_w = roi_x2 - roi_x
        roi_h = roi_y2 - roi_y

        _log(f"Setze Hardware-ROI: x={roi_x:.0f} y={roi_y:.0f} w={roi_w:.0f} h={roi_h:.0f}")
        roi_active = camera.set_roi(roi_x, roi_y, roi_w, roi_h)
        if roi_active:
            roi_info = {"x": roi_x, "y": roi_y, "width": roi_w, "height": roi_h}
            draw_detections = [
                {**d, "x1": d["x1"] - roi_x, "y1": d["y1"] - roi_y,
                 "x2": d["x2"] - roi_x, "y2": d["y2"] - roi_y}
                for d in detections
            ]
        else:
            _log(f"Hardware-ROI konnte nicht gesetzt werden ({camera.last_error}) - nehme volles Bild auf.")

    def _save(img: np.ndarray, suffix: str, kind: str, z_offset_mm: float, exposure_us: Optional[float]) -> None:
        annotated = _draw_boxes(img, draw_detections)
        for ext in extensions:
            raw_filename = f"{base_stem}_{suffix}_raw.{ext}"
            boxed_filename = f"{base_stem}_{suffix}_boxed.{ext}"
            raw_path = save_image(img, str(raw_dir), raw_filename)
            boxed_path = save_image(annotated, str(boxed_dir), boxed_filename)
            saved_frames.append(SavedFrame(
                kind, z_offset_mm, exposure_us,
                raw_filename, raw_path, boxed_filename, boxed_path,
            ))

    try:
        # --- Z-Stack (eine repraesentative Belichtung) ---
        camera.set_exposure(z_stack_exposure_us)
        n_steps = int(round((2 * z_stack_range_mm) / z_stack_step_mm)) + 1
        offsets = np.linspace(-z_stack_range_mm, z_stack_range_mm, n_steps)

        stage.jog("Z", float(offsets[0]))
        time.sleep(settle_s)
        current = float(offsets[0])

        for i, target in enumerate(offsets):
            target = float(target)
            step = target - current
            if abs(step) > 1e-9:
                stage.jog("Z", step)
                current = target
                time.sleep(settle_s)

            _log(f"Z-Stack [{i + 1}/{n_steps}] bei Offset {target:+.4f} mm")
            img = _get_frame_or_raise()
            _save(img, f"zstack_{i:02d}", "zstack", target, z_stack_exposure_us)

        # Zurueck zur Ausgangs-Z-Position
        stage.jog("Z", -current)
        time.sleep(settle_s)

        # --- Belichtungs-Presets auf der Ausgangs-Fokusebene ---
        for exposure_us in exposure_presets_us:
            _log(f"Aufnahme bei {exposure_us:.0f}µs Belichtung...")
            camera.set_exposure(exposure_us)
            time.sleep(settle_s)
            img = _get_frame_or_raise()
            _save(img, f"exp_{int(exposure_us)}us", "exposure", 0.0, exposure_us)

    finally:
        # Belichtung in jedem Fall zuruecksetzen, auch bei einem Fehler
        # mittendrin - sonst bleibt die Kamera in einem der Presets haengen.
        if original_exposure_us is not None:
            camera.set_exposure(original_exposure_us)
        # Gleiches Prinzip fuer den Sensor-Ausschnitt - sonst bleibt die
        # Live-Ansicht/naechste Analyse auf den kleinen ROI beschraenkt.
        if roi_active:
            camera.reset_roi()

    _log(f"Fertig. {len(saved_frames)} Bild-Paare gespeichert, {len(detections)} Events erkannt.")

    metadata_path = _write_metadata(base_stem, {
        "hit": True,
        "detections": detections,
        "hardware_roi": roi_info,
        "z_stack": {
            "range_mm": z_stack_range_mm,
            "step_mm": z_stack_step_mm,
            "exposure_us": z_stack_exposure_us,
        },
        "exposure_presets_us": list(exposure_presets_us),
        "frames": [asdict(f) for f in saved_frames],
    })

    return AnalysisResult(detections=detections, saved_frames=saved_frames, metadata_path=metadata_path)