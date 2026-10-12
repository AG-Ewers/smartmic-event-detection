"""Empirische Pixel-zu-Stage-Kalibrierung mittels Kreuzkorrelation.

Bestimmt, wie eine bekannte Stage-Bewegung (in µm) sich im Kamerabild als
Pixel-Verschiebung äußert - und zwar als vollständige 2x2-Transformations-
matrix (Jacobi-Matrix), nicht nur als einfacher µm/Pixel-Faktor pro Achse.
Das macht die Kalibrierung robust gegen eine leichte Verdrehung zwischen
Kamera- und Stage-Achsen, wie sie bei einem selbstgebauten Mikroskop-Aufbau
üblich ist (Kamera nicht exakt achsenparallel zur Stage montiert).

Ergebnis wird als JSON-Datei persistiert, damit die Kalibrierung nicht bei
jedem Programmstart wiederholt werden muss und für die Thesis dokumentiert
ist (Reproduzierbarkeit).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from camera import CameraManager
from laser import LaserController
from stage import StageController

DEFAULT_CALIBRATION_PATH = Path("calibration.json")


@dataclass
class CalibrationResult:
    """2x2-Transformationsmatrix: Bild-Pixel-Verschiebung <- Stage-µm-Bewegung.

    [dx_px]   [a  b] [stage_dx_um]
    [dy_px] = [c  d] [stage_dy_um]

    a,c: Pixel-Verschiebung pro µm Stage-Bewegung in X
    b,d: Pixel-Verschiebung pro µm Stage-Bewegung in Y
    """

    a: float
    b: float
    c: float
    d: float
    step_um: float
    timestamp: str

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=np.float64)

    def um_per_pixel_estimate(self) -> Tuple[float, float]:
        """Grobe Sanity-Check-Werte: effektives µm/Pixel je Achse (Norm der
        jeweiligen Spalte) - nuetzlich zum Plausibilisieren, nicht fuer die
        eigentliche Umrechnung (dafuer immer die volle Matrix nutzen)."""
        col_x = np.array([self.a, self.c])
        col_y = np.array([self.b, self.d])
        um_per_px_x = self.step_um / np.linalg.norm(col_x) if np.linalg.norm(col_x) > 0 else float("nan")
        um_per_px_y = self.step_um / np.linalg.norm(col_y) if np.linalg.norm(col_y) > 0 else float("nan")
        return um_per_px_x, um_per_px_y

    def rotation_deg_estimate(self) -> float:
        """Grobe Sanity-Check: impliziter Verdrehwinkel zwischen Kamera- und
        Stage-X-Achse, aus der ersten Spalte der Matrix (Reaktion auf reine
        X-Bewegung). 0 Grad = Bild-X und Stage-X exakt parallel."""
        return float(np.degrees(np.arctan2(self.c, self.a)))


def _to_gray_float32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 1:
            # Mono-Kamera liefert (H, B, 1) statt (H, B) - kein Farbbild,
            # nur eine ueberzaehlige Kanal-Dimension. cvtColor(BGR2GRAY)
            # wuerde hier mit "Bad number of channels" abstuerzen.
            img = img[:, :, 0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def measure_pixel_shift(img_before: np.ndarray, img_after: np.ndarray) -> Tuple[float, float]:
    """Misst die Verschiebung zwischen zwei Bildern per Phasenkorrelation.

    Robuster als z.B. Feature-Matching bei den eher strukturarmen,
    sich wiederholenden Mustern in Mikroskop-Aufnahmen. Ein Hanning-Fenster
    reduziert Rand-/Wraparound-Artefakte, die die Messung sonst verfälschen.

    Returns:
        (dx_px, dy_px): gemessene Verschiebung in Pixeln (x=horizontal,
        y=vertikal, wie von cv2 zurückgegeben).
    """
    a = _to_gray_float32(img_before)
    b = _to_gray_float32(img_after)
    if a.shape != b.shape:
        raise ValueError(f"Bildgrößen stimmen nicht überein: {a.shape} vs {b.shape}")

    hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx_px, dy_px), _response = cv2.phaseCorrelate(a * hann, b * hann)
    return dx_px, dy_px


def run_calibration(
    stage: StageController,
    camera: CameraManager,
    step_um: float = 20.0,
    settle_s: float = 0.5,
    n_repeats: int = 3,
    laser: Optional[LaserController] = None,
    laser_percent: float = 50.0,
    laser_on_delay_s: float = 0.15,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> CalibrationResult:
    """Führt die vollständige Kalibrierung durch: je n_repeats Testbewegungen
    in X und Y, Kreuzkorrelation der Bilder davor/danach (Median über die
    Wiederholungen), Rückkehr zur Ausgangsposition.

    Args:
        stage: Verbundene StageController-Instanz.
        camera: Verbundene CameraManager-Instanz.
        step_um: Größe der Testbewegung je Achse und Wiederholung.
        settle_s: Zusätzliche Wartezeit NACH bestätigtem Bewegungsende
            (wait_until_idle), bevor das Bild aufgenommen wird - deckt
            kamera-seitige Verzögerung ab (z.B. laufende Belichtungszeit).
            Sollte etwas über der längsten genutzten Belichtungszeit liegen.
        n_repeats: Anzahl Wiederholungen je Achse. Der Median über die
            Wiederholungen ist robuster gegen einzelne Ausreißer (Backlash-
            Restunsicherheit, Bleich-Schwankungen) als eine Einzelmessung.
        laser: Optionale LaserController-Instanz. Wenn angegeben, bleibt
            der Laser NICHT durchgehend an, sondern wird nur unmittelbar
            vor jeder Bildaufnahme kurz eingeschaltet und danach sofort
            wieder aus - reduziert kumulatives Ausbleichen des Fluorophors
            zwischen den zu vergleichenden Bildern spürbar. Ohne laser wird
            die Beleuchtung nicht angefasst (z.B. wenn sie manuell/extern
            gesteuert wird).
        laser_percent: Laserleistung (%) für die kurzen Aufnahme-Blitze,
            falls laser angegeben ist.
        laser_on_delay_s: Kurze Wartezeit nach dem Einschalten des Lasers,
            bevor das Bild aufgenommen wird (Intensität braucht etwas, bis
            sie stabil ist).
        progress_callback: Optionale Funktion für Status-Updates (z.B. für
            die UI), erhält jeweils einen kurzen Text.

    Returns:
        CalibrationResult mit der vollständigen 2x2-Transformationsmatrix
        (aus den Median-Werten der Wiederholungen berechnet).

    Raises:
        RuntimeError: Wenn Stage oder Kamera nicht verbunden sind, oder ein
            Bild während der Kalibrierung nicht aufgenommen werden konnte.
    """
    if not stage.is_connected:
        raise RuntimeError("Stage ist nicht verbunden")
    if not camera.is_connected:
        raise RuntimeError("Kamera ist nicht verbunden")

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _get_frame_or_raise() -> np.ndarray:
        if laser is not None:
            laser.set_laser(True, laser_percent)
            time.sleep(laser_on_delay_s)
        try:
            img = camera.get_frame()
        finally:
            if laser is not None:
                laser.set_laser(False)
        if img is None:
            raise RuntimeError(f"Konnte kein Bild aufnehmen ({camera.last_error})")
        return img

    def _measure_axis(axis: str) -> Tuple[float, float]:
        """Bewegt eine Achse n_repeats mal um step_um, misst jeweils die
        Pixel-Verschiebung gegen ein frisches Referenzbild direkt davor
        (nicht gegen ein einziges, ganz zu Beginn aufgenommenes Referenz-
        bild - so ist jede Einzelmessung unabhängig vom kumulativen
        Bleichzustand vorheriger Wiederholungen), und gibt den Median
        (dx_px, dy_px) über die Wiederholungen zurück."""
        dxs: list[float] = []
        dys: list[float] = []
        for i in range(n_repeats):
            _log(f"{axis}: Referenzbild {i + 1}/{n_repeats}...")
            img_before = _get_frame_or_raise()

            stage.jog(axis, step_um / 1000.0)  # µm -> mm
            if not stage.wait_until_idle():
                _log(f"Warnung: Bewegung ({axis}) wurde nicht als abgeschlossen bestätigt")
            time.sleep(settle_s)
            img_after = _get_frame_or_raise()

            dx, dy = measure_pixel_shift(img_before, img_after)
            dxs.append(dx)
            dys.append(dy)
            _log(f"{axis} [{i + 1}/{n_repeats}]: Verschiebung ({dx:.2f}, {dy:.2f}) px")

            stage.jog(axis, -step_um / 1000.0)
            stage.wait_until_idle()
            time.sleep(settle_s)

        dx_med, dy_med = float(np.median(dxs)), float(np.median(dys))
        _log(f"{axis}: Median über {n_repeats} Messungen: ({dx_med:.2f}, {dy_med:.2f}) px")
        return dx_med, dy_med

    dx1, dy1 = _measure_axis("X")
    dx2, dy2 = _measure_axis("Y")

    result = CalibrationResult(
        a=dx1 / step_um, c=dy1 / step_um,
        b=dx2 / step_um, d=dy2 / step_um,
        step_um=step_um,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    um_x, um_y = result.um_per_pixel_estimate()
    rot = result.rotation_deg_estimate()
    _log(
        f"Fertig. ~{um_x:.4f} µm/px (X), ~{um_y:.4f} µm/px (Y), "
        f"Achsverdrehung ~{rot:.1f}°"
    )

    return result


def pixel_offset_to_stage_delta_um(
    result: CalibrationResult, px_offset: float, py_offset: float
) -> Tuple[float, float]:
    """Rechnet einen Pixel-Offset (Ziel minus Bildmitte) in die nötige
    Stage-Bewegung (µm) um, um diesen Punkt zu zentrieren.

    Löst J * [stage_dx, stage_dy] = [-px_offset, -py_offset] nach der
    Stage-Bewegung auf (negatives Vorzeichen, da der Bildinhalt sich um
    genau den negativen Offset verschieben muss, damit der Zielpunkt in
    der Mitte landet).

    Raises:
        np.linalg.LinAlgError: Falls die Kalibrierungsmatrix singulär ist
            (z.B. beide Achsen haben identische Pixelantwort ergeben -
            deutet auf einen Fehler bei der Kalibrierung selbst hin).
    """
    jacobian = result.as_matrix()
    target = np.array([-px_offset, -py_offset], dtype=np.float64)
    stage_dx_um, stage_dy_um = np.linalg.solve(jacobian, target)
    return float(stage_dx_um), float(stage_dy_um)


def save_calibration(result: CalibrationResult, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> Optional[CalibrationResult]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationResult(**data)