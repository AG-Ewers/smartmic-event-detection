"""Autofokus-Sweep über die Z-Achse mittels Bildschärfe (Laplace-Varianz).

Enthält ausschließlich die Autofokus-Logik. StageController und
CameraManager werden als Abhängigkeiten übergeben, nicht selbst erzeugt.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from camera import CameraManager
from stage import StageController
from utils import sharpness_score

# Signatur: (aktueller_schritt, gesamt_schritte, offset_mm, schaerfe_score)
ProgressCallback = Callable[[int, int, float, float], None]


def autofocus_sweep(
    stage: StageController,
    camera: CameraManager,
    range_mm: float,
    step_mm: float,
    settle_s: float = 0.3,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Optional[float], List[Tuple[float, float]]]:
    """Fährt die Z-Achse relativ ab und ermittelt die schärfste Position.

    Fährt von -range_mm bis +range_mm in step_mm-Schritten, misst bei jeder
    Position die Bildschärfe und fährt am Ende zur schärfsten gefundenen
    Position zurück.

    Args:
        stage: StageController-Instanz zur Bewegung der Z-Achse.
        camera: CameraManager-Instanz zur Bildaufnahme.
        range_mm: Sweep-Bereich in Millimetern (symmetrisch um die
            Startposition).
        step_mm: Schrittweite in Millimetern.
        settle_s: Wartezeit nach jeder Bewegung, bevor ein Bild geholt wird.
        progress_callback: Optionale Funktion, die nach jedem Schritt mit
            (aktueller_schritt, gesamt_schritte, offset_mm, score) aufgerufen
            wird.

    Returns:
        Tuple aus (bestes_offset_mm, alle_ergebnisse). bestes_offset_mm ist
        None, wenn kein einziges Bild aufgenommen werden konnte.
    """
    n_steps = int((2 * range_mm) / step_mm) + 1
    offsets = np.linspace(-range_mm, range_mm, n_steps)

    stage.jog("Z", float(offsets[0]))
    time.sleep(settle_s)

    results: List[Tuple[float, float]] = []
    current = float(offsets[0])

    for i, target in enumerate(offsets):
        target = float(target)
        step = target - current
        if abs(step) > 1e-6:
            stage.jog("Z", step)
            current = target
            time.sleep(settle_s)

        img = camera.get_frame()
        if img is None:
            continue

        score = sharpness_score(img)
        results.append((target, score))

        if progress_callback:
            progress_callback(i + 1, n_steps, target, score)

    # Zurück zum Sweep-Start
    stage.jog("Z", -current)
    time.sleep(settle_s)

    if not results:
        return None, []

    best_offset, _best_score = max(results, key=lambda item: item[1])
    stage.jog("Z", best_offset)
    time.sleep(settle_s)

    return best_offset, results