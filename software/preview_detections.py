"""Testet das trainierte Modell auf gespeicherten Bildern und zeichnet die
gefundenen Bounding Boxes ein.

Nutzt detect_events() aus event_detector.py unveraendert - keine Aenderung
am Modell-Aufruf selbst, nur Visualisierung der Ergebnisse.

Nutzung:
    python preview_detections.py
    (oder mit eigenem Ordner:)
    python preview_detections.py "C:/Users/jakob/.../Zellteilung/Testbilder"
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from event_detector import detect_events

# Anpassen, falls kein Ordner als Kommandozeilen-Argument uebergeben wird.
DEFAULT_INPUT_DIR = Path("C:/Users/jakob/Documents/Bachelorarbeit/PySide6/images/Testbilder")
OUTPUT_SUBDIR = "annotiert"
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

BOX_COLOR = (0, 255, 0)  # BGR - Gruen
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2


def draw_detections(image, detections: list[dict]):
    """Zeichnet Bounding Boxes + Konfidenz auf eine Kopie des Bildes.

    Erwartet, dass detections die Felder x1,y1,x2,y2,confidence enthaelt
    (siehe erweiterte detect_events() in event_detector.py).
    """
    annotated = image.copy()
    if annotated.ndim == 2:
        # Graustufenbild -> BGR, damit die gruene Box sichtbar ist
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = (int(round(det[k])) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{det['confidence']:.2f}"
        label_y = y1 - 8 if y1 - 8 > 10 else y1 + 18
        cv2.putText(
            annotated, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2,
        )

    return annotated


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_DIR

    if not input_dir.exists():
        print(f"Ordner nicht gefunden: {input_dir}")
        print("Aufruf: python preview_detections.py <Ordner mit Bildern>")
        return

    output_dir = input_dir / OUTPUT_SUBDIR
    output_dir.mkdir(exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"Keine Bilder ({', '.join(sorted(IMAGE_EXTENSIONS))}) gefunden in: {input_dir}")
        return

    print(f"{len(image_paths)} Bild(er) gefunden in {input_dir}\n")

    total_detections = 0
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  {path.name}: konnte nicht gelesen werden, übersprungen")
            continue

        detections = detect_events(image)
        total_detections += len(detections)

        if detections:
            confidences = ", ".join(f"{d['confidence']:.2f}" for d in detections)
            print(f"  {path.name}: {len(detections)} Treffer (Konfidenz: {confidences})")
        else:
            print(f"  {path.name}: kein Treffer")

        annotated = draw_detections(image, detections)
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), annotated)

    print(f"\nFertig. {total_detections} Treffer insgesamt über {len(image_paths)} Bild(er).")
    print(f"Annotierte Bilder liegen in: {output_dir}")


if __name__ == "__main__":
    main()
