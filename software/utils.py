"""Allgemeine Hilfsfunktionen, die von mehreren Modulen genutzt werden."""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np


def sharpness_score(img: np.ndarray) -> float:
    """Berechnet die Bildschärfe eines Graustufenbildes.

    Nutzt die Varianz des Laplace-Filters als Fokus-Metrik: je höher der
    Wert, desto schärfer das Bild.

    Args:
        img: Graustufenbild als NumPy-Array.

    Returns:
        Schärfe-Score (float). Höher = schärfer.
    """
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def timestamped_filename(prefix: str = "snapshot", extension: str = "png") -> str:
    """Erzeugt einen Dateinamen mit aktuellem Zeitstempel.

    Args:
        prefix: Präfix des Dateinamens.
        extension: Dateiendung ohne Punkt.

    Returns:
        Dateiname im Format "<prefix>_<YYYYMMDD_HHMMSS>.<extension>".
    """
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.{extension}"


def save_image(img: np.ndarray, directory: str, filename: str) -> str:
    """Speichert ein Bild in einem Verzeichnis und legt dieses bei Bedarf an.

    Args:
        img: Zu speicherndes Bild als NumPy-Array.
        directory: Zielverzeichnis (wird angelegt, falls nicht vorhanden).
        filename: Dateiname inklusive Endung.

    Returns:
        Vollständiger Pfad der gespeicherten Datei.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    cv2.imwrite(path, img)
    return path