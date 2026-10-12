"""
camera.py

Kamerasteuerung für Allied Vision Kameras über vmbpy.
Die Kamera wird einmal geöffnet und bleibt verbunden, bis disconnect()
aufgerufen wird.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from vmbpy import VmbSystem
except ImportError:
    VmbSystem = None


class CameraManager:

    def __init__(self):

        self._vmb = None
        self._cam = None

        self._connected = False
        self.last_error = None

    # ---------------------------------------------------------
    # Eigenschaften
    # ---------------------------------------------------------

    @property
    def is_connected(self):
        return self._connected

    # ---------------------------------------------------------
    # Verbindung
    # ---------------------------------------------------------

    def connect(self):

        if self._connected:
            return True

        if VmbSystem is None:
            self.last_error = "vmbpy nicht installiert"
            return False

        try:

            self._vmb = VmbSystem.get_instance()
            self._vmb.__enter__()

            cameras = self._vmb.get_all_cameras()

            if len(cameras) == 0:
                self.last_error = "Keine Allied Vision Kamera gefunden"
                self.disconnect()
                return False

            self._cam = cameras[0]
            self._cam.__enter__()

            self._connected = True
            self.last_error = None

            print(f"Verbunden mit: {self._cam.get_name()}")

            return True

        except Exception as e:

            self.last_error = str(e)
            self.disconnect()
            return False

    # ---------------------------------------------------------
    # Bild aufnehmen
    # ---------------------------------------------------------

    def get_frame(self, timeout_ms: int = 1000):
        """Holt ein Einzelbild von der Kamera.

        Args:
            timeout_ms: Maximale Wartezeit auf ein Frame. Ohne dieses Limit
                kann der Aufruf nach einem Sleep/Wakeup des PCs (USB-Gerät
                haengt oder ist verschwunden) unbegrenzt blockieren - das
                friert die GUI komplett ein, da get_frame() im Live-View-
                Timer direkt im GUI-Thread aufgerufen wird.
        """

        if not self._connected:
            self.last_error = "Kamera nicht verbunden"
            return None

        try:

            frame = self._cam.get_frame(timeout_ms=timeout_ms)

            img = frame.as_numpy_ndarray()

            # Mono12 -> uint8 für Streamlit
            if img.dtype == np.uint16:
                img = (img / 16).astype(np.uint8)

            return img

        except Exception as e:

            self.last_error = str(e)
            # Frame-Abruf fehlgeschlagen (z.B. Timeout nach Sleep/Wakeup,
            # oder Geraet wurde getrennt) - Verbindung als tot markieren,
            # statt bei jedem weiteren Timer-Tick erneut zu blockieren.
            # Nutzer sieht dann "Nicht verbunden" und kann ueber den
            # bestehenden Verbinden-Button neu verbinden, ohne App-Neustart.
            self.disconnect()
            return None

    # ---------------------------------------------------------
    # Kameraeinstellungen
    # ---------------------------------------------------------

    def set_exposure(self, exposure_us):

        if not self._connected:
            return

        try:
            self._cam.ExposureTime.set(exposure_us)
        except Exception:
            pass

    def get_exposure(self):

        if not self._connected:
            return None

        try:
            return self._cam.ExposureTime.get()
        except Exception:
            return None

    def set_gain(self, gain):

        if not self._connected:
            return

        try:
            self._cam.Gain.set(gain)
        except Exception:
            pass

    def get_gain(self):

        if not self._connected:
            return None

        try:
            return self._cam.Gain.get()
        except Exception:
            return None

    # ---------------------------------------------------------
    # Hardware-ROI (Sensor-Ausschnitt, nicht nachtraeglicher Zuschnitt)
    # ---------------------------------------------------------

    def get_sensor_size(self):
        """Liefert die volle Sensorgroesse (width, height) in Pixeln,
        oder None falls nicht verbunden/nicht auslesbar."""

        if not self._connected:
            return None

        try:
            width_max = self._cam.Width.get_range()[1]
            height_max = self._cam.Height.get_range()[1]
            return int(width_max), int(height_max)
        except Exception:
            return None

    def set_roi(self, x, y, width, height):
        """Setzt einen Sensor-Ausschnitt (Hardware-ROI) - die Kamera liest
        danach NUR noch diesen Bereich aus, nicht das volle Bild.

        Args:
            x, y: obere linke Ecke des Ausschnitts in Pixeln.
            width, height: Groesse des Ausschnitts in Pixeln.

        Returns:
            True bei Erfolg, sonst False (last_error gesetzt). Werte
            werden dabei auf die von der Kamera erlaubten Schrittweiten/
            Grenzen gerundet bzw. begrenzt - der tatsaechlich gesetzte
            Ausschnitt kann daher minimal von den Wunschwerten abweichen.
        """

        if not self._connected:
            self.last_error = "Kamera nicht verbunden"
            return False

        try:
            # Erst Offset auf 0, sonst kann ein zu grosser alter Offset
            # das Setzen einer kleineren Width/Height blockieren (GenICam
            # prueft Width+OffsetX <= SensorWidth bei JEDEM einzelnen Setzen,
            # nicht erst am Ende).
            self._cam.OffsetX.set(0)
            self._cam.OffsetY.set(0)

            def _snap(value, feature):
                lo, hi = feature.get_range()
                step = feature.get_increment()
                value = max(lo, min(hi, value))
                return int(round(value / step) * step)

            width_snapped = _snap(width, self._cam.Width)
            height_snapped = _snap(height, self._cam.Height)
            self._cam.Width.set(width_snapped)
            self._cam.Height.set(height_snapped)

            x_snapped = _snap(x, self._cam.OffsetX)
            y_snapped = _snap(y, self._cam.OffsetY)
            self._cam.OffsetX.set(x_snapped)
            self._cam.OffsetY.set(y_snapped)

            self.last_error = None
            return True

        except Exception as e:
            self.last_error = str(e)
            return False

    def reset_roi(self):
        """Setzt den Sensor-Ausschnitt zurueck auf den vollen Sensor."""

        if not self._connected:
            return False

        try:
            self._cam.OffsetX.set(0)
            self._cam.OffsetY.set(0)
            self._cam.Width.set(self._cam.Width.get_range()[1])
            self._cam.Height.set(self._cam.Height.get_range()[1])
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    # ---------------------------------------------------------
    # Informationen
    # ---------------------------------------------------------

    def get_camera_name(self):

        if not self._connected:
            return None

        try:
            return self._cam.get_name()
        except Exception:
            return None

    def get_resolution(self):

        if not self._connected:
            return None

        try:

            img = self.get_frame()

            if img is None:
                return None

            return img.shape[::-1]

        except Exception:
            return None

    # ---------------------------------------------------------
    # Trennen
    # ---------------------------------------------------------

    def disconnect(self):

        if self._cam is not None:

            try:
                self._cam.__exit__(None, None, None)
            except Exception:
                pass

            self._cam = None

        if self._vmb is not None:

            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass

            self._vmb = None

        self._connected = False