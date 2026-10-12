"""Lasersteuerung über G-Code (M3/M5).

Enthält ausschließlich die LaserController-Klasse. Der Laser wird über
G-Code-Befehle geschaltet, die über eine bestehende StageController-Instanz
gesendet werden (gleiche serielle Verbindung wie die Motorsteuerung).

Wichtig: Es wird bewusst M3 (statischer Modus) statt M4 (dynamischer
Modus) verwendet. M4 gibt die PWM-Leistung nur waehrend einer aktiven
Bewegung aus (gedacht fuers Lasergravieren, damit der Laser im Stillstand
kein Loch brennt) - fuer eine dauerhafte Mikroskop-Beleuchtung, die
unabhaengig von Stage-Bewegungen leuchten soll, ist M3 (sofort an, bleibt
an bis M5) das richtige Kommando.
"""

from __future__ import annotations

from settings import LASER_MAX_S
from stage import StageController


class LaserController:
    """Steuert den Laser über G-Code-Befehle (M3/M5).

    Nutzung:
        laser = LaserController(stage)
        laser.set_laser(True, percent=50)
        laser.set_laser(False)
    """

    def __init__(self, stage: StageController, max_intensity: int = LASER_MAX_S) -> None:
        """Initialisiert den LaserController.

        Args:
            stage: StageController-Instanz, über die G-Code gesendet wird.
            max_intensity: Maximal zulässiger Intensitätswert (S-Wert).
        """
        self.stage = stage
        self.max_intensity = max_intensity
        self.is_on: bool = False
        self.power_percent: float = 0.0

    def set_laser(self, active: bool, percent: float = 100.0) -> str:
        percent = max(0.0, min(100.0, percent))

        if active:
            # Board: S0 = volle Helligkeit, S{max} = aus -> invertiert zur
            # gewuenschten Nutzer-Prozentangabe (0% = aus, 100% = voll)
            s_value = int((100 - percent) / 100 * self.max_intensity)
            response = self.stage.send_gcode(f"M3 S{s_value}")
            self.is_on = True
            self.power_percent = percent
        else:
            # M5 funktioniert nicht zuverlaessig (kein enable_pin vorhanden) -
            # stattdessen ueber M3 mit maximalem S-Wert ausschalten, das
            # entspricht auf diesem Board dem realen "aus"-Zustand.
            response = self.stage.send_gcode(f"M3 S{self.max_intensity}")
            self.is_on = False
            self.power_percent = 0.0

        return response

    def set_power(self, percent: float) -> str:
        """Ändert die Laserleistung, falls der Laser bereits eingeschaltet ist.

        Ist der Laser aus, wird der Prozentwert nur für die nächste
        Aktivierung gemerkt, ohne einen G-Code-Befehl zu senden.
        """
        if not self.is_on:
            self.power_percent = percent
            return "Laser aus"

        return self.set_laser(True, percent)