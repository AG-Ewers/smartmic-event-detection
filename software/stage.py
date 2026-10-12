"""FluidNC/GRBL-Steuerung über eine serielle Verbindung.

Enthält ausschließlich die StageController-Klasse, die die serielle
Kommunikation mit der Motorsteuerung kapselt.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import serial
import serial.tools.list_ports

from settings import GRBL_BAUDRATE, GRBL_PORT, GRBL_TIMEOUT, JOG_FEEDRATE

# Wie lange nach dem Öffnen des Ports maximal auf eine erste Antwort vom
# GRBL/FluidNC-Board gewartet wird, bevor connect() als fehlgeschlagen gilt.
# (Zusätzlich zur ~2s Boot-Wartezeit direkt nach dem Öffnen des Ports.)
CONNECT_RESPONSE_TIMEOUT_S = 3.0


class StageController:
    """Kapselt die serielle Kommunikation mit einer FluidNC/GRBL-Steuerung.

    Nutzung:
        stage = StageController()
        stage.connect()
        stage.jog("X", 1.0)
        stage.home()
        stage.disconnect()
    """

    def __init__(
        self,
        port: str = GRBL_PORT,
        baudrate: int = GRBL_BAUDRATE,
        timeout: float = GRBL_TIMEOUT,
        jog_feedrate: int = JOG_FEEDRATE,
    ) -> None:
        """Initialisiert den StageController im nicht verbundenen Zustand.

        Args:
            port: Serieller Port (z. B. "COM6").
            baudrate: Baudrate der Verbindung.
            timeout: Timeout in Sekunden für serielle Lesevorgänge.
            jog_feedrate: Standard-Vorschub (mm/min) für Jog-Bewegungen.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.jog_feedrate = jog_feedrate
        self._connection: Optional[serial.Serial] = None
        self.last_error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Gibt zurück, ob die serielle Verbindung aktuell offen ist."""
        return self._connection is not None and self._connection.is_open

    def set_port(self, port: str) -> None:
        """Setzt einen neuen seriellen Port.

        Trennt eine eventuell bestehende Verbindung zuerst. Nach dem
        Aufruf muss connect() erneut aufgerufen werden, um mit dem neuen
        Port zu verbinden.
        """
        if self.is_connected:
            self.disconnect()
        self.port = port

    @staticmethod
    def available_ports() -> list[str]:
        """Liefert alle aktuell verfügbaren seriellen Ports."""
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, response_timeout: float = CONNECT_RESPONSE_TIMEOUT_S) -> bool:
        """Öffnet die serielle Verbindung zu GRBL/FluidNC.

        Ruft man connect() erneut auf, während bereits verbunden ist, passiert
        nichts (idempotent).

        Anders als zuvor wird nach dem Reset-Handshake tatsächlich auf eine
        Antwort vom Board gewartet (statt blind anzunehmen, dass alles
        geklappt hat). Kommt innerhalb von `response_timeout` Sekunden keine
        Antwort, gilt die Verbindung als fehlgeschlagen - auch wenn sich der
        Port selbst öffnen ließ (z. B. weil dort ein anderes/kein Gerät
        hängt).

        Args:
            response_timeout: Maximale Wartezeit in Sekunden auf die erste
                Antwort des Boards, nachdem der Reset-Handshake gesendet
                wurde.

        Returns:
            True bei Erfolg, sonst False (siehe last_error für Details).
        """
        if self.is_connected:
            return True

        try:
            self._connection = serial.Serial(
                self.port, self.baudrate, timeout=0.2, write_timeout=1.0
            )
        except Exception as exc:  # noqa: BLE001 - Hardwarefehler sind vielfältig
            self.last_error = str(exc)
            self._connection = None
            return False

        try:
            # GRBL/FluidNC resettet beim Öffnen des Ports (DTR) und braucht
            # kurz Zeit, bis die Firmware wieder bereit ist - das lässt sich
            # nicht sinnvoll verkürzen.
            time.sleep(2)
            self._connection.reset_input_buffer()
            self._connection.write(b"\r\n\r\n")

            # Statt blind zu warten und Erfolg anzunehmen: aktiv auf eine
            # echte Antwort vom Board warten, begrenzt auf response_timeout.
            #
            # Wichtig: nicht schon beim allerersten eingetroffenen Byte-
            # Häppchen abbrechen. GRBL/FluidNC schickt beim Neustart eine
            # mehrzeilige Boot-Meldung und ist teils noch nicht bereit,
            # G-Code zu verarbeiten, während diese noch eintrudelt. Bricht
            # man zu früh ab, kann der allererste danach gesendete Befehl
            # (z.B. "Laser an") verloren gehen, obwohl die Verbindung
            # technisch schon stand. Deshalb: weiterlesen, bis mindestens
            # QUIET_PERIOD Sekunden lang keine neuen Daten mehr angekommen
            # sind (die Boot-Meldung also fertig durch ist).
            QUIET_PERIOD = 0.4
            deadline = time.monotonic() + response_timeout
            received = b""
            last_byte_time: Optional[float] = None

            while time.monotonic() < deadline:
                chunk = self._connection.read(64)
                now = time.monotonic()
                if chunk:
                    received += chunk
                    last_byte_time = now
                elif last_byte_time is not None and (now - last_byte_time) >= QUIET_PERIOD:
                    break  # Antwort ist fertig angekommen (kurze Ruhephase danach)

            self._connection.reset_input_buffer()

            if not received:
                self.last_error = (
                    f"Keine Antwort vom Gerät auf {self.port} "
                    f"(Timeout nach {response_timeout:.0f}s)"
                )
                self._connection.close()
                self._connection = None
                return False

            # Kleine zusätzliche Sicherheitswartezeit, damit die Steuerung
            # nach der Boot-Meldung sicher bereit ist, bevor der erste
            # echte Befehl gesendet wird.
            time.sleep(0.3)

            self.last_error = None
            return True

        except Exception as exc:  # noqa: BLE001 - Hardwarefehler sind vielfältig
            self.last_error = str(exc)
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:  # noqa: BLE001
                    pass
            self._connection = None
            return False

    def disconnect(self) -> None:
        """Schließt die serielle Verbindung.

        Fehler beim Schließen werden ignoriert, damit disconnect() auch in
        einem inkonsistenten Zustand sicher aufgerufen werden kann.
        """
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._connection = None

    def send_gcode(self, cmd: str) -> str:
        """Sendet einen G-Code-Befehl an GRBL/FluidNC und liest die Antwort.

        Args:
            cmd: G-Code-Befehl ohne abschließenden Zeilenumbruch.

        Returns:
            Die von der Steuerung zurückgesendeten Zeilen (zusammengefügt),
            oder ein Hinweistext falls nicht verbunden bzw. keine Antwort kam.
        """
        if not self.is_connected or self._connection is None:
            return "GRBL nicht verbunden"

        try:
            self._connection.write((cmd.strip() + "\n").encode())
            time.sleep(0.05)

            response_lines = []
            while self._connection.in_waiting:
                line = self._connection.readline().decode(errors="ignore").strip()
                if line:
                    response_lines.append(line)

            return "\n".join(response_lines) if response_lines else "OK (keine Antwort)"

        except Exception as exc:  # noqa: BLE001
            # z.B. serial.SerialTimeoutException nach Sleep/Wakeup, wenn das
            # USB-Geraet nicht mehr da ist - Verbindung als tot markieren,
            # statt mit einem kaputten Handle weiterzuarbeiten. Nutzer sieht
            # dann "Nicht verbunden" und kann ueber connect() neu verbinden.
            self.last_error = str(exc)
            self.disconnect()
            return f"Verbindung verloren: {exc}"

    def jog(self, axis: str, distance_mm: float) -> str:
        """Führt eine relative Jog-Bewegung auf der angegebenen Achse aus.

        Args:
            axis: Achsenbezeichnung, z. B. "X", "Y" oder "Z".
            distance_mm: Relative Distanz in Millimetern (kann negativ sein).

        Returns:
            Die Antwort der Steuerung (siehe send_gcode()).
        """
        cmd = f"$J=G91 G21 {axis}{distance_mm} F{self.jog_feedrate}"
        return self.send_gcode(cmd)

    def move_to(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        feedrate: int | None = None,
    ) -> str:
        """Bewegt mehrere Achsen gleichzeitig zu einer absoluten Zielposition.

        Da alle Achsen in einer einzigen G1-Zeile mit gemeinsamer Vorschub-
        geschwindigkeit übergeben werden, koordiniert FluidNC die Bewegung
        automatisch linear (die Achsen kommen gleichzeitig an, die
        Einzelgeschwindigkeiten werden intern angepasst).

        Beispiel:
            move_to(x=10, y=30) -> G90 G1 X10.000 Y30.000 F1000
        """
        cmd = "G90 G1"

        if x is not None:
            cmd += f" X{x:.3f}"
        if y is not None:
            cmd += f" Y{y:.3f}"
        if z is not None:
            cmd += f" Z{z:.3f}"

        cmd += f" F{feedrate or self.jog_feedrate}"

        return self.send_gcode(cmd)

    def home(self) -> str:
        """Startet den Homing-Zyklus für alle Achsen.

        Returns:
            Die Antwort der Steuerung (siehe send_gcode()).
        """
        return self.send_gcode("$H")

    def home_axis(self, axis: str) -> str:
        """Startet den Homing-Zyklus nur für eine einzelne Achse.

        Args:
            axis: Achsenbezeichnung, z. B. "X", "Y" oder "Z".

        Returns:
            Die Antwort der Steuerung (siehe send_gcode()).
        """
        return self.send_gcode(f"$H{axis.upper()}")

    def jog_cancel(self) -> str:
        """Bricht eine aktuell laufende Jog-Bewegung sofort ab.

        Sendet das GRBL/FluidNC-Echtzeit-Byte 0x85 (Jog Cancel). Geeignet
        für einen "Stop"-Button pro Achse - weniger disruptiv als ein
        vollständiger Soft-Reset.

        Returns:
            Die Antwort der Steuerung, oder ein Hinweistext falls nicht
            verbunden.
        """
        if not self.is_connected or self._connection is None:
            return "GRBL nicht verbunden"

        self._connection.write(bytes([0x85]))
        time.sleep(0.05)
        return self.status()

    def soft_reset(self) -> str:
        """Sendet einen vollständigen Soft-Reset.

        Bricht sämtliche Bewegungen sofort ab und setzt den Planner zurück
        (Echtzeit-Byte 0x18). Geeignet für einen globalen "Stop All"-Button.

        Returns:
            Die Antwort der Steuerung, oder ein Hinweistext falls nicht
            verbunden.
        """
        if not self.is_connected or self._connection is None:
            return "GRBL nicht verbunden"

        self._connection.write(bytes([0x18]))
        time.sleep(0.5)
        return self.status()

    def status(self) -> str:
        """Fragt den aktuellen Status der Steuerung ab.

        Returns:
            Die Antwort der Steuerung (siehe send_gcode()).
        """
        return self.send_gcode("?")

    def wait_until_idle(self, timeout_s: float = 5.0, poll_interval_s: float = 0.05) -> bool:
        """Wartet, bis GRBL/FluidNC den Status 'Idle' meldet.

        Wichtig: Jog-Kommandos ($J=...) werden nur asynchron entgegen-
        genommen - eine "ok"-Antwort auf den Befehl bedeutet nur, dass er
        angenommen wurde, NICHT dass die Bewegung abgeschlossen ist. Eine
        feste Wartezeit (time.sleep) danach ist deshalb immer nur eine
        Schätzung, die abhängig von Distanz/Feedrate falsch sein kann.
        Diese Methode fragt stattdessen wiederholt den echten Status ab.

        Args:
            timeout_s: Maximale Wartezeit, danach wird False zurückgegeben
                (z.B. falls die Bewegung feststeckt oder die Verbindung
                währenddessen verloren geht).
            poll_interval_s: Wartezeit zwischen zwei Status-Abfragen.

        Returns:
            True, wenn 'Idle' innerhalb des Timeouts erreicht wurde, sonst
            False.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.is_connected:
                return False
            response = self.status()
            if "Idle" in response:
                return True
            time.sleep(poll_interval_s)
        return False

    def get_position(self) -> tuple[float, float, float]:
        """Liest die aktuelle Maschinenposition aus der Statusantwort.

        Returns:
            Tuple (x, y, z) in Millimetern. Gibt (0.0, 0.0, 0.0) zurück,
            falls die Position nicht aus der Antwort geparst werden konnte.
        """
        reply = self.status()

        match = re.search(r"MPos:([-\d.]+),([-\d.]+),([-\d.]+)", reply)

        if match:
            return (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )

        return (0.0, 0.0, 0.0)