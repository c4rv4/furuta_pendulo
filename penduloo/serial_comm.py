"""Comunicación serial con el Arduino (hilo independiente). Formato: time_ms,motor_deg,sensor_deg."""
import queue
import threading

import serial


def parse_line(line):
    """Devuelve (time_ms, sensor_deg) o None si la línea es inválida. Ignora motor_deg."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        t_ms = int(parts[0])
        sensor = float(parts[2])
        float(parts[1])                      # se valida pero no se usa
    except ValueError:
        return None
    if not (0.0 <= sensor <= 360.0) or sensor != sensor:
        return None
    return t_ms, sensor


class SerialReader(threading.Thread):
    """Lee líneas y las deja en `out_q` como tuplas:
       ("sample", t_s, sensor_deg) | ("bad", linea) | ("info", texto) | ("error", texto)."""

    def __init__(self, port, baud, out_q):
        super().__init__(daemon=True)
        self.port, self.baud, self.out_q = port, baud, out_q
        self._stop_evt = threading.Event()
        self.bad_count = 0
        self.sample_count = 0
        self._ser = None

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except serial.SerialException as e:
            self.out_q.put(("error", f"No se pudo abrir {self.port}: {e}"))
            return
        self.out_q.put(("info", f"Conectado a {self.port} @ {self.baud}"))
        t0_ms, last_ms = None, None
        while not self._stop_evt.is_set():
            try:
                raw = self._ser.readline()
            except serial.SerialException as e:
                self.out_q.put(("error", f"Error serial: {e}"))
                break
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                self.out_q.put(("info", line))
                continue
            parsed = parse_line(line)
            if parsed is None:
                self.bad_count += 1
                self.out_q.put(("bad", line))
                continue
            t_ms, sensor = parsed
            if last_ms is not None and t_ms < last_ms:      # reinicio del Arduino
                t0_ms = None
            if t0_ms is None:
                t0_ms = t_ms
            last_ms = t_ms
            self.sample_count += 1
            self.out_q.put(("sample", (t_ms - t0_ms) / 1000.0, sensor))
        if self._ser and self._ser.is_open:
            self._ser.close()

    def stop(self):
        self._stop_evt.set()
