"""Procesamiento de señales angulares: desenvolvimiento, tara, detección de la liberación y errores."""
import numpy as np

import config
from model import simulate


def wrap_pi(x):
    """Lleva ángulos [rad] al intervalo [-pi, pi)."""
    return (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def circular_median(angles_rad):
    """Mediana de ángulos periódicos (evita el salto 359°->0°)."""
    a = np.asarray(angles_rad, dtype=float)
    ref = a[0]
    return float(ref + np.median(wrap_pi(a - ref)))


class AngleTracker:
    """Convierte el AS5600 (0..360°) en un ángulo continuo [rad] respecto a la tara, muestra a muestra.
    El primer valor queda en [-pi, pi): 225° se representa como -135° (equivalente físico)."""

    def __init__(self, offset_rad):
        self.offset = float(offset_rad)
        self._prev = None
        self._cont = 0.0

    def update(self, sensor_deg):
        rel = float(wrap_pi(np.deg2rad(sensor_deg) - self.offset))
        if self._prev is None:
            self._cont = rel
        else:
            self._cont += float(wrap_pi(rel - self._prev))
        self._prev = rel
        return self._cont


def continuous_angle(sensor_deg, offset_rad):
    """Versión por lotes de AngleTracker."""
    rel = wrap_pi(np.deg2rad(np.asarray(sensor_deg, dtype=float)) - offset_rad)
    return rel[0] + np.concatenate([[0.0], np.cumsum(wrap_pi(np.diff(rel)))])


def _time_to_deviation(th0, delta, p, t_max=1.0, dt=0.0005):
    """Tiempo que tarda el modelo (desde reposo en th0) en alejarse `delta` rad de th0."""
    tg = np.arange(0.0, t_max, dt)
    dev = np.abs(simulate(p, tg, th0) - th0)
    if not np.all(np.isfinite(dev)) or dev.max() < delta:
        return None
    j = int(np.argmax(dev >= delta))
    if j == 0:
        return 0.0
    return float(tg[j - 1] + dt * (delta - dev[j - 1]) / (dev[j] - dev[j - 1]))


def detect_release(t, theta, p, baseline_s=config.BASELINE_S, thr_deg=config.RELEASE_THRESHOLD_DEG):
    """Detecta el instante de liberación del péndulo.

    1. Ángulo inicial theta0 = mediana de los primeros `baseline_s` segundos (péndulo sujeto).
    2. Primera muestra k con |theta - theta0| > umbral.
    3. El instante de liberación se estima retrocediendo desde k el tiempo que el modelo (con los parámetros
       vigentes, velocidad inicial nula) tarda en recorrer esa desviación. Si la aceleración inicial es ~0
       (cerca de 180° o 0°) esa extrapolación no es fiable: se usa la muestra anterior y se marca `uncertain`.
    Devuelve None si aún no hay liberación."""
    t = np.asarray(t, dtype=float)
    th = np.asarray(theta, dtype=float)
    nb = int(np.searchsorted(t, t[0] + baseline_s)) if len(t) else 0
    if nb < 5 or len(t) < nb + 2:
        return None
    th0 = float(np.median(th[:nb]))
    idx = np.nonzero(np.abs(th[nb:] - th0) > np.deg2rad(thr_deg))[0]
    if idx.size == 0:
        return None
    k = nb + int(idx[0])
    a0 = abs(p["m2"] * p["g"] * p["l"] * np.sin(th0) / p["I_p"])
    uncertain = a0 < 1.0
    if uncertain:
        t_rel = t[k - 1]
    else:
        tau = _time_to_deviation(th0, abs(th[k] - th0), p)
        t_rel = t[k] - (tau if tau is not None else np.sqrt(2.0 * abs(th[k] - th0) / a0))
    t_rel = float(min(max(t_rel, t[nb - 1]), t[k]))
    return dict(t_rel=t_rel, theta0=th0, k=k, uncertain=uncertain,
                baseline_std_deg=float(np.rad2deg(np.std(th[:nb]))))


def angle_error(theta_exp, theta_mod):
    """Error periódico exp - modelo, en [-pi, pi): no hay errores artificiales de 360°."""
    return wrap_pi(np.asarray(theta_exp) - np.asarray(theta_mod))


def rmse(err):
    err = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def interp_to(t_src, y_src, t_dst):
    """Interpola una señal (p. ej. simulación en malla densa) a los tiempos experimentales."""
    return np.interp(t_dst, t_src, y_src)
