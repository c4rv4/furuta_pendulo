"""Intentos (Trial), comparación modelo-experimento y optimización de parámetros."""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

import config
from model import simulate
from signals import angle_error, detect_release, rmse


@dataclass
class Trial:
    experiment: int                  # 1..5
    angle_deg: float                 # ángulo inicial nominal
    attempt: int                     # 1..5
    t_all: np.ndarray                # s, relativo al instante de liberación (negativo = péndulo sujeto)
    sensor_deg: np.ndarray           # AS5600 crudo 0..360
    theta_all: np.ndarray            # rad, continuo y referido a la tara
    theta0: float                    # rad, ángulo inicial medido (mediana de la línea base)
    t_release_abs: float             # s, instante de liberación en el reloj de adquisición
    uncertain_release: bool = False
    warnings: list = field(default_factory=list)
    weight: float = 1.0

    @property
    def mask(self):
        return self.t_all >= 0.0

    @property
    def t(self):
        return self.t_all[self.mask]

    @property
    def theta(self):
        return self.theta_all[self.mask]


def build_trial(t_abs, sensor_deg, theta, p, experiment, angle_deg, attempt, bad_lines=0):
    """Construye un Trial a partir de la adquisición completa. Lanza ValueError si no hay liberación."""
    t_abs, sensor_deg, theta = map(lambda a: np.asarray(a, dtype=float), (t_abs, sensor_deg, theta))
    rel = detect_release(t_abs, theta, p)
    if rel is None:
        raise ValueError("No se detectó la liberación del péndulo (¿se soltó? ¿pasó la línea base de 0.5 s?)")
    tr = Trial(experiment, angle_deg, attempt, t_abs - rel["t_rel"],
               sensor_deg, theta, rel["theta0"], rel["t_rel"], rel["uncertain"],
               weight=config.EXPERIMENT_WEIGHTS.get(angle_deg, 1.0))
    w = tr.warnings
    if rel["baseline_std_deg"] > config.BASELINE_STD_WARN_DEG:
        w.append(f"El péndulo no estaba quieto antes de soltarlo (σ={rel['baseline_std_deg']:.2f}°)")
    if rel["uncertain"]:
        w.append("Cerca de un equilibrio: instante de liberación poco fiable")
    if abs(np.rad2deg(rel["theta0"]) - angle_deg) > 5:
        w.append(f"Ángulo inicial medido {np.rad2deg(rel['theta0']):.1f}° difiere del nominal {angle_deg}°")
    if tr.t.size and tr.t[-1] < 3.0:
        w.append("Registro posterior a la liberación muy corto (<3 s)")
    dts = np.diff(t_abs)
    if dts.size and dts.max() > 3 * config.SAMPLE_PERIOD_S:
        w.append(f"Hueco de datos de {dts.max() * 1000:.0f} ms (muestras perdidas)")
    if bad_lines:
        w.append(f"{bad_lines} línea(s) serial descartada(s)")
    return tr


def retime(tr, p):
    """Recalcula el instante de liberación con otros parámetros (la estimación depende débilmente del modelo)."""
    t_abs = tr.t_all + tr.t_release_abs
    rel = detect_release(t_abs, tr.theta_all, p)
    if rel is None:
        return tr
    return replace(tr, t_all=t_abs - rel["t_rel"], theta0=rel["theta0"],
                   t_release_abs=rel["t_rel"], uncertain_release=rel["uncertain"])


def evaluate(trials, p):
    """RMSE [rad] por intento y global (sqrt de la media de los MSE por intento, todos con igual peso)."""
    per = []
    for tr in trials:
        err = angle_error(tr.theta, simulate(p, tr.t, tr.theta0))
        per.append(rmse(err) if np.all(np.isfinite(err)) else float("inf"))
    return float(np.sqrt(np.mean(np.square(per)))), per


@dataclass
class FitResult:
    names: list
    p_initial: dict
    p_final: dict
    rmse_before: float
    rmse_after: float
    per_trial_before: list
    per_trial_after: list
    cond: float
    identifiable: bool
    success: bool
    message: str
    nfev: int
    trials: list = field(default=None, repr=False)     # intentos realineados con p_final

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "trials"}
        d["rmse_before_deg"] = float(np.rad2deg(self.rmse_before))
        d["rmse_after_deg"] = float(np.rad2deg(self.rmse_after))
        return d


def fit(trials, p_start, names=None, bounds=None, loss=None, max_nfev=None, progress=None):
    """Ajusta `names` minimizando el error periódico entre experimento y modelo en todos los intentos.
    Cada intento pesa igual (residuo normalizado por sqrt(N)) para que ninguno domine el ajuste."""
    names = list(names or config.OPTIMIZE)
    bounds = bounds or config.BOUND_FACTORS
    loss = loss or config.LOSS
    max_nfev = max_nfev or config.MAX_NFEV
    p_start = dict(p_start)
    p_initial = dict(p_start)
    counter = {"n": 0}

    def to_params(x):                      # x = parámetros normalizados respecto a p_start (que se actualiza)
        p = dict(p_start)
        for n, xi in zip(names, x):
            p[n] = float(xi) * p_start[n]
        return p

    def residuals(x, trials):
        p = to_params(x)
        out = []
        for tr in trials:
            r = angle_error(tr.theta, simulate(p, tr.t, tr.theta0))
            if not np.all(np.isfinite(r)):
                r = np.full(tr.t.shape, 1.0)
            out.append(r * np.sqrt(tr.weight / len(r)))
        counter["n"] += 1
        if progress and counter["n"] % 10 == 0:
            progress(counter["n"])
        return np.concatenate(out)

    rmse_b, per_b = evaluate(trials, p_start)
    x0 = np.ones(len(names))
    lb, ub = np.full_like(x0, bounds[0]), np.full_like(x0, bounds[1])
    # El instante de liberación se estima con el propio modelo: se alterna ajuste y realineación.
    cur_trials, nfev = trials, 0
    for rnd in range(config.RETIME_ROUNDS):
        if rnd:
            cur_trials = [retime(tr, p_start) for tr in trials]
        sol = least_squares(lambda x: residuals(x, cur_trials), x0, bounds=(lb, ub), loss=loss,
                            x_scale="jac", diff_step=1e-3, max_nfev=max_nfev)
        nfev += int(sol.nfev)
        p_start = to_params(sol.x)
    cur_trials = [retime(tr, p_start) for tr in trials]
    p_final = p_start
    rmse_a, per_a = evaluate(cur_trials, p_final)

    s_ = np.linalg.svd(sol.jac, compute_uv=False)
    cond = float(s_[0] / s_[-1]) if s_[-1] > 0 else float("inf")
    return FitResult(names, dict(p_initial), p_final, rmse_b, rmse_a, per_b, per_a, cond,
                     bool(cond < 1e5), bool(sol.success), sol.message, nfev, cur_trials)
