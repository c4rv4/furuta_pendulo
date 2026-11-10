"""Almacenamiento de intentos y resultados (CSV / JSON)."""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from model import simulate
from signals import angle_error


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_np)


def simulate_trial(trial, p):
    """Simulación para todas las muestras: antes de la liberación el péndulo está sujeto (theta0)."""
    sim = np.full(trial.t_all.shape, trial.theta0, dtype=float)
    sim[trial.mask] = simulate(p, trial.t, trial.theta0)
    return sim


def save_trial(trial, p, offset_rad, out_dir=config.OUTPUT_DIR):
    """CSV por intento: tiempo, ángulo experimental, ángulo simulado, error, ids y parámetros usados."""
    sim = simulate_trial(trial, p)
    err = np.where(trial.mask, np.rad2deg(angle_error(trial.theta_all, sim)), np.nan)
    df = pd.DataFrame({
        "experimento": trial.experiment, "intento": trial.attempt,
        "t_s": trial.t_all, "usado_en_ajuste": trial.mask,
        "sensor_deg_crudo": trial.sensor_deg,
        "theta_exp_deg": np.rad2deg(trial.theta_all),
        "theta_sim_deg": np.rad2deg(sim),
        "error_deg": err,
    })
    for k, v in p.items():
        df[f"p_{k}"] = v
    base = os.path.join(out_dir, "intentos", f"exp{trial.experiment}_int{trial.attempt}")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    df.to_csv(base + ".csv", index=False)
    save_json(base + ".json", dict(
        experimento=trial.experiment, angulo_nominal_deg=trial.angle_deg, intento=trial.attempt,
        theta0_medido_deg=float(np.rad2deg(trial.theta0)), t_liberacion_reloj_s=trial.t_release_abs,
        liberacion_incierta=trial.uncertain_release, avisos=trial.warnings,
        offset_tara_deg=float(np.rad2deg(offset_rad)), parametros=p,
        rmse_deg=float(np.sqrt(np.nanmean(err[trial.mask] ** 2)))))
    return base + ".csv"


def save_discarded(trial_t_abs, sensor_deg, experiment, attempt, out_dir=config.OUTPUT_DIR):
    """Guarda los datos crudos de un intento descartado para no perder mediciones."""
    os.makedirs(os.path.join(out_dir, "descartados"), exist_ok=True)
    name = datetime.now().strftime(f"exp{experiment}_int{attempt}_%H%M%S.csv")
    pd.DataFrame({"t_s": trial_t_abs, "sensor_deg_crudo": sensor_deg}).to_csv(
        os.path.join(out_dir, "descartados", name), index=False)
