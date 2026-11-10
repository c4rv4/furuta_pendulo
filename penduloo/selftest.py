"""Pruebas sin hardware: (1) reducción del modelo, (2) energía y periodo, (3) tubería completa con datos sintéticos.
Ejecuta:  python selftest.py"""
import time

import numpy as np

import config
from fitting import build_trial, fit, evaluate
from model import full_accelerations, holding_torque, simulate, theta_ddot
from signals import AngleTracker, circular_median, wrap_pi

P0 = dict(config.INITIAL_PARAMS)
rng = np.random.default_rng(1)


def test_reduction():
    worst_phi, worst_th = 0.0, 0.0
    for th in np.linspace(-3, 3, 13):
        for w in (-8.0, -1.0, 0.3, 5.0):
            tau = holding_torque(th, w, P0)                    # par que mantiene el brazo fijo
            phi_dd, th_dd = full_accelerations(th, 0.0, w, tau, P0)
            worst_phi = max(worst_phi, abs(phi_dd))
            worst_th = max(worst_th, abs(th_dd - theta_ddot(th, w, P0)))
    print(f"[1] Reducción: |phi''|max = {worst_phi:.2e} rad/s^2, |theta''_completo - theta''_reducido|max = {worst_th:.2e}")
    assert worst_phi < 1e-9 and worst_th < 1e-9


def test_energy_and_period():
    p = dict(P0, b2=0.0, C2=0.0)
    from scipy.integrate import solve_ivp
    k = p["m2"] * p["g"] * p["l"] / p["I_p"]
    sol = solve_ivp(lambda _t, y: [y[1], theta_ddot(y[0], y[1], p)], (0, 5), [2.0, 0.0],
                    method="LSODA", rtol=1e-8, atol=1e-10, t_eval=np.linspace(0, 5, 2001))
    th, w = sol.y                                   # velocidad del propio integrador (sin diferencias finitas)
    E = 0.5 * w ** 2 - k * np.cos(th)               # energía por unidad de I_p
    drift = (E.max() - E.min()) / abs(E[0])
    # periodo de pequeña amplitud
    ts = np.linspace(0, 3, 30001)
    ths = simulate(p, ts, np.deg2rad(1.0))
    cross = ts[1:][(ths[:-1] > 0) & (ths[1:] <= 0)]
    T_sim = np.mean(np.diff(cross))
    T_th = 2 * np.pi * np.sqrt(p["I_p"] / (p["m2"] * p["g"] * p["l"]))
    print(f"[2] Deriva de energía = {drift:.1e}; periodo simulado = {T_sim:.5f} s, teórico = {T_th:.5f} s")
    assert drift < 1e-3 and abs(T_sim - T_th) / T_th < 1e-3


def synth_stream(angle_deg, p_true, offset_deg, hold_s, dur_s=10.0, noise_deg=0.10):
    """Simula lo que enviaría el Arduino: AS5600 0..360, 50 Hz, cuantizado a 12 bits y con ruido."""
    t = np.arange(0, hold_s + dur_s, config.SAMPLE_PERIOD_S)
    th0 = np.deg2rad(angle_deg)
    th = np.full(t.shape, th0)
    after = t >= hold_s
    th[after] = simulate(p_true, t[after] - hold_s, th0)
    deg = np.rad2deg(th) + offset_deg + rng.normal(0, noise_deg, t.shape)
    deg = np.round((deg % 360.0) / (360 / 4096)) * (360 / 4096)
    return t, deg % 360.0, hold_s


def test_pipeline():
    p_true = dict(P0, I_p=P0["I_p"] * 1.15, b2=P0["b2"] * 2.0, C2=P0["C2"] * 0.7)
    offset_deg = 123.4                       # el "cero" del sensor no coincide con la vertical colgante
    # tara sintética: péndulo quieto colgando
    tare = (offset_deg + rng.normal(0, 0.1, 150)) % 360
    offset = circular_median(np.deg2rad(tare))
    print(f"[3] Tara: offset real {offset_deg:.2f}°, estimado {np.rad2deg(offset) % 360:.2f}°")

    trials, t_err, holds = [], [], []
    for angle in (45, 90, 135, 225):         # 180° se discute aparte (equilibrio inestable)
        for att in (1, 2):
            hold = rng.uniform(1.0, 2.0)
            t, deg, hold_true = synth_stream(angle, p_true, offset_deg, hold)
            trk = AngleTracker(offset)
            theta = np.array([trk.update(d) for d in deg])
            tr = build_trial(t, deg, theta, P0, 1, angle, att)
            t_err.append(tr.t_release_abs - hold_true)
            holds.append(hold_true)
            trials.append(tr)
    print(f"    Error de la liberación estimada: media {np.mean(t_err) * 1e3:+.1f} ms, "
          f"max |.| {np.max(np.abs(t_err)) * 1e3:.1f} ms (periodo de muestreo 20 ms)")
    rmse0, _ = evaluate(trials, P0)
    t0 = time.time()
    res = fit(trials, P0, ["I_p", "b2", "C2"])
    print(f"    Ajuste en {time.time() - t0:.1f} s, nfev={res.nfev}, cond(J)={res.cond:.1e}")
    print(f"    RMSE antes {np.rad2deg(res.rmse_before):.3f}° -> después {np.rad2deg(res.rmse_after):.3f}°")
    t_err2 = [tr.t_release_abs - h for tr, h in zip(res.trials, holds)]
    print(f"    Tras realinear con los parámetros ajustados: error de liberación media {np.mean(t_err2) * 1e3:+.1f} ms, "
          f"max |.| {np.max(np.abs(t_err2)) * 1e3:.1f} ms")
    for n in ("I_p", "b2", "C2"):
        print(f"    {n}: verdadero {p_true[n]:.4e} | ajustado {res.p_final[n]:.4e} | error {100 * (res.p_final[n] / p_true[n] - 1):+.1f} %")
    assert res.rmse_after < res.rmse_before
    return res


if __name__ == "__main__":
    test_reduction()
    test_energy_and_period()
    test_pipeline()
    print("selftest terminado")
