"""Modelo del péndulo de Furuta: ecuaciones completas (solo verificación) y modelo reducido (brazo fijo)."""
import numpy as np
from scipy.integrate import solve_ivp

import config


# ----------------------------------------------------------------------------------------
# Modelo REDUCIDO (phi = 0, phi' = 0, phi'' = 0)
# En la ecuación (2) desaparecen D*phi'' y -Ip*sin*cos*phi'^2, y queda:
#   Ip*theta'' + m2*g*l*sin(theta) + b2*theta' + C2*sign(theta') = 0
# La ecuación (1) pasa a ser  tau_m = D*theta'' - m2*L*l*sin(theta)*theta'^2 : es el par que el motor
# debe aplicar para mantener el brazo quieto (reacción); NO afecta a theta si el brazo está realmente fijo.
# Convención: theta = 0 es la posición colgante estable.
# ----------------------------------------------------------------------------------------
def theta_ddot(theta, omega, p, eps=config.SIGN_EPS):
    return -(p["m2"] * p["g"] * p["l"] * np.sin(theta)
             + p["b2"] * omega
             + p["C2"] * np.tanh(omega / eps)) / p["I_p"]


def simulate(p, t_eval, theta0, omega0=0.0, eps=config.SIGN_EPS):
    """Integra el modelo reducido desde t=0 y devuelve theta [rad] en los instantes t_eval (>= 0)."""
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.size == 0:
        return t_eval.copy()

    def f(_t, y):
        return [y[1], theta_ddot(y[0], y[1], p, eps)]

    sol = solve_ivp(f, (0.0, max(float(t_eval[-1]), 1e-6)), [theta0, omega0],
                    method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if (not sol.success) or sol.y.shape[1] != t_eval.size:
        return np.full(t_eval.shape, np.nan)
    return sol.y[0]


# ----------------------------------------------------------------------------------------
# Modelo COMPLETO (solo para verificar la reducción; no se usa en la identificación)
# ----------------------------------------------------------------------------------------
def full_accelerations(theta, phi_dot, theta_dot, tau, p, eps=config.SIGN_EPS):
    """Resuelve el sistema 2x2 de las ecuaciones completas y devuelve [phi_ddot, theta_ddot]."""
    s, c = np.sin(theta), np.cos(theta)
    A = p["J1_O"] + p["m2"] * p["L"] ** 2 + p["I_p"] * s ** 2
    D = p["m2"] * p["L"] * p["l"] * c
    M = np.array([[A, D], [D, p["I_p"]]])
    rhs = np.array([
        tau - 2 * p["I_p"] * s * c * phi_dot * theta_dot + p["m2"] * p["L"] * p["l"] * s * theta_dot ** 2,
        p["I_p"] * s * c * phi_dot ** 2 - p["m2"] * p["g"] * p["l"] * s
        - p["b2"] * theta_dot - p["C2"] * np.tanh(theta_dot / eps),
    ])
    return np.linalg.solve(M, rhs)


def holding_torque(theta, theta_dot, p, eps=config.SIGN_EPS):
    """Par del motor que mantiene phi'' = 0 (ecuación (1) con phi' = phi'' = 0)."""
    D = p["m2"] * p["L"] * p["l"] * np.cos(theta)
    th_dd = theta_ddot(theta, theta_dot, p, eps)
    return D * th_dd - p["m2"] * p["L"] * p["l"] * np.sin(theta) * theta_dot ** 2
