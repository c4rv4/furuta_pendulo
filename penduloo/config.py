"""Configuración editable del sistema de identificación."""

# ---------- Comunicación ----------
SERIAL_PORT = "COM5"
BAUD = 115200

# ---------- Protocolo experimental ----------
EXPERIMENT_ANGLES_DEG = [45, 90, 135, 180, 225]   # ángulo inicial de cada experimento (0° = colgando)
ATTEMPTS_PER_EXPERIMENT = 5
SAMPLE_PERIOD_S = 0.020          # 50 Hz (debe coincidir con el Arduino)
MAX_TRIAL_S = 30.0               # corte de seguridad de un intento (el usuario también puede finalizar a mano)
TARE_S = 3.0                     # duración de la tara con el péndulo quieto y colgando

# ---------- Detección de la liberación ----------
BASELINE_S = 0.5                 # el péndulo debe estar quieto en el ángulo inicial al menos este tiempo tras "Iniciar"
RELEASE_THRESHOLD_DEG = 3.0      # desviación respecto al ángulo inicial que se considera "ya se soltó"
BASELINE_STD_WARN_DEG = 0.5      # aviso si el péndulo no estaba quieto durante la línea base

# ---------- Modelo ----------
SIGN_EPS = 0.02                  # [rad/s] regularización sign(w) ~ tanh(w/eps) para integrar la fricción de Coulomb

# Parámetros iniciales (NO se asume que sean exactos). En el modelo reducido solo intervienen
# m2, l, I_p, g, b2 y C2; el resto se conserva por trazabilidad con el modelo completo.
INITIAL_PARAMS = dict(
    m1=0.030, L=0.065, r_CM=0.035, J1_CM=1.05625e-5, J1_O=4.73125e-5,
    m2=0.014, l=0.051, I_CM=8.06e-6, I_p=4.4474e-5, g=9.81,
    b2=9.89913e-6, C2=1.84647e-5,
)

# ---------- Optimización ----------
# Con el brazo fijo solo son identificables por separado 3 combinaciones:
#   w0^2 = m2*g*l/I_p ,  b2/I_p ,  C2/I_p.
# Por eso, por defecto se ajustan I_p, b2 y C2 y se dejan fijos m2 y l (medibles directamente).
# Si añades m2 o l a esta lista, el problema queda mal condicionado (el programa lo avisará).
OPTIMIZE = ["I_p", "b2", "C2"]
BOUND_FACTORS = (0.2, 5.0)       # cotas = factor * valor inicial (mantiene los parámetros positivos)
LOSS = "linear"                  # "linear", "soft_l1", "huber"
MAX_NFEV = 60
RETIME_ROUNDS = 2                # rondas ajuste <-> realineación del instante de liberación
EXPERIMENT_WEIGHTS = {}          # p. ej. {180: 0.5} para pesar menos un experimento

# ---------- Salida ----------
OUTPUT_DIR = "resultados"
