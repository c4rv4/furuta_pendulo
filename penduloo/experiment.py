"""Gestión del protocolo: 5 experimentos x 5 intentos, parámetros vigentes y resultados de ajuste."""
import config


class ExperimentManager:
    def __init__(self, params0):
        self.p0 = dict(params0)          # parámetros iniciales (no se modifican)
        self.params = dict(params0)      # parámetros vigentes (se actualizan tras cada experimento)
        self.trials = []                 # intentos aceptados
        self.exp_fits = {}               # experimento -> FitResult
        self.global_fit = None
        self.n_exp = len(config.EXPERIMENT_ANGLES_DEG)
        self.n_att = config.ATTEMPTS_PER_EXPERIMENT

    @property
    def n_done(self):
        return len(self.trials)

    @property
    def all_done(self):
        return self.n_done >= self.n_exp * self.n_att

    @property
    def exp_idx(self):                   # 1..n_exp del siguiente intento
        return min(self.n_done // self.n_att + 1, self.n_exp)

    @property
    def attempt(self):                   # 1..n_att del siguiente intento
        return self.n_done % self.n_att + 1

    @property
    def angle_deg(self):
        return config.EXPERIMENT_ANGLES_DEG[self.exp_idx - 1]

    def trials_of(self, exp):
        return [t for t in self.trials if t.experiment == exp]

    def accept(self, trial):
        """Registra un intento. Devuelve 'attempt', 'experiment' (cerró un experimento) o 'all'."""
        self.trials.append(trial)
        if self.all_done:
            return "all"
        return "experiment" if self.n_done % self.n_att == 0 else "attempt"

    def done_matrix(self):
        return [[(e * self.n_att + a) < self.n_done for a in range(self.n_att)] for e in range(self.n_exp)]
