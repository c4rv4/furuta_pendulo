"""Interfaz gráfica 100 % matplotlib (sin tkinter).

Layout de la ventana principal
─────────────────────────────────────────────────────────────────────
 ┌─────────────────────────────────────────────────────────────────┐
 │  Puerto [__COM5__]  [Conectar]  [Tara]  [Iniciar]              │  ← botones arriba
 │  [Finalizar]  [Aceptar]  [Descartar]                           │
 ├───────────────────────────────────────────────────┬─────────────┤
 │                                                   │  Progreso   │
 │   θ vs t  (línea experimental + modelo)           │  grilla 5×5 │
 │                                                   │             │
 ├───────────────────────────────────────────────────┤  Info       │
 │   Error exp − modelo                              │  texto      │
 ├───────────────────────────────────────────────────┴─────────────┤
 │  LOG (texto scrollable mediante un Axes de texto)               │
 └─────────────────────────────────────────────────────────────────┘

Adquisición serial y ajuste numérico corren en hilos daemon.
FuncAnimation refresca la figura cada 60 ms sin bloquear la GUI.
"""
from __future__ import annotations

import os
import queue
import textwrap
import threading
import traceback
from collections import deque

import matplotlib
matplotlib.use("QtAgg")          # usa PyQt6 (disponible en venv_bci)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

import config
import storage
from experiment import ExperimentManager
from fitting import build_trial, fit
from model import simulate
from serial_comm import SerialReader
from signals import (AngleTracker, angle_error, circular_median,
                     detect_release, rmse, wrap_pi)

# ── colores ──────────────────────────────────────────────────────────────────
C_EXP  = "#2196F3"   # azul   (experimental)
C_MOD  = "#FF9800"   # naranja (modelo)
C_ERR  = "#F44336"   # rojo   (error)
C_OK   = "#4CAF50"   # verde  (celda completada)
C_CUR  = "#FFC107"   # ámbar  (celda actual)
C_IDLE = "#ECEFF1"   # gris claro (celda pendiente)
C_DARK = "#212121"   # fondo oscuro
C_PANEL= "#263238"   # panel lateral
C_TEXT = "#ECEFF1"   # texto claro
C_GRID = "#37474F"   # líneas de cuadrícula


# ── figuras emergentes para revisión ─────────────────────────────────────────

def fig_attempt(trial, sim, p):
    err = np.rad2deg(angle_error(trial.theta_all, sim))
    m   = trial.mask
    r   = np.rad2deg(rmse(angle_error(trial.theta_all[m], sim[m]))) if m.any() else float("nan")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7),
                                   facecolor=C_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors=C_TEXT)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.title.set_color(C_TEXT)
        for sp in ax.spines.values():
            sp.set_color(C_GRID)
    ax1.plot(trial.t_all, np.rad2deg(trial.theta_all), color=C_EXP,
             label="Experimental (AS5600)", lw=1.5)
    ax1.plot(trial.t_all, np.rad2deg(sim), color=C_MOD,
             ls="--", label="Modelo", lw=1.5)
    ax1.set_ylabel("θ [°]", color=C_TEXT)
    ax1.legend(facecolor=C_PANEL, labelcolor=C_TEXT)
    ax1.grid(True, color=C_GRID)
    ax1.set_title(
        f"Exp. {trial.experiment} ({trial.angle_deg}°)  "
        f"intento {trial.attempt}  RMSE = {r:.2f}°", color=C_TEXT)
    ax2.plot(trial.t_all[m], err[m], color=C_ERR, lw=1.2)
    ax2.set_ylabel("Error exp − modelo [°]", color=C_TEXT)
    ax2.set_xlabel("t desde la liberación [s]", color=C_TEXT)
    ax2.grid(True, color=C_GRID)
    fig.tight_layout()
    return fig


def fig_experiment(res, exp, angle):
    trials = res.trials
    fig = plt.figure(figsize=(11, 9), facecolor=C_DARK)
    for i, tr in enumerate(trials):
        ax = fig.add_subplot(3, 2, i + 1, facecolor=C_PANEL)
        sim = simulate(res.p_final, tr.t, tr.theta0)
        ax.plot(tr.t, np.rad2deg(tr.theta), color=C_EXP, lw=1, label="Exp.")
        ax.plot(tr.t, np.rad2deg(sim), color=C_MOD, lw=1, ls="--", label="Modelo")
        ax.set_title(f"Intento {tr.attempt} · RMSE {np.rad2deg(res.per_trial_after[i]):.2f}°",
                     fontsize=9, color=C_TEXT)
        ax.grid(True, color=C_GRID)
        ax.tick_params(colors=C_TEXT)
        for sp in ax.spines.values():
            sp.set_color(C_GRID)
        if i == 0:
            ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
    ax = fig.add_subplot(3, 2, 6, facecolor=C_PANEL)
    idx = np.arange(len(trials))
    ax.bar(idx - 0.2, np.rad2deg(res.per_trial_before), 0.4, label="antes",
           color="#607D8B")
    ax.bar(idx + 0.2, np.rad2deg(res.per_trial_after), 0.4, label="después",
           color=C_MOD)
    ax.set_xticks(idx)
    ax.set_xticklabels([t.attempt for t in trials])
    ax.set_ylabel("RMSE [°]", color=C_TEXT)
    ax.tick_params(colors=C_TEXT)
    ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    fig.suptitle(
        f"Experimento {exp} ({angle}°): "
        f"RMSE {np.rad2deg(res.rmse_before):.2f}° → {np.rad2deg(res.rmse_after):.2f}°",
        color=C_TEXT)
    fig.tight_layout()
    return fig


def _summary_text(res):
    lines = [f"{'Parámetro':<8}{'Inicial':>14}{'Final':>14}{'Cambio':>9}"]
    for n in res.names:
        a, b = res.p_initial[n], res.p_final[n]
        lines.append(f"{n:<8}{a:>14.4e}{b:>14.4e}{100*(b/a-1):>+8.1f}%")
    lines.append(f"RMSE: {np.rad2deg(res.rmse_before):.3f}° → "
                 f"{np.rad2deg(res.rmse_after):.3f}°")
    lines.append(f"cond(J) = {res.cond:.1e}" +
                 ("" if res.identifiable else "  ¡mal condicionado!"))
    return "\n".join(lines)


def fig_global(res, mgr):
    fig = plt.figure(figsize=(12, 9), facecolor=C_DARK)
    for i, ang in enumerate(config.EXPERIMENT_ANGLES_DEG):
        ax = fig.add_subplot(3, 2, i + 1, facecolor=C_PANEL)
        for j, tr in enumerate([t for t in res.trials if t.experiment == i + 1]):
            sim = simulate(res.p_final, tr.t, tr.theta0)
            ax.plot(tr.t, np.rad2deg(tr.theta), color=C_EXP,
                    lw=0.8, alpha=0.6, label="Exp." if j == 0 else None)
            ax.plot(tr.t, np.rad2deg(sim), color=C_MOD,
                    lw=0.8, ls="--", alpha=0.8, label="Modelo" if j == 0 else None)
        ax.set_title(f"Experimento {i+1} ({ang}°)", fontsize=9, color=C_TEXT)
        ax.grid(True, color=C_GRID)
        ax.tick_params(colors=C_TEXT)
        for sp in ax.spines.values():
            sp.set_color(C_GRID)
        if i == 0:
            ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
    ax = fig.add_subplot(3, 2, 6, facecolor=C_PANEL)
    ax.axis("off")
    ax.text(0.05, 0.95, _summary_text(res), family="monospace", fontsize=8,
            va="top", color=C_TEXT, transform=ax.transAxes)
    fig.suptitle("Ajuste global: 25 intentos", color=C_TEXT)
    fig.tight_layout()
    return fig


# ── clase principal ───────────────────────────────────────────────────────────

class App:
    """Aplicación completa en matplotlib sin tkinter."""

    # ── inicialización ────────────────────────────────────────────────────────
    def __init__(self):
        self.rxq   = queue.Queue()
        self.jobs  = queue.Queue()
        self.reader: SerialReader | None = None
        self.mgr   = ExperimentManager(config.INITIAL_PARAMS)
        self.offset = None
        self.state  = "desconectado"
        self.acq_id = 0
        self._tick_n = 0
        self._log_lines: deque[str] = deque(maxlen=200)
        self._popup_figs: list = []          # referencias para evitar GC
        self._last_angle_str = ""
        self._live_elapsed   = 0.0
        self._live_rmse      = "--"
        self._reset_acq()

        self._build_figure()
        self._set_state("desconectado")
        self._update_progress()

        # Animación continua: llama _frame() cada 60 ms
        self._anim = FuncAnimation(
            self.fig, self._frame, interval=60, cache_frame_data=False)
        plt.show()

    # ── construcción de la figura ─────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor": C_DARK,
            "axes.facecolor":   C_PANEL,
            "axes.edgecolor":   C_GRID,
            "axes.labelcolor":  C_TEXT,
            "xtick.color":      C_TEXT,
            "ytick.color":      C_TEXT,
            "text.color":       C_TEXT,
            "grid.color":       C_GRID,
            "grid.linewidth":   0.5,
        })

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(
            "Identificación del péndulo de Furuta (brazo fijo)")

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            3, 2,
            figure=self.fig,
            left=0.04, right=0.98,
            top=0.97,  bottom=0.03,
            hspace=0.08, wspace=0.25,
            height_ratios=[0.10, 0.60, 0.30],
            width_ratios=[0.72, 0.28],
        )

        # ── fila de botones ───────────────────────────────────────────────────
        self._ax_ctrl = self.fig.add_subplot(gs[0, :])
        self._ax_ctrl.axis("off")
        self._ax_ctrl.set_facecolor(C_DARK)

        # ── gráficas de señal (columna izquierda) ────────────────────────────
        gs_left = gs[1, 0].subgridspec(2, 1, height_ratios=[3, 1.3], hspace=0.05)
        self.ax1 = self.fig.add_subplot(gs_left[0])
        self.ax2 = self.fig.add_subplot(gs_left[1], sharex=self.ax1)

        (self.l_exp,) = self.ax1.plot([], [], color=C_EXP, lw=1.6,
                                      label="Experimental (AS5600)")
        (self.l_mod,) = self.ax1.plot([], [], color=C_MOD, lw=1.6,
                                      ls="--", label="Modelo")
        (self.l_err,) = self.ax2.plot([], [], color=C_ERR, lw=1.2)

        self.ax1.set_ylabel("θ [°]")
        self.ax1.legend(loc="upper right", fontsize=8,
                        facecolor=C_PANEL, labelcolor=C_TEXT)
        self.ax1.grid(True)
        self.ax1.tick_params(labelbottom=False)
        self.ax2.set_ylabel("Error [°]")
        self.ax2.set_xlabel("t desde liberación [s]")
        self.ax2.grid(True)
        self.ax1_title = self.ax1.set_title("", fontsize=9, color=C_TEXT)

        # ── panel derecho: progreso + info ───────────────────────────────────
        gs_right = gs[1, 1].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
        self.ax_prog = self.fig.add_subplot(gs_right[0])
        self.ax_info = self.fig.add_subplot(gs_right[1])
        for ax in (self.ax_prog, self.ax_info):
            ax.set_facecolor(C_PANEL)
            ax.axis("off")
        self._build_progress_grid()

        self._info_text = self.ax_info.text(
            0.05, 0.95, "", transform=self.ax_info.transAxes,
            fontsize=8, va="top", family="monospace", color=C_TEXT)

        # ── log (fila inferior, toda la anchura) ────────────────────────────
        self.ax_log = self.fig.add_subplot(gs[2, :])
        self.ax_log.set_facecolor(C_PANEL)
        self.ax_log.axis("off")
        self._log_text = self.ax_log.text(
            0.005, 0.98, "", transform=self.ax_log.transAxes,
            fontsize=7.5, va="top", family="monospace", color=C_TEXT,
            linespacing=1.4)

        # ── botones y TextBox ─────────────────────────────────────────────────
        self._make_widgets()

    # ── grilla de progreso ────────────────────────────────────────────────────
    def _build_progress_grid(self):
        n_exp = len(config.EXPERIMENT_ANGLES_DEG)
        n_att = config.ATTEMPTS_PER_EXPERIMENT
        self.ax_prog.set_xlim(-0.6, n_att)
        self.ax_prog.set_ylim(0, n_exp)
        self.ax_prog.set_facecolor(C_PANEL)

        for e, ang in enumerate(config.EXPERIMENT_ANGLES_DEG):
            self.ax_prog.text(
                -0.1, n_exp - 1 - e + 0.5,
                f"{ang}°", ha="right", va="center",
                fontsize=7, color=C_TEXT)

        self._cells: list[list[mpatches.FancyBboxPatch]] = []
        for e in range(n_exp):
            row = []
            for a in range(n_att):
                rect = mpatches.FancyBboxPatch(
                    (a + 0.06, n_exp - 1 - e + 0.08), 0.88, 0.84,
                    boxstyle="round,pad=0.02",
                    facecolor=C_IDLE, edgecolor=C_GRID, linewidth=0.6)
                self.ax_prog.add_patch(rect)
                self.ax_prog.text(
                    a + 0.5, n_exp - 1 - e + 0.5,
                    str(a + 1), ha="center", va="center",
                    fontsize=6.5, color="#212121")
                row.append(rect)
            self._cells.append(row)

        self.ax_prog.set_title("Progreso", fontsize=8, color=C_TEXT, pad=2)

    # ── widgets ───────────────────────────────────────────────────────────────
    def _make_widgets(self):
        ctrl = self._ax_ctrl.get_position()
        y0 = ctrl.y0 + ctrl.height * 0.08
        h  = ctrl.height * 0.84
        x0 = ctrl.x0

        # Etiqueta + TextBox de puerto
        ax_lbl = self.fig.add_axes([x0, y0, 0.025, h])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.5, "Puerto:", ha="center", va="center",
                    color=C_TEXT, fontsize=8)

        ax_port = self.fig.add_axes([x0 + 0.028, y0, 0.052, h])
        self._tb_port = TextBox(ax_port, "", initial=config.SERIAL_PORT,
                                color=C_PANEL, hovercolor=C_GRID)
        self._tb_port.text_disp.set_color(C_TEXT)

        # Botones
        specs = [
            ("Conectar",    0.090, 0.055, self._toggle_connect, "btn_conn"),
            ("Tara",        0.150, 0.045, self._tare,           "btn_tare"),
            ("▶ Iniciar",   0.200, 0.058, self._start,          "btn_start"),
            ("■ Finalizar", 0.263, 0.062, self._finish,         "btn_finish"),
            ("✔ Aceptar",   0.330, 0.062, self._accept,         "btn_accept"),
            ("✗ Descartar", 0.397, 0.072, self._repeat,         "btn_repeat"),
        ]
        for label, x, w, cb, attr in specs:
            ax_b = self.fig.add_axes([x0 + x, y0, w, h])
            btn  = Button(ax_b, label, color="#37474F", hovercolor="#546E7A")
            btn.label.set_color(C_TEXT)
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda _, f=cb: f())
            setattr(self, attr, btn)

    # ── estado de botones ─────────────────────────────────────────────────────
    _STATE_ENABLED = {
        "desconectado": {"btn_conn"},
        "idle":         {"btn_conn", "btn_tare", "btn_start"},
        "tara":         set(),
        "adquiriendo":  {"btn_finish"},
        "revision":     {"btn_accept", "btn_repeat"},
        "ajustando":    set(),
        "completado":   {"btn_conn", "btn_tare"},
    }

    def _set_state(self, st: str):
        self.state = st
        enabled = self._STATE_ENABLED[st]
        all_btns = {
            "btn_conn":   self.btn_conn,
            "btn_tare":   self.btn_tare,
            "btn_start":  self.btn_start,
            "btn_finish": self.btn_finish,
            "btn_accept": self.btn_accept,
            "btn_repeat": self.btn_repeat,
        }
        for name, btn in all_btns.items():
            active = name in enabled
            btn.active = active
            btn.color      = "#37474F" if active else "#1A2327"
            btn.hovercolor = "#546E7A" if active else "#1A2327"
            btn.label.set_alpha(1.0 if active else 0.35)
        self.btn_conn.label.set_text(
            "Desconectar" if self.reader else "Conectar")

    # ── utilidades ────────────────────────────────────────────────────────────
    def log(self, msg: str):
        for line in msg.split("\n"):
            for chunk in textwrap.wrap(line, 160) or [line]:
                self._log_lines.append(chunk)

    def _reset_acq(self):
        self.t, self.sensor, self.theta = [], [], []
        self.tracker = None
        self.release = None
        self.live    = None
        self.review  = None
        self.bad0    = 0
        self.tare_buf, self.tare_t0 = [], None

    def run_async(self, fn, cb):
        def work():
            try:
                res = fn()
            except Exception as e:
                traceback.print_exc()
                res = e
            self.jobs.put((cb, res))
        threading.Thread(target=work, daemon=True).start()

    # ── conexión / tara ───────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
            self._set_state("desconectado")
            return
        port = self._tb_port.text.strip()
        self.reader = SerialReader(port, config.BAUD, self.rxq)
        self.reader.start()
        self._set_state("idle")

    def _tare(self):
        self.tare_buf, self.tare_t0 = [], None
        self.log(f"Tara: deja el péndulo colgando y quieto {config.TARE_S:.0f} s …")
        self._set_state("tara")

    def _finish_tare(self):
        ang = np.deg2rad(np.array(self.tare_buf))
        self.offset = circular_median(ang)
        sd = float(np.rad2deg(np.std(wrap_pi(ang - self.offset))))
        self.log(f"Tara lista: offset = {np.rad2deg(self.offset) % 360:.2f}°  "
                 f"({len(ang)} muestras, σ = {sd:.2f}°)")
        if sd > 0.5:
            self.log("  ⚠ El péndulo se movió durante la tara; repítela.")
        storage.save_json(
            os.path.join(config.OUTPUT_DIR, "tara.json"),
            dict(offset_deg=np.rad2deg(self.offset) % 360, std_deg=sd))
        self._set_state("idle")

    # ── bucle de animación ────────────────────────────────────────────────────
    def _frame(self, _):
        """Llamado por FuncAnimation cada 60 ms (hilo principal)."""
        # 1) mensajes seriales
        try:
            while True:
                self._handle(self.rxq.get_nowait())
        except queue.Empty:
            pass

        # 2) callbacks de hilos de trabajo
        try:
            while True:
                cb, res = self.jobs.get_nowait()
                if isinstance(res, Exception):
                    self.log(f"ERROR: {res}")
                    self._set_state("idle" if self.reader else "desconectado")
                else:
                    cb(res)
        except queue.Empty:
            pass
        except Exception:
            self.log(traceback.format_exc())

        # 3) actualizar señal en tiempo real
        if self.state == "adquiriendo":
            try:
                self._acq_update()
            except Exception:
                self.log(traceback.format_exc())

        # 4) refrescar textos / celdas
        self._refresh_ui()

    def _handle(self, item):
        kind = item[0]
        if kind == "sample":
            self._on_sample(item[1], item[2])
        elif kind == "info":
            self.log(item[1])
        elif kind == "bad":
            if self.reader and self.reader.bad_count <= 5:
                self.log(f"Línea descartada: {item[1][:60]!r}")
        elif kind == "error":
            self.log(item[1])
            self.reader = None
            self._set_state("desconectado")

    def _on_sample(self, t, s):
        """Procesa cada muestra serial recibida."""
        if self.offset is not None and self.state != "tara":
            rel = float(np.rad2deg(wrap_pi(np.deg2rad(s) - self.offset)))
            self._last_angle_str = f"θ actual: {rel:+7.1f}°  (obj {self.mgr.angle_deg}°)"
        elif self.offset is None:
            self._last_angle_str = f"Sensor crudo: {s:6.1f}°  (falta tara)"

        if self.state == "tara":
            if self.tare_t0 is None:
                self.tare_t0 = t
            self.tare_buf.append(s)
            if t - self.tare_t0 >= config.TARE_S:
                self._finish_tare()
        elif self.state == "adquiriendo":
            if self.tracker is None:
                self.tracker = AngleTracker(self.offset)
            self.t.append(t)
            self.sensor.append(s)
            self.theta.append(self.tracker.update(s))

    # ── adquisición en tiempo real ────────────────────────────────────────────
    def _start(self):
        if not self.btn_start.active:
            return
        if self.offset is None:
            self.log("⚠ Primero realiza la tara con el péndulo colgando y quieto.")
            return
        if self.mgr.all_done:
            return
        self._reset_acq()
        self.acq_id += 1
        self.bad0 = self.reader.bad_count if self.reader else 0
        self._live_rmse    = "--"
        self._live_elapsed = 0.0
        self.log(f"Intento {self.mgr.exp_idx}-{self.mgr.attempt}: coloca el péndulo "
                 f"en {self.mgr.angle_deg}°, mantenlo quieto ≥ 1 s y suéltalo.")
        self._set_state("adquiriendo")

    def _acq_update(self):
        n = len(self.t)
        if n < 10:
            return
        elapsed = self.t[-1] - self.t[0]
        self._live_elapsed = elapsed

        if self.release is None:
            rel = detect_release(np.array(self.t), np.array(self.theta),
                                 self.mgr.params)
            if rel is not None:
                self.release = rel
                self.log(f"  Liberación detectada (θ₀ = {np.rad2deg(rel['theta0']):.1f}°). "
                         "Simulando modelo …")
                p, th0, my_id = dict(self.mgr.params), rel["theta0"], self.acq_id
                grid = np.arange(0.0, config.MAX_TRIAL_S + 1.0, 0.005)
                self.run_async(lambda: (grid, simulate(p, grid, th0)),
                               lambda res: self._live_ready(res, my_id))

        self._tick_n += 1
        if self._tick_n % 2 == 0:
            self._redraw()

        if elapsed > config.MAX_TRIAL_S:
            self.log("Tiempo máximo alcanzado; se finaliza el intento.")
            self._finish()

    def _live_ready(self, res, acq_id):
        if acq_id == self.acq_id and self.state == "adquiriendo":
            self.live = res

    def _redraw(self):
        t,  th = np.array(self.t), np.array(self.theta)
        tref = self.release["t_rel"] if self.release else t[0]
        x    = t - tref

        self.l_exp.set_data(x, np.rad2deg(th))

        if self.release and self.live is not None:
            gx, gy = self.live
            m   = x >= 0
            sim = np.interp(x[m], gx, gy)
            err = angle_error(th[m], sim)
            self.l_mod.set_data(x[m], np.rad2deg(sim))
            self.l_err.set_data(x[m], np.rad2deg(err))
            self._live_rmse = f"{np.rad2deg(rmse(err)):.2f}°"
        else:
            self.l_mod.set_data([], [])
            self.l_err.set_data([], [])
            self._live_rmse = "--"

        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        if len(x):
            self.ax1.set_xlim(min(x[0], -0.5), max(x[-1], 5.0))

    def _finish(self):
        if self.state != "adquiriendo":
            return
        t, s, th = np.array(self.t), np.array(self.sensor), np.array(self.theta)
        exp, att, ang = self.mgr.exp_idx, self.mgr.attempt, self.mgr.angle_deg
        if len(t) < 25:
            self.log("Muy pocas muestras; intento descartado.")
            self._set_state("idle")
            return
        bad = (self.reader.bad_count - self.bad0) if self.reader else 0
        try:
            trial = build_trial(t, s, th, self.mgr.params, exp, ang, att, bad)
        except ValueError as e:
            self.log(f"⚠ {e}")
            storage.save_discarded(t, s, exp, att)
            self._set_state("idle")
            return
        p = dict(self.mgr.params)
        self._set_state("ajustando")
        self.run_async(lambda: storage.simulate_trial(trial, p),
                       lambda sim: self._review(trial, sim, p))

    def _review(self, trial, sim, p):
        self.review = (trial, sim)
        m = trial.mask
        r = np.rad2deg(rmse(angle_error(trial.theta_all[m], sim[m])))
        self.log(f"Intento {trial.experiment}-{trial.attempt}: "
                 f"RMSE = {r:.2f}°  ·  liberación t = {trial.t_release_abs:.2f} s")
        for w in trial.warnings:
            self.log(f"  ⚠ {w}")
        fig = fig_attempt(trial, sim, p)
        fig.canvas.manager.set_window_title(
            f"Intento {trial.experiment}-{trial.attempt}")
        self._popup_figs.append(fig)
        fig.show()
        self._set_state("revision")

    def _accept(self):
        if not self.btn_accept.active:
            return
        trial, _ = self.review
        storage.save_trial(trial, self.mgr.params, self.offset)
        ev = self.mgr.accept(trial)
        self.review = None
        self._update_progress()
        if ev == "attempt":
            self._set_state("idle")
        else:
            self._run_exp_fit(trial.experiment, final=(ev == "all"))

    def _repeat(self):
        if not self.btn_repeat.active:
            return
        trial, _ = self.review
        storage.save_discarded(trial.t_all + trial.t_release_abs,
                               trial.sensor_deg, trial.experiment, trial.attempt)
        self.review = None
        self.log("Intento descartado (datos guardados). Repite el intento.")
        self._set_state("idle")

    # ── ajuste de parámetros ──────────────────────────────────────────────────
    def _progress(self, n):
        self.rxq.put(("info", f"   … {n} evaluaciones del modelo"))

    def _run_exp_fit(self, exp, final):
        trials, p0 = self.mgr.trials_of(exp), dict(self.mgr.params)
        self._set_state("ajustando")
        self.log(f"Ajustando parámetros con los 5 intentos del experimento {exp} …")
        self.run_async(lambda: fit(trials, p0, progress=self._progress),
                       lambda res: self._exp_fit_done(exp, res, final))

    def _exp_fit_done(self, exp, res, final):
        self.mgr.exp_fits[exp] = res
        self.mgr.params = dict(res.p_final)
        storage.save_json(
            os.path.join(config.OUTPUT_DIR, f"ajuste_experimento_{exp}.json"),
            res.to_dict())
        self.log(f"Experimento {exp}:\n{_summary_text(res)}")
        if not res.identifiable:
            self.log("  ⚠ Jacobiano mal condicionado.")
        fig = fig_experiment(res, exp, config.EXPERIMENT_ANGLES_DEG[exp - 1])
        fig.canvas.manager.set_window_title(f"Experimento {exp}")
        self._popup_figs.append(fig)
        fig.show()
        if final:
            self._run_global()
        else:
            self._set_state("idle")

    def _run_global(self):
        trials, p0 = list(self.mgr.trials), dict(self.mgr.p0)
        self._set_state("ajustando")
        self.log("Optimización global con los 25 intentos …")
        self.run_async(lambda: fit(trials, p0, progress=self._progress),
                       self._global_done)

    def _global_done(self, res):
        self.mgr.global_fit = res
        self.mgr.params = dict(res.p_final)
        out = config.OUTPUT_DIR
        storage.save_json(os.path.join(out, "parametros_identificados.json"), dict(
            parametros_iniciales=self.mgr.p0,
            parametros_globales=res.p_final,
            ajuste_global=res.to_dict(),
            ajustes_por_experimento={k: v.to_dict() for k, v in self.mgr.exp_fits.items()},
            offset_tara_deg=float(np.rad2deg(self.offset) % 360)))
        for tr in res.trials:
            storage.save_trial(tr, res.p_final, self.offset,
                               out_dir=os.path.join(out, "global"))
        self.log("AJUSTE GLOBAL:\n" + _summary_text(res))
        fig = fig_global(res, self.mgr)
        fig.canvas.manager.set_window_title("Ajuste global")
        self._popup_figs.append(fig)
        fig.show()
        self._update_progress()
        self._set_state("completado")

    # ── refresco de UI ────────────────────────────────────────────────────────
    def _refresh_ui(self):
        # panel de info
        if self.mgr.all_done:
            exp_str = "Protocolo completo (25/25)"
        else:
            warn = "  ⚠ inestable" if self.mgr.angle_deg == 180 else ""
            exp_str = (f"Exp {self.mgr.exp_idx}/{self.mgr.n_exp}  "
                       f"int {self.mgr.attempt}/{self.mgr.n_att}  "
                       f"{self.mgr.angle_deg}°{warn}")

        info = (f"{exp_str}\n"
                f"{self._last_angle_str}\n"
                f"t = {self._live_elapsed:5.1f} s\n"
                f"RMSE intento: {self._live_rmse}\n"
                f"Estado: {self.state}")
        self._info_text.set_text(info)

        # log (últimas 18 líneas visibles)
        self._log_text.set_text("\n".join(list(self._log_lines)[-18:]))

        # título de la gráfica
        if self.mgr.all_done:
            self.ax1_title.set_text("Protocolo completo")
        else:
            self.ax1_title.set_text(
                f"Experimento {self.mgr.exp_idx} ({self.mgr.angle_deg}°) "
                f"· intento {self.mgr.attempt}")

    def _update_progress(self):
        matrix = self.mgr.done_matrix()
        n_exp  = len(config.EXPERIMENT_ANGLES_DEG)
        n_att  = config.ATTEMPTS_PER_EXPERIMENT
        for e in range(n_exp):
            for a in range(n_att):
                done = matrix[e][a]
                cur  = (e + 1 == self.mgr.exp_idx and
                        a + 1 == self.mgr.attempt and
                        not self.mgr.all_done)
                color = C_OK if done else (C_CUR if cur else C_IDLE)
                self._cells[e][a].set_facecolor(color)
