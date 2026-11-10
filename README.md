# Identificación experimental del péndulo de Furuta

Software en Python para **identificar los parámetros del péndulo de un péndulo rotatorio tipo Furuta** a partir de ensayos de oscilación libre. Lee el ángulo del péndulo en tiempo real con un sensor AS5600 (vía Arduino), simula en paralelo el modelo matemático teórico con las mismas condiciones iniciales reales y ajusta los parámetros del modelo hasta que su respuesta coincida con la medida.

## ¿Cómo funciona?

1. Con el **brazo fijo** (el par de retención del motor paso a paso lo sujeta), el sistema se reduce a un péndulo simple con fricción viscosa y de Coulomb, derivado de las ecuaciones de Euler–Lagrange del Furuta:

   ```
   I_p·θ'' + m2·g·l·sin(θ) + b2·θ' + C2·sign(θ') = 0        (θ = 0: péndulo colgando)
   ```

   La función signo se regulariza con `tanh(θ'/ε)` para integrar la fricción de Coulomb.
2. El péndulo se suelta desde 5 ángulos iniciales (**45°, 90°, 135°, 180° y 225°**), con **5 intentos por ángulo** (25 ensayos).
3. Para cada intento el programa detecta el instante de liberación, integra el modelo desde las condiciones iniciales medidas y muestra en vivo la respuesta experimental, la simulada y el error (RMSE).
4. Al completar cada experimento ajusta `I_p`, `b2` y `C2` por mínimos cuadrados no lineales (`scipy.optimize.least_squares`). Al completar los 25 intentos realiza una **optimización global** que entrega el modelo experimental final.

> Con el brazo fijo solo son identificables por separado `m2·g·l/I_p`, `b2/I_p` y `C2/I_p`. Por eso se ajustan `I_p`, `b2` y `C2` y se mantienen fijos `m2` y `l`. Si se añaden más parámetros, el programa avisa del mal condicionamiento (`cond(J) > 1e5`).

## Hardware

- Péndulo de Furuta impreso en 3D con motor paso a paso NEMA 17 y driver TMC2209 (parámetros iniciales tomados del proyecto de Ferrolho).
- Sensor magnético **AS5600** (12 bits) en el eje del péndulo, por I²C.
- **Arduino UNO R4** (también compatible con UNO R3): lee el sensor a 50 Hz y envía por serie `time_ms,motor_deg,sensor_deg` a 115200 baudios.
  - AS5600: SDA → A4, SCL → A5 (5 V).
  - Pin 5: EN del driver (activo en bajo) para mantener el brazo sujeto.

## Estructura del código

| Archivo | Función |
|---|---|
| `main.py` | Punto de entrada. |
| `config.py` | Puerto, protocolo (ángulos, intentos, 50 Hz), parámetros iniciales, cotas y opciones del ajuste. |
| `serial_comm.py` | Hilo de lectura serial con validación de líneas. |
| `signals.py` | Ángulo continuo, tara, detección de la liberación, error periódico y RMSE. |
| `model.py` | Modelo reducido (brazo fijo) y modelo completo (solo para verificar la reducción). |
| `fitting.py` | Definición de cada intento, evaluación del error y optimización. |
| `experiment.py` | Gestión del protocolo 5×5 y de los parámetros vigentes. |
| `storage.py` | Guardado de intentos, descartados y resultados (CSV/JSON). |
| `gui.py` | Interfaz gráfica en matplotlib (sin tkinter). |
| `selftest.py` | Pruebas sin hardware. |

## Instalación

```bash
pip install numpy scipy pandas matplotlib pyserial PyQt6
```

Carga el firmware del Arduino con el IDE de Arduino y cierra el monitor serie antes de ejecutar el programa.

## Uso

1. Edita `SERIAL_PORT` en `config.py` (por defecto `COM5`) o cámbialo en la ventana.
2. Ejecuta:

   ```bash
   python main.py
   ```
3. Pulsa **Conectar** y luego **Tara** (péndulo colgando y quieto durante 3 s).
4. Pulsa **▶ Iniciar**, coloca el péndulo en el ángulo indicado, mantenlo quieto al menos 0.5 s y suéltalo.
5. Pulsa **■ Finalizar** (o espera el corte de 30 s), revisa la figura del intento y elige **✔ Aceptar** o **✗ Descartar**.
6. Tras cada experimento (5 intentos aceptados) se ajustan los parámetros; tras los 25 se realiza el ajuste global.

El intento a 180° parte del equilibrio inestable: la liberación es menos fiable y el programa lo advierte. Puede reducirse su peso con `EXPERIMENT_WEIGHTS = {180: 0.5}` en `config.py`.

## Resultados generados

Todo se guarda en `resultados/`:

- `tara.json`: cero del sensor.
- `intentos/exp{k}_int{n}.csv|json`: señal experimental, simulada, error y parámetros de cada intento.
- `descartados/`: datos crudos de los intentos descartados.
- `ajuste_experimento_{k}.json`: parámetros y RMSE de cada experimento.
- `parametros_identificados.json`: parámetros iniciales y finales del ajuste global.
- `global/`: los 25 intentos con los parámetros finales.

## Pruebas sin hardware

```bash
python selftest.py
```

Comprueba que el modelo reducido coincide con el completo, que se conservan la energía y el periodo teórico sin fricción, y que el ajuste recupera parámetros conocidos con datos sintéticos.

## Referencias

- Proyecto de Ferrolho (parámetros geométricos y de fricción iniciales del péndulo de Furuta).
