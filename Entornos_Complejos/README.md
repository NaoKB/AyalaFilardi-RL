## Estructura

- `01_gymnasium_y_agente.ipynb`
- `02_estudio_riesgo_tabular.ipynb`
- `03_control_continuo_aproximado.ipynb`
- `main.ipynb`
- `src/`
  - `approx_agents.py`
  - `artifacts.py`
  - `core.py`
  - `environments.py`
  - `experiments.py`
  - `plotting.py`
  - `tabular_agents.py`

Los notebooks presentan el estudio y reutilizan los resultados completos ya
generados. Si falta un CSV o grafico, `src/artifacts.py` lo reconstruye desde los
datos disponibles o, cuando es imprescindible, vuelve a ejecutar solo el estudio
afectado. El codigo de entrenamiento y evaluacion esta separado en `src/`.

## Ejecución

En Colab, abre cualquiera de los notebooks y ejecuta todas las celdas. Cada
notebook localiza la raíz del proyecto y, si es necesario, intenta clonar el
repositorio configurado en `REPO_URL`.