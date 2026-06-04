# Parte 1 - Bandido de k-brazos

Esta parte estudia el dilema exploración-explotación en bandidos estacionarios,
comparando las familias epsilon-greedy, UCB1 y Softmax/Boltzmann.

## Estructura

- `main.ipynb`: índice ejecutable de esta parte.
- `Greedy.ipynb`: estudio intrafamilia de epsilon-greedy.
- `UCB.ipynb`: estudio intrafamilia de UCB1.
- `Softmax.ipynb`: estudio intrafamilia de Softmax/Boltzmann.
- `Comparativa.ipynb`: comparación transversal entre familias.
- `src/`
  - `algorithms/`: implementaciones de los algoritmos.
  - `arms/`: definición de los brazos y del bandido.
  - `plotting/`: funciones de representación.
  - `scenarios.py`: construcción de los tres escenarios.
  - `bandit_experiment.py`: motor de evaluación y tablas resumen.

## Diseño experimental

Se utilizan tres bandidos fijos de siete brazos. En todos ellos el brazo óptimo
es el índice `6`, pero cambia la escala y la incertidumbre de las recompensas:

- **Bernoulli**: recompensas binarias con probabilidades de éxito cercanas.
- **Binomial**: recompensas de conteo con diferencias entre brazos más visibles.
- **Normal**: recompensas continuas afectadas por ruido.

Las configuraciones evaluadas son:

- **epsilon-greedy**: `epsilon in {0.00, 0.03, 0.10}`.
- **UCB1**: `c in {0.35, 0.70, 1.40}`.
- **Softmax**: `tau in {0.10, 0.35, 1.00}`.

Todos los notebooks usan `SEED = 2026`, `STEPS = 500` y `RUNS = 75`. Se
registran la recompensa media global y del último 20%, el porcentaje de
selección óptima, el pseudo-regret acumulado y las estadísticas finales por
brazo. El regret permite ordenar configuraciones dentro de un escenario, pero
sus valores absolutos no deben compararse entre distribuciones con escalas
distintas.

## Ejecución

En Colab, abre cualquiera de los notebooks y ejecuta todas las celdas. Cada
notebook localiza la raíz del proyecto y, si es necesario, intenta clonar el
repositorio configurado en `REPO_URL`.

En local, desde esta carpeta:

```powershell
py -m pip install numpy pandas matplotlib ipython
```

Orden recomendado:

1. `Greedy.ipynb`
2. `UCB.ipynb`
3. `Softmax.ipynb`
4. `Comparativa.ipynb`