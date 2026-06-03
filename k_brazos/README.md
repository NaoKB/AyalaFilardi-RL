# Parte 1 - Bandido de k-brazos

En la parte 1 de la práctica se realiza un estudio de aprendizaje por refuerzo, centrándose en el estudio del bandido multibrazo con diversas familias de algoritmos. 

## Estructura

- `main.ipynb`: fichero principal de esta parte.
- `Greedy.ipynb`: estudio intrafamilia de epsilon-greedy.
- `UCB.ipynb`: estudio intrafamilia de UCB1.
- `Softmax.ipynb`: estudio intrafamilia de Softmax/Boltzmann.
- `Comparativa.ipynb`: comparacion transversal entre familias.
- `src/`
  - `algorithms/`: carpeta que contiene los algoritmos a utilizar.
  - `arms/`: ficheros que contienen la definición de los brazos del bandido.
  - `plotting/`: contiene la definición de funciones de creación de gráficas.
  - `scenarios.py`: construccion de los tres escenarios propios.
  - `bandit_experiment.py`: motor de evaluacion reproducible y tablas resumen.

## Diseno experimental

Se usan tres bandidos estacionarios de siete brazos:

- **Bernoulli**: simula anuncios con CTR bajo y gaps pequenos.
- **Binomial**: simula promociones medidas por lotes, con recompensas de conteo.
- **Normal**: simula recomendadores de tiempo de visualizacion con ruido heterocedastico.

Los algoritmos evaluados son:

- **epsilon-greedy** con `epsilon in {0.00, 0.03, 0.10}`.
- **UCB1** con `c in {0.35, 0.70, 1.40}`.
- **Softmax** con `tau in {0.10, 0.35, 1.00}`.

Todos los notebooks usan semilla base `SEED = 2026`, `STEPS = 500` y `RUNS = 75`. Las metricas principales son recompensa media, porcentaje de seleccion del brazo optimo, pseudo-regret acumulado y estadisticas finales por brazo.

## Ejecucion

En Colab, abre cualquiera de los notebooks y ejecuta todas las celdas. Cada notebook localiza la raiz del proyecto y, si se ejecuta fuera del repositorio, intenta clonar el repositorio configurado en `REPO_URL`.

En local, desde esta carpeta:

```powershell
py -m pip install numpy pandas matplotlib ipython
```

Después abre los notebooks con Jupyter, VS Code o Colab y ejecutalos en este orden recomendado:

1. `Greedy.ipynb`
2. `UCB.ipynb`
3. `Softmax.ipynb`
4. `Comparativa.ipynb`