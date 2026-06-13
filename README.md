# Proyecto Final - Algoritmos Paralelos: Monte Carlo Financiero con OpenMP
Profesor: Mario Arturo Nieto Butrón  
Entrega: Viernes 12 de junio de 2026

### Integrantes

- Reyna Mendez Cristian Ignacio - 320149579
- López Montúfar José Eleazar - 320207219
- Dorantes Perez Brando - 31425356

---

## Descripción

Implementación secuencial y paralela (OpenMP) de una simulación Monte Carlo para valuar una **opción call europea** usando el modelo de Movimiento Browniano Geométrico (GBM) bajo medida de riesgo neutral. Los resultados se verifican contra la fórmula cerrada de Black-Scholes.

---

## Requisitos

- GCC con soporte OpenMP (`gcc -fopenmp`)
- Make
- Python 3 con `pandas`, `matplotlib` y `numpy` (solo para gráficas)

Instalar dependencias de Python:

```bash
pip install pandas matplotlib numpy
```

---

## Compilación

```bash
# Compilar ambas versiones de una vez
make all

# O compilar individualmente
gcc -O2 -std=c11 -Wall -Wextra -o montecarlo_seq montecarlo_seq.c -lm
gcc -O2 -std=c11 -Wall -Wextra -fopenmp -o montecarlo_omp montecarlo_omp.c -lm
```

---

## Ejecución

### Versión secuencial

```bash
./montecarlo_seq [N] [seed]
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `N` | Número de trayectorias a simular | 1 000 000 |
| `seed` | Semilla del generador aleatorio | 12345 |

Ejemplos:

```bash
./montecarlo_seq                    # N=1M, seed=12345
./montecarlo_seq 10000000           # N=10M
./montecarlo_seq 10000000 42        # N=10M, seed=42
```

### Versión paralela (OpenMP)

```bash
./montecarlo_omp [N] [hilos] [seed]
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `N` | Número de trayectorias a simular | 1 000 000 |
| `hilos` | Número de hilos OpenMP | Todos los núcleos disponibles |
| `seed` | Semilla base del generador aleatorio | 12345 |

Ejemplos:

```bash
./montecarlo_omp                    # N=1M, hilos automáticos
./montecarlo_omp 10000000 8         # N=10M, 8 hilos
./montecarlo_omp 10000000 4 42      # N=10M, 4 hilos, seed=42
```

### Salida esperada

Ambos programas imprimen en consola:

- Precio de la opción call (Monte Carlo y Black-Scholes)
- Error relativo vs Black-Scholes
- VaR al 5%
- Retorno esperado promedio
- Tiempo de ejecución y trayectorias por segundo

Y generan un archivo CSV con los resultados (`resultados_seq.csv` / `resultados_omp.csv`).

---

## Benchmarking completo

El script `benchmark.sh` automatiza todos los experimentos: prueba 3 tamaños de entrada (N = 100K, 1M, 10M) con 4 configuraciones de hilos (1, 2, 4, 8) y 5 repeticiones cada una.

```bash
chmod +x benchmark.sh
./benchmark.sh
```

> El benchmark ejecuta **75 ejecuciones en total**, incluyendo simulaciones con N = 10 millones de trayectorias. En una laptop de uso general se espera que tarde entre **5 y 10 minutos**. En máquinas con un solo núcleo lógico disponible puede tomar hasta **15 minutos**. Es normal ver la terminal sin output durante las ejecuciones de N = 10M, el programa está siendo ejecutado.

Al finalizar genera dos archivos:

| Archivo | Contenido |
|---------|-----------|
| `benchmarks.csv` | Datos crudos: una fila por repetición |
| `benchmarks_resumen.csv` | Métricas agregadas: promedio, desviación estándar, speedup y eficiencia |

---

## Gráficas

Una vez generados los CSVs del benchmark, ejecutar:

```bash
python graficas.py
```

> `graficas.py` requiere que `benchmarks_resumen.csv`, `resultados_seq.csv` y `resultados_omp.csv` existan en el mismo directorio. Correr primero el benchmark y ambas versiones del programa al menos una vez.

Genera 5 imágenes PNG listas para el informe:

| Archivo | Contenido |
|---------|-----------|
| `fig1_speedup.png` | Speedup vs número de hilos (con curva ideal) |
| `fig2_eficiencia.png` | Mapa de calor de eficiencia paralela |
| `fig3_tiempos.png` | Tiempos de ejecución con barras de error (±1σ) |
| `fig4_financiero.png` | Comparación Monte Carlo vs Black-Scholes |
| `fig5_dashboard.png` | Dashboard resumen con los 4 paneles anteriores |

### Ejecución Sugerida
Para poder realizar las pruebas del programa, se deja un flujo sugerido con el fin de poder probar correctamente.


```bash
./montecarlo_seq 10000000
 
./montecarlo_omp 10000000 8

./benchmark.sh
 
python graficas.py
```

---

## Parámetros financieros

Los parámetros del modelo están definidos como constantes en ambos archivos `.c`. Para reproducir exactamente los resultados del informe no es necesario modificarlos.

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `S0` | 100.0 | Precio inicial del activo |
| `K` | 105.0 | Precio de ejercicio (strike) |
| `T` | 1.0 año | Tiempo al vencimiento |
| `r` | 0.05 | Tasa libre de riesgo anual |
| `σ` | 0.20 | Volatilidad anual |
| Pasos | 252 | Días hábiles por año |

---

## Estructura del repositorio

```
.
├── montecarlo_seq.c      # Versión secuencial
├── montecarlo_omp.c      # Versión paralela (OpenMP)
├── blackscholes.h        # Fórmula cerrada para verificación
├── benchmark.sh          # Script de benchmarking automatizado
├── graficas.py           # Generación de gráficas
└── README.md             # Este archivo
```
