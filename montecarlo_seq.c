```c
/*
 * montecarlo_seq.c
 * Versión SECUENCIAL — Base para el proyecto de Monte Carlo.
 * Materia: Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
 * Entrega: Viernes 12 de junio de 2026
 *
 * Notas rápidas para el reporte de la Fase 1:
 * - Usamos el Movimiento Browniano Geométrico (GBM) para simular los precios.
 * - Generamos los aleatorios usando Box-Muller junto con xorshift32 para ir en friega.
 * - Guardamos todos los precios finales en un arreglo porque los necesitamos ordenados para el VaR.
 *
 * Cómo compilar en la terminal:
 * gcc -O2 -std=c11 -Wall -Wextra -o montecarlo_seq montecarlo_seq.c -lm
 *
 * Cómo correrlo:
 * ./montecarlo_seq [N_simulaciones] [semilla]
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "blackscholes.h"

/* --- Parámetros financieros fijos (Modificar aquí cambia todo el proyecto) --- */
#define S0      100.0   /* Precio inicial de la acción */
#define K       105.0   /* Strike price (precio de ejercicio) */
#define T_ANN     1.0   /* Tiempo de vencimiento (1 año) */
#define R         0.05  /* Tasa libre de riesgo anual */
#define SIGMA     0.20  /* Volatilidad anual de la acción */
#define PASOS     252   /* Días hábiles del año (pasos del bucle temporal) */

/* --- Valores por defecto por si corremos el programa sin pasar argumentos --- */
#define N_DEFAULT    1000000L
#define SEED_DEFAULT 12345u


/* Generador xorshift32: Nos da números aleatorios en tiempo O(1).
 * Es el clásico de Marsaglia. Usamos uint32_t para que la aritmética 
 * no haga cosas raras o desbordamientos raros en sistemas de 64 bits. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


/* Box-Muller: Transforma la distribución uniforme en una normal N(0,1).
 * El truco de sumarle 0.5 antes de dividir es para que nunca nos dé un 0 clavado 
 * y el logaritmo de adentro no vaya a explotar por una indeterminación.
 * Escupe dos números por referencia (z1 y z2) para aprovechar la llamada. */
static void box_muller(uint32_t *seed, double *z1, double *z2) {
    double u1 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double u2 = (xorshift32(seed) + 0.5) / 4294967296.0;

    double mag = sqrt(-2.0 * log(u1));
    *z1 = mag * cos(2.0 * M_PI * u2);
    *z2 = mag * sin(2.0 * M_PI * u2);
}


/* Función para aventar una trayectoria completa de la acción.
 * Precalculamos el drift y la volatilidad AFUERA del while para ahorrar CPU.
 * Como Box-Muller nos da pares de números normales, avanzamos el bucle de 2 en 2. */
static double simular_precio_final(double s0,   double r,
                                    double sigma, double dt,
                                    int pasos,   uint32_t *seed) {
    double drift    = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt = sigma * sqrt(dt);

    double S = s0;
    double z1, z2;
    int i = 0;

    /* Avanzamos doble paso por iteración */
    while (i < pasos - 1) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
        S *= exp(drift + vol_sqrt * z2);
        i += 2;
    }
    /* Si el número de pasos fuera impar, hacemos el último que falta suelto */
    if (i < pasos) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
    }

    return S;
}


/* Comparador clásico que nos pide la función qsort para ordenar de menor a mayor */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}


/* Sacar el Value at Risk (VaR): Ordena el array de precios con qsort en O(n log n).
 * Luego busca la posición del percentil y calcula la pérdida máxima restando S0. */
static double calcular_var(double *precios, long n, double nivel) {
    qsort(precios, (size_t)n, sizeof(double), cmp_double);
    long idx = (long)(nivel * (double)n);
    if (idx < 0)   idx = 0;
    if (idx >= n)  idx = n - 1;
    return S0 - precios[idx];
}


int main(int argc, char *argv[]) {

    /* Cachar los parámetros por consola, si no metemos nada agarra los de fábrica */
    long     N    = N_DEFAULT;
    uint32_t seed = SEED_DEFAULT;

    if (argc >= 2) N    = atol(argv[1]);
    if (argc >= 3) seed = (uint32_t)atol(argv[2]);

    if (N <= 0) {
        fprintf(stderr, "Error: N debe ser un entero positivo.\n");
        return EXIT_FAILURE;
    }

    printf("═══════════════════════════════════════════════════════\n");
    printf("   Monte Carlo Financiero — Versión SECUENCIAL\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Parámetros financieros:\n");
    printf("     S0=%.1f  K=%.1f  T=%.1f año(s)  r=%.2f  σ=%.2f\n",
           S0, K, T_ANN, R, SIGMA);
    printf("     Pasos por trayectoria: %d (días hábiles)\n", PASOS);
    printf("  Parámetros de simulación:\n");
    printf("     N    = %ld trayectorias\n", N);
    printf("     Seed = %u\n", seed);
    printf("───────────────────────────────────────────────────────\n");

    /* Apartar memoria dinámica para guardar los precios finales */
    double *precios = (double *)malloc((size_t)N * sizeof(double));
    if (!precios) {
        fprintf(stderr, "Error: malloc falló para %ld doubles (%.1f MB).\n",
                N, (double)N * sizeof(double) / 1048576.0);
        return EXIT_FAILURE;
    }

    double dt = T_ANN / (double)PASOS;

    /* Cronómetro: Usamos clock_gettime con CLOCK_MONOTONIC porque clock() normal
     * no sirve para comparar de forma justa contra la versión paralela de OpenMP. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double suma_payoffs = 0.0;

    /* ════════════════════════════════════════════════════════════
     * LOOP PRINCIPAL — Aquí se va casi todo el tiempo de CPU
     *
     * Nota para la Fase 2: Como ninguna iteración depende de la otra,
     * este for está regalado para meterle un #pragma omp parallel for.
     * Solo habrá que cuidar la semilla haciendo que cada hilo tenga la suya.
     * ════════════════════════════════════════════════════════════ */
    for (long i = 0; i < N; i++) {
        double ST = simular_precio_final(S0, R, SIGMA, dt, PASOS, &seed);

        precios[i] = ST;                /* Guardamos para poder calcular el VaR luego */

        double payoff = ST - K;
        if (payoff < 0.0) payoff = 0.0; /* Si no conviene ejercer la opción Call, vale cero */

        suma_payoffs += payoff;
    }

    /* Detener el cronómetro y sacar la diferencia de tiempo */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tiempo_s = (t1.tv_sec  - t0.tv_sec) +
                      (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* --- Cálculos finales y post-procesamiento de datos --- */
    double precio_mc = exp(-R * T_ANN) * (suma_payoffs / (double)N); /* Traer a valor presente */
    double precio_bs = bs_call_price(S0, K, T_ANN, R, SIGMA);
    double error_rel = fabs(precio_mc - precio_bs) / precio_bs * 100.0;
    double var_5     = calcular_var(precios, N, 0.05);

    /* Recalcular la media después del sort para sacar el retorno esperado */
    double suma_st = 0.0;
    for (long i = 0; i < N; i++) suma_st += precios[i];
    double ret_esp = (suma_st / N - S0) / S0 * 100.0;

    /* Mostrar las métricas por pantalla */
    printf("\n  RESULTADOS:\n");
    printf("     Precio call Monte Carlo  : $%8.4f\n", precio_mc);
    printf("     Precio call Black-Scholes: $%8.4f\n", precio_bs);
    printf("     Error relativo           : %8.4f%%\n", error_rel);
    printf("     VaR 5%% (pérdida máx.)   : $%8.4f\n", var_5);
    printf("     Retorno esperado promedio: %8.4f%%\n", ret_esp);
    printf("\n  DESEMPEÑO:\n");
    printf("     Tiempo de ejecución : %.6f s\n", tiempo_s);
    printf("     Trayectorias/segundo: %.0f\n", (double)N / tiempo_s);
    printf("───────────────────────────────────────────────────────\n");

    if (error_rel < 1.0)
        printf("  [OK] Verificación passed — error < 1%% vs Black-Scholes\n");
    else
        printf("  [WARN] Error relativo alto (%.2f%%) — considerar N mayor\n",
               error_rel);

    /* Guardar los datos en un CSV para mandarlos a Supabase en la Fase 4 */
    FILE *csv = fopen("resultados_seq.csv", "w");
    if (csv) {
        fprintf(csv,
                "version,N,pasos,S0,K,T,r,sigma,"
                "precio_mc,precio_bs,error_rel_pct,"
                "var_5pct,ret_esperado_pct,tiempo_s\n");
        fprintf(csv,
                "secuencial,%ld,%d,%.2f,%.2f,%.2f,%.4f,%.4f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                N, PASOS, S0, K, T_ANN, R, SIGMA,
                precio_mc, precio_bs, error_rel,
                var_5, ret_esp, tiempo_s);
        fclose(csv);
        printf("  [CSV] resultados_seq.csv generado\n");
    }

    printf("═══════════════════════════════════════════════════════\n");

    free(precios); /* Liberar la memoria asignada */
    return EXIT_SUCCESS;
}

```
