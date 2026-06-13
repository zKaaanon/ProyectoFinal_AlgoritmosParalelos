```c
/*
 * montecarlo_omp.c
 * PROYECTO: Monte Carlo con OpenMP — Fase 2 (Paralela)
 * MATERIA: Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
 * ENTREGA: Viernes 12 de junio de 2026
 * * NOTAS DE ESTUDIO / EXPLICACIÓN DE MI CÓDIGO:
 * - El modelo matemático es el mismo que en el secuencial (GBM).
 * - Como cada simulación es independiente, se puede paralelizar en corto (paralelismo ideal).
 * - OJO CON EL RNG: Si los hilos comparten la semilla, van a chocar (race condition)
 * y los resultados van a salir idénticos o corruptos. Solución: Cada hilo calcula
 * su propia "local_seed" usando su ID de hilo al entrar a la zona paralela.
 * * Compilación rápida en la terminal:
 * gcc -O2 -std=c11 -Wall -Wextra -fopenmp -o montecarlo_omp montecarlo_omp.c -lm
 * * Ejecución:
 * ./montecarlo_omp [N_simulaciones] [num_hilos] [semilla]
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
#include <omp.h>        /* Librería de OpenMP para hilos y tiempos */

#include "blackscholes.h"

/* --- Constantes financieras del problema --- */
#define S0      100.0
#define K       105.0
#define T_ANN     1.0
#define R         0.05
#define SIGMA     0.20
#define PASOS     252

/* --- Valores por defecto por si no pasan argumentos --- */
#define N_DEFAULT    1000000L
#define SEED_DEFAULT 12345u


/* Generador xorshift32: Rápido y no pesa nada. Igual al secuencial. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


/* Box-Muller para transformar los números aleatorios a una distribución normal */
static void box_muller(uint32_t *seed, double *z1, double *z2) {
    double u1 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double u2 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double mag = sqrt(-2.0 * log(u1));
    *z1 = mag * cos(2.0 * M_PI * u2);
    *z2 = mag * sin(2.0 * M_PI * u2);
}


/* * Simula la trayectoria del precio hasta el final.
 * Es thread-safe (segura para hilos) porque no toca variables globales 
 * y la seed que recibe es la copia privada de cada hilo.
 */
static double simular_precio_final(double s0,   double r,
                                    double sigma, double dt,
                                    int pasos,   uint32_t *seed) {
    double drift    = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt = sigma * sqrt(dt);

    double S = s0;
    double z1, z2;
    int i = 0;

    /* Avanzamos de dos en dos pasos por Box-Muller */
    while (i < pasos - 1) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
        S *= exp(drift + vol_sqrt * z2);
        i += 2;
    }
    /* Si el número de pasos es impar, hacemos el último que falta */
    if (i < pasos) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
    }

    return S;
}


/* Funciones auxiliares para ordenar con qsort y sacar el VaR */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double calcular_var(double *precios, long n, double nivel) {
    qsort(precios, (size_t)n, sizeof(double), cmp_double);
    long idx = (long)(nivel * (double)n);
    if (idx < 0)   idx = 0;
    if (idx >= n)  idx = n - 1;
    return S0 - precios[idx];
}


int main(int argc, char *argv[]) {

    /* Cachar parámetros desde la consola, si no, se quedan los de fábrica */
    long     N         = N_DEFAULT;
    int      num_hilos = 0;          /* 0 significa que OpenMP decida según la compu */
    uint32_t seed_base = SEED_DEFAULT;

    if (argc >= 2) N         = atol(argv[1]);
    if (argc >= 3) num_hilos = atoi(argv[2]);
    if (argc >= 4) seed_base = (uint32_t)atol(argv[3]);

    if (N <= 0) {
        fprintf(stderr, "Error: N debe ser un entero positivo.\n");
        return EXIT_FAILURE;
    }

    /* Forzar los hilos en OpenMP si el usuario metió el dato */
    if (num_hilos > 0)
        omp_set_num_threads(num_hilos);

    /* Preguntar a OpenMP cuántos hilos terminó levantando realmente */
    int hilos_reales = 0;
    #pragma omp parallel
    {
        #pragma omp single
        hilos_reales = omp_get_num_threads();
    }

    printf("═══════════════════════════════════════════════════════\n");
    printf("   Monte Carlo Financiero — Versión PARALELA (OpenMP)\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Parámetros financieros:\n");
    printf("     S0=%.1f  K=%.1f  T=%.1f año(s)  r=%.2f  σ=%.2f\n",
           S0, K, T_ANN, R, SIGMA);
    printf("     Pasos por trayectoria: %d (días hábiles)\n", PASOS);
    printf("  Parámetros de simulación:\n");
    printf("     N         = %ld trayectorias\n", N);
    printf("     Hilos OMP = %d\n", hilos_reales);
    printf("     Seed base = %u\n", seed_base);
    printf("───────────────────────────────────────────────────────\n");

    /* Guardar memoria para el arreglo de precios */
    double *precios = (double *)malloc((size_t)N * sizeof(double));
    if (!precios) {
        fprintf(stderr, "Error: malloc falló para %ld doubles (%.1f MB).\n",
                N, (double)N * sizeof(double) / 1048576.0);
        return EXIT_FAILURE;
    }

    double dt = T_ANN / (double)PASOS;

    /* * Cronómetro: Se usa omp_get_wtime() porque mide tiempo real (wall-clock).
     * La función clock() clásica de C sumaría el tiempo de todos los hilos 
     * y daría una medición falsa/gigante en paralelo.
     */
    double t_inicio = omp_get_wtime();

    double suma_payoffs = 0.0;

    /* * ¡AQUÍ EMPIEZA LA PARALELIZACIÓN CHIDA!
     * Separamos el 'parallel' del 'for' para inicializar la semilla local 
     * UNA SOLA VEZ por hilo en lugar de recalcularla en cada iteración del bucle.
     * * reduction(+:suma_payoffs): Hace que cada hilo sume por su cuenta y al final 
     * OpenMP junta todo de forma segura para evitar problemas de colisión de datos.
     */
    #pragma omp parallel reduction(+:suma_payoffs) shared(precios, N, dt)
    {
        /* * Hack de la semilla única por hilo: 
         * Multiplicar por 2654435761u (hash de Fibonacci) distribuye bien las semillas 
         * para que las secuencias aleatorias de cada hilo no se parezcan entre sí.
         * Se le suma 1u porque xorshift32 se rompe por completo si la semilla es 0.
         */
        uint32_t local_seed = seed_base
                              + (uint32_t)omp_get_thread_num() * 2654435761u
                              + 1u;

        /* Repartimos el trabajo del for equitativamente entre los hilos (schedule static) */
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            double ST     = simular_precio_final(S0, R, SIGMA, dt, PASOS, &local_seed);
            double payoff = ST - K;
            if (payoff < 0.0) payoff = 0.0; /* Opción Call europea: si no conviene, no se ejerce */

            precios[i]   = ST;
            suma_payoffs += payoff;
        }
    } /* Fin de la zona de hilos */

    double t_fin   = omp_get_wtime();
    double tiempo_s = t_fin - t_inicio;

    /* --- Post-procesamiento y cálculos finales (en el hilo principal) --- */
    double precio_mc = exp(-R * T_ANN) * (suma_payoffs / (double)N);
    double precio_bs = bs_call_price(S0, K, T_ANN, R, SIGMA);
    double error_rel = fabs(precio_mc - precio_bs) / precio_bs * 100.0;
    double var_5     = calcular_var(precios, N, 0.05);

    double suma_st = 0.0;
    for (long i = 0; i < N; i++) suma_st += precios[i];
    double ret_esp = (suma_st / N - S0) / S0 * 100.0;

    /* Imprimir las estadísticas */
    printf("\n  RESULTADOS:\n");
    printf("     Precio call Monte Carlo  : $%8.4f\n", precio_mc);
    printf("     Precio call Black-Scholes: $%8.4f\n", precio_bs);
    printf("     Error relativo           : %8.4f%%\n", error_rel);
    printf("     VaR 5%% (pérdida máx.)   : $%8.4f\n", var_5);
    printf("     Retorno esperado promedio: %8.4f%%\n", ret_esp);
    printf("\n  DESEMPEÑO:\n");
    printf("     Tiempo de ejecución : %.6f s\n", tiempo_s);
    printf("     Hilos utilizados    : %d\n", hilos_reales);
    printf("     Trayectorias/segundo: %.0f\n", (double)N / tiempo_s);
    printf("───────────────────────────────────────────────────────\n");

    if (error_rel < 1.0)
        printf("  [OK] Verificación passed — error < 1%% vs Black-Scholes\n");
    else
        printf("  [WARN] Error relativo alto (%.2f%%) — considerar N mayor\n",
               error_rel);

    /* Exportar resultados a CSV para meterlos a Supabase en la Fase 4 */
    FILE *csv = fopen("resultados_omp.csv", "w");
    if (csv) {
        fprintf(csv,
                "version,N,pasos,hilos,S0,K,T,r,sigma,"
                "precio_mc,precio_bs,error_rel_pct,"
                "var_5pct,ret_esperado_pct,tiempo_s\n");
        fprintf(csv,
                "paralelo,%ld,%d,%d,%.2f,%.2f,%.2f,%.4f,%.4f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                N, PASOS, hilos_reales, S0, K, T_ANN, R, SIGMA,
                precio_mc, precio_bs, error_rel,
                var_5, ret_esp, tiempo_s);
        fclose(csv);
        printf("  [CSV] resultados_omp.csv generado\n");
    }

    printf("═══════════════════════════════════════════════════════\n");

    free(precios); /* Limpiar la memoria */
    return EXIT_SUCCESS;
}

```
