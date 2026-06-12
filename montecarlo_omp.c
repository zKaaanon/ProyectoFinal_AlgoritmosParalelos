/*
 * montecarlo_omp.c
 * Versión PARALELA con OpenMP — Monte Carlo para valoración de opción call europea.
 *
 * ── Modelo financiero ──────────────────────────────────────────────────────
 * Idéntico a montecarlo_seq.c: GBM bajo medida de riesgo neutral.
 *
 *   S(t + Δt) = S(t) · exp( (r − σ²/2)·Δt + σ·ε·√Δt )
 *
 * Precio de la opción call europea:
 *   C = e^(−rT) · E[ max(S_T − K, 0) ]
 *
 * ── Estrategia de paralelización ──────────────────────────────────────────
 * Cada trayectoria i es COMPLETAMENTE INDEPENDIENTE del resto.
 * No hay dependencias de datos entre iteraciones → paralelismo perfecto.
 *
 * Problema del RNG en paralelo:
 *   Si todos los hilos compartieran la misma seed, habría race condition
 *   (condición de carrera): dos hilos leen/modifican *seed al mismo tiempo
 *   → resultados incorrectos o no reproducibles.
 *
 *   Solución: cada hilo tiene su PROPIO estado RNG, derivado de la seed
 *   base más su ID de hilo. Así cada hilo genera una secuencia aleatoria
 *   independiente, sin sincronización.
 *
 * Directivas OpenMP usadas:
 *   #pragma omp parallel for          — divide el for entre hilos
 *   reduction(+:suma_payoffs)          — acumula suma de forma segura
 *   schedule(static)                   — reparte bloques iguales (carga uniforme)
 *   private(local_seed, ST, payoff)    — variables locales por hilo
 *   firstprivate(seed_base)            — cada hilo recibe copia del valor inicial
 *
 * ── Complejidad ────────────────────────────────────────────────────────────
 *   Temporal: O(N·PASOS / p)   p = número de hilos
 *   Espacial: O(N)             arreglo de precios para VaR (compartido)
 *
 * ── Compilación ───────────────────────────────────────────────────────────
 *   gcc -O2 -std=c11 -Wall -Wextra -fopenmp -o montecarlo_omp montecarlo_omp.c -lm
 *
 * ── Uso ───────────────────────────────────────────────────────────────────
 *   ./montecarlo_omp [N] [hilos] [seed]
 *   N      = número de simulaciones  (default: 1 000 000)
 *   hilos  = número de hilos OpenMP  (default: todos los núcleos disponibles)
 *   seed   = semilla base del RNG    (default: 12345)
 *
 * ── Proyecto ──────────────────────────────────────────────────────────────
 *   Monte Carlo con OpenMP — Fase 2 (Paralela)
 *   Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
 *   Entrega: Viernes 12 de junio de 2026
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
#include <omp.h>        /* ← OpenMP: omp_get_wtime, omp_set_num_threads, etc. */

#include "blackscholes.h"

/* ══════════════════════════════════════════════════════════════════
 *  PARÁMETROS FINANCIEROS — idénticos a montecarlo_seq.c
 *  (mismos valores garantizan comparación justa seq vs paralelo)
 * ══════════════════════════════════════════════════════════════════ */
#define S0      100.0
#define K       105.0
#define T_ANN     1.0
#define R         0.05
#define SIGMA     0.20
#define PASOS     252

/* ══════════════════════════════════════════════════════════════════
 *  PARÁMETROS DE SIMULACIÓN POR DEFECTO
 * ══════════════════════════════════════════════════════════════════ */
#define N_DEFAULT    1000000L
#define SEED_DEFAULT 12345u


/* ──────────────────────────────────────────────────────────────────
 * xorshift32 — idéntico a la versión secuencial.
 * Copiado aquí para que el archivo sea autónomo.
 * ────────────────────────────────────────────────────────────────── */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


/* ──────────────────────────────────────────────────────────────────
 * box_muller — idéntico a la versión secuencial.
 * ────────────────────────────────────────────────────────────────── */
static void box_muller(uint32_t *seed, double *z1, double *z2) {
    double u1 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double u2 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double mag = sqrt(-2.0 * log(u1));
    *z1 = mag * cos(2.0 * M_PI * u2);
    *z2 = mag * sin(2.0 * M_PI * u2);
}


/* ──────────────────────────────────────────────────────────────────
 * simular_precio_final — idéntico a la versión secuencial.
 *
 * Esta función es llamada desde dentro de la región paralela.
 * Es thread-safe porque:
 *   1. No usa variables globales ni estáticas.
 *   2. El puntero 'seed' apunta al estado LOCAL del hilo (privado).
 *   3. Todas las variables internas son automáticas (pila del hilo).
 * ────────────────────────────────────────────────────────────────── */
static double simular_precio_final(double s0,   double r,
                                    double sigma, double dt,
                                    int pasos,   uint32_t *seed) {
    double drift    = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt = sigma * sqrt(dt);

    double S = s0;
    double z1, z2;
    int i = 0;

    while (i < pasos - 1) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
        S *= exp(drift + vol_sqrt * z2);
        i += 2;
    }
    if (i < pasos) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
    }

    return S;
}


/* ──────────────────────────────────────────────────────────────────
 * cmp_double / calcular_var — idénticos a la versión secuencial.
 * (Se ejecutan FUERA de la región paralela, en el hilo principal.)
 * ────────────────────────────────────────────────────────────────── */
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


/* ══════════════════════════════════════════════════════════════════
 *  PROGRAMA PRINCIPAL
 * ══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    /* ── Parámetros de línea de comandos ── */
    long     N         = N_DEFAULT;
    int      num_hilos = 0;          /* 0 = dejar que OpenMP elija */
    uint32_t seed_base = SEED_DEFAULT;

    if (argc >= 2) N         = atol(argv[1]);
    if (argc >= 3) num_hilos = atoi(argv[2]);
    if (argc >= 4) seed_base = (uint32_t)atol(argv[3]);

    if (N <= 0) {
        fprintf(stderr, "Error: N debe ser un entero positivo.\n");
        return EXIT_FAILURE;
    }

    /* ── Configurar número de hilos ── */
    if (num_hilos > 0)
        omp_set_num_threads(num_hilos);

    /* Obtener cuántos hilos se usarán realmente */
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
    printf("    S0=%.1f  K=%.1f  T=%.1f año(s)  r=%.2f  σ=%.2f\n",
           S0, K, T_ANN, R, SIGMA);
    printf("    Pasos por trayectoria: %d (días hábiles)\n", PASOS);
    printf("  Parámetros de simulación:\n");
    printf("    N         = %ld trayectorias\n", N);
    printf("    Hilos OMP = %d\n", hilos_reales);
    printf("    Seed base = %u\n", seed_base);
    printf("───────────────────────────────────────────────────────\n");

    /* ── Reservar memoria para precios finales (VaR) ── */
    double *precios = (double *)malloc((size_t)N * sizeof(double));
    if (!precios) {
        fprintf(stderr, "Error: malloc falló para %ld doubles (%.1f MB).\n",
                N, (double)N * sizeof(double) / 1048576.0);
        return EXIT_FAILURE;
    }

    double dt = T_ANN / (double)PASOS;

    /* ── Inicio de medición con omp_get_wtime() ─────────────────────
     * omp_get_wtime() mide tiempo de pared (wall-clock), no CPU.
     * Es la medición correcta para benchmarking paralelo porque
     * clock() mediría la suma de tiempo de TODOS los hilos.
     * ──────────────────────────────────────────────────────────────── */
    double t_inicio = omp_get_wtime();

    /* ════════════════════════════════════════════════════════════════
     *  LOOP PRINCIPAL PARALELO — O(N·PASOS / p)
     *
     *  #pragma omp parallel for:
     *    Crea un equipo de hilos y divide el rango [0, N) entre ellos.
     *    OpenMP reparte automáticamente las iteraciones.
     *
     *  reduction(+:suma_payoffs):
     *    Cada hilo acumula su suma PARCIAL en una variable privada.
     *    Al finalizar, OpenMP suma todas las parciales de forma segura.
     *    SIN reduction, habría race condition en suma_payoffs += ...
     *
     *  schedule(static):
     *    Divide las N iteraciones en bloques iguales (N/p por hilo).
     *    Adecuado porque todas las iteraciones tienen el mismo costo.
     *
     *  private(local_seed, ST, payoff, z1, z2):
     *    Cada hilo tiene su PROPIA copia de estas variables.
     *    local_seed es el estado RNG exclusivo del hilo → thread-safe.
     *
     *  firstprivate(seed_base):
     *    Cada hilo recibe una COPIA del valor de seed_base.
     *    Luego deriva su propia seed: seed_base + thread_id + 1.
     *    +1 garantiza que ningún hilo tenga seed = 0 (inválido en xorshift).
     * ════════════════════════════════════════════════════════════════ */
    double suma_payoffs = 0.0;

    /* ── Región paralela explícita ────────────────────────────────────
     * Se separa #pragma omp parallel del #pragma omp for para poder
     * calcular local_seed UNA SOLA VEZ por hilo (al entrar a la región),
     * y no en cada iteración del loop.
     *
     * Si usáramos solo "#pragma omp parallel for" con local_seed dentro
     * del for, la seed se recalcularía en cada iteración → todos los
     * llamados a simular_precio_final partirían del mismo estado RNG
     * → precios incorrectos (bug de reinicio de seed).
     * ──────────────────────────────────────────────────────────────── */
    #pragma omp parallel reduction(+:suma_payoffs) shared(precios, N, dt)
    {
        /* ── Seed privada: se calcula UNA VEZ al entrar el hilo ───────
         * omp_get_thread_num() devuelve 0, 1, … (p-1).
         * Multiplicar por el número de Fibonacci-hash (2654435761)
         * distribuye las seeds en el espacio de 32 bits, minimizando
         * correlaciones entre secuencias de hilos distintos.
         * +1 garantiza seed != 0 (xorshift32 con estado 0 es invalido).
         * ─────────────────────────────────────────────────────────── */
        uint32_t local_seed = seed_base
                              + (uint32_t)omp_get_thread_num() * 2654435761u
                              + 1u;

        /* ── Loop paralelo: cada hilo procesa su porción de [0, N) ── */
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            double ST     = simular_precio_final(S0, R, SIGMA, dt, PASOS, &local_seed);
            double payoff = ST - K;
            if (payoff < 0.0) payoff = 0.0;

            precios[i]   = ST;
            suma_payoffs += payoff;
        }
    } /* fin region paralela */

    double t_fin   = omp_get_wtime();
    double tiempo_s = t_fin - t_inicio;

    /* ── Resultados (hilo principal, fuera de región paralela) ── */
    double precio_mc = exp(-R * T_ANN) * (suma_payoffs / (double)N);
    double precio_bs = bs_call_price(S0, K, T_ANN, R, SIGMA);
    double error_rel = fabs(precio_mc - precio_bs) / precio_bs * 100.0;
    double var_5     = calcular_var(precios, N, 0.05);

    double suma_st = 0.0;
    for (long i = 0; i < N; i++) suma_st += precios[i];
    double ret_esp = (suma_st / N - S0) / S0 * 100.0;

    printf("\n  RESULTADOS:\n");
    printf("    Precio call Monte Carlo  : $%8.4f\n", precio_mc);
    printf("    Precio call Black-Scholes: $%8.4f\n", precio_bs);
    printf("    Error relativo           : %8.4f%%\n", error_rel);
    printf("    VaR 5%% (pérdida máx.)   : $%8.4f\n", var_5);
    printf("    Retorno esperado promedio: %8.4f%%\n", ret_esp);
    printf("\n  DESEMPEÑO:\n");
    printf("    Tiempo de ejecución : %.6f s\n", tiempo_s);
    printf("    Hilos utilizados    : %d\n", hilos_reales);
    printf("    Trayectorias/segundo: %.0f\n", (double)N / tiempo_s);
    printf("───────────────────────────────────────────────────────\n");

    if (error_rel < 1.0)
        printf("  [OK] Verificación passed — error < 1%% vs Black-Scholes\n");
    else
        printf("  [WARN] Error relativo alto (%.2f%%) — considerar N mayor\n",
               error_rel);

    /* ── Exportar CSV para Supabase (Fase 4) ── */
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

    free(precios);
    return EXIT_SUCCESS;
}
