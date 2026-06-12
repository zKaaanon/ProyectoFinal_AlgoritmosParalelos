/*
 * montecarlo_seq.c
 * Versión SECUENCIAL — Monte Carlo para valoración de opción call europea.
 *
 * ── Modelo financiero ──────────────────────────────────────────────────────
 * Movimiento Browniano Geométrico (GBM) bajo medida de riesgo neutral:
 *
 *   S(t + Δt) = S(t) · exp( (r − σ²/2)·Δt + σ·ε·√Δt )
 *
 *   donde ε ~ N(0,1)  (generado con método Box-Muller + xorshift32)
 *   y r es la tasa libre de riesgo (medida de riesgo neutral, NO el drift μ).
 *
 * Precio de la opción call europea al vencimiento T:
 *   C = e^(−rT) · E[ max(S_T − K, 0) ]
 *
 * ── Complejidad ────────────────────────────────────────────────────────────
 *   Temporal:  O(N · PASOS)    N = trayectorias, PASOS = pasos de tiempo
 *   Espacial:  O(N)            se almacenan precios finales para VaR
 *
 * ── Compilación ───────────────────────────────────────────────────────────
 *   gcc -O2 -std=c11 -Wall -Wextra -o montecarlo_seq montecarlo_seq.c -lm
 *
 * ── Uso ───────────────────────────────────────────────────────────────────
 *   ./montecarlo_seq [N] [seed]
 *   N    = número de simulaciones  (default: 1 000 000)
 *   seed = semilla del RNG          (default: 12345)
 *
 * ── Proyecto ──────────────────────────────────────────────────────────────
 *   Monte Carlo con OpenMP — Fase 1 (Secuencial)
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

#include "blackscholes.h"

/* ══════════════════════════════════════════════════════════════════
 *  PARÁMETROS FINANCIEROS — fijos para todo el proyecto
 *  (sec. + paralelo + BD + Power BI). Cambiar aquí es suficiente.
 * ══════════════════════════════════════════════════════════════════ */
#define S0      100.0   /* Precio inicial del activo                  */
#define K       105.0   /* Strike (precio de ejercicio)               */
#define T_ANN     1.0   /* Tiempo al vencimiento (años)               */
#define R         0.05  /* Tasa libre de riesgo anual (risk-neutral)  */
#define SIGMA     0.20  /* Volatilidad anual                          */
#define PASOS     252   /* Pasos de tiempo (días hábiles / año)       */

/* ══════════════════════════════════════════════════════════════════
 *  PARÁMETROS DE SIMULACIÓN POR DEFECTO
 * ══════════════════════════════════════════════════════════════════ */
#define N_DEFAULT    1000000L
#define SEED_DEFAULT 12345u


/* ──────────────────────────────────────────────────────────────────
 * xorshift32 — generador de números pseudoaleatorios de 32 bits.
 *
 * Algoritmo de G. Marsaglia (2003). Período 2^32 − 1.
 * Se usa uint32_t para garantizar aritmética correcta en cualquier
 * plataforma (evita desbordamiento silencioso de unsigned long en 64-bit).
 *
 * Complejidad: O(1) por llamada.
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
 * box_muller — genera DOS muestras independientes N(0,1).
 *
 * Método de Box-Muller (1958):
 *   Z1 = √(−2 ln U1) · cos(2π U2)
 *   Z2 = √(−2 ln U1) · sin(2π U2)
 *   con U1, U2 ~ Uniforme(0,1) independientes.
 *
 * Genera pares para aprovechar ambas muestras en el loop principal.
 * Parámetros de salida: z1, z2 escritos por referencia.
 *
 * Complejidad: O(1)
 * ────────────────────────────────────────────────────────────────── */
static void box_muller(uint32_t *seed, double *z1, double *z2) {
    /* +0.5 antes de dividir garantiza U1, U2 ∈ (0, 1) — evita log(0) */
    double u1 = (xorshift32(seed) + 0.5) / 4294967296.0;
    double u2 = (xorshift32(seed) + 0.5) / 4294967296.0;

    double mag = sqrt(-2.0 * log(u1));
    *z1 = mag * cos(2.0 * M_PI * u2);
    *z2 = mag * sin(2.0 * M_PI * u2);
}


/* ──────────────────────────────────────────────────────────────────
 * simular_precio_final — simula UNA trayectoria GBM.
 *
 * Evolución discreta bajo medida de riesgo neutral:
 *   S_{i+1} = S_i · exp( (r − σ²/2)·Δt + σ·ε_i·√Δt )
 *
 * Los términos constantes (drift, vol_sqrt) se calculan fuera del
 * loop para evitar operaciones repetidas.
 *
 * Box-Muller genera pares ε, así que el loop avanza de 2 en 2.
 *
 * Parámetros:
 *   s0   precio inicial
 *   r    tasa libre de riesgo (medida risk-neutral)
 *   sigma volatilidad
 *   dt   tamaño de paso temporal
 *   pasos número de pasos
 *   seed puntero a estado del RNG (modificado in-place)
 *
 * Retorna: precio final S_T
 * Complejidad: O(pasos)
 * ────────────────────────────────────────────────────────────────── */
static double simular_precio_final(double s0,   double r,
                                    double sigma, double dt,
                                    int pasos,   uint32_t *seed) {
    /* Precalcular términos del exponente — O(1), fuera del loop */
    double drift    = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt = sigma * sqrt(dt);

    double S = s0;
    double z1, z2;
    int i = 0;

    /* Loop principal: avanza de 2 en 2 aprovechando el par Box-Muller */
    while (i < pasos - 1) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);   /* paso i     */
        S *= exp(drift + vol_sqrt * z2);   /* paso i + 1 */
        i += 2;
    }
    /* Paso final si PASOS es impar */
    if (i < pasos) {
        box_muller(seed, &z1, &z2);
        S *= exp(drift + vol_sqrt * z1);
    }

    return S;
}


/* ──────────────────────────────────────────────────────────────────
 * cmp_double — comparador para qsort (orden ascendente).
 * ────────────────────────────────────────────────────────────────── */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}


/* ──────────────────────────────────────────────────────────────────
 * calcular_var — Value at Risk al nivel dado.
 *
 * Ordena los precios finales (O(n log n)) y toma el percentil
 * indicado. El VaR se expresa como pérdida respecto a S0.
 *
 * Modifica el arreglo 'precios' in-place (lo ordena).
 * Complejidad: O(n log n)
 * ────────────────────────────────────────────────────────────────── */
static double calcular_var(double *precios, long n, double nivel) {
    qsort(precios, (size_t)n, sizeof(double), cmp_double);
    long idx = (long)(nivel * (double)n);
    if (idx < 0)   idx = 0;
    if (idx >= n)  idx = n - 1;
    return S0 - precios[idx];   /* pérdida positiva = S0 > S_T */
}


/* ══════════════════════════════════════════════════════════════════
 *  PROGRAMA PRINCIPAL
 * ══════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    /* ── Parámetros de línea de comandos ── */
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
    printf("    S0=%.1f  K=%.1f  T=%.1f año(s)  r=%.2f  σ=%.2f\n",
           S0, K, T_ANN, R, SIGMA);
    printf("    Pasos por trayectoria: %d (días hábiles)\n", PASOS);
    printf("  Parámetros de simulación:\n");
    printf("    N    = %ld trayectorias\n", N);
    printf("    Seed = %u\n", seed);
    printf("───────────────────────────────────────────────────────\n");

    /* ── Reservar memoria para precios finales (VaR) ── */
    double *precios = (double *)malloc((size_t)N * sizeof(double));
    if (!precios) {
        fprintf(stderr, "Error: malloc falló para %ld doubles (%.1f MB).\n",
                N, (double)N * sizeof(double) / 1048576.0);
        return EXIT_FAILURE;
    }

    /* ── Constante de tiempo ── */
    double dt = T_ANN / (double)PASOS;

    /* ── Inicio de medición ── */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ════════════════════════════════════════════════════════════
     *  LOOP PRINCIPAL DE MONTE CARLO ── O(N · PASOS)
     *
     *  Para cada trayectoria i = 0 … N−1:
     *    1. Simular S_T con GBM discreto bajo medida risk-neutral
     *    2. Calcular payoff: max(S_T − K, 0)
     *    3. Acumular payoff para el promedio
     *
     *  Cada iteración es COMPLETAMENTE INDEPENDIENTE:
     *  no hay dependencia de datos entre trayectorias.
     *  → Candidato natural a paralelización con OpenMP (Fase 2).
     *
     *  La única "dependencia" es el estado del RNG (seed), que en
     *  la Fase 2 se volverá privado por hilo.
     * ════════════════════════════════════════════════════════════ */
    double suma_payoffs = 0.0;

    for (long i = 0; i < N; i++) {
        double ST = simular_precio_final(S0, R, SIGMA, dt, PASOS, &seed);

        precios[i] = ST;                /* guardar para VaR */

        double payoff = ST - K;
        if (payoff < 0.0) payoff = 0.0; /* max(S_T − K, 0) */

        suma_payoffs += payoff;
    }

    /* ── Fin de medición ── */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tiempo_s = (t1.tv_sec  - t0.tv_sec) +
                      (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* ── Resultados ── */
    double precio_mc = exp(-R * T_ANN) * (suma_payoffs / (double)N);
    double precio_bs = bs_call_price(S0, K, T_ANN, R, SIGMA);
    double error_rel = fabs(precio_mc - precio_bs) / precio_bs * 100.0;
    double var_5     = calcular_var(precios, N, 0.05);

    /* Retorno esperado promedio (post-sort, recalcular media) */
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
    printf("    Trayectorias/segundo: %.0f\n", (double)N / tiempo_s);
    printf("───────────────────────────────────────────────────────\n");

    if (error_rel < 1.0)
        printf("  [OK] Verificación passed — error < 1%% vs Black-Scholes\n");
    else
        printf("  [WARN] Error relativo alto (%.2f%%) — considerar N mayor\n",
               error_rel);

    /* ── Exportar CSV (Fase 4: Supabase) ── */
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

    free(precios);
    return EXIT_SUCCESS;
}
