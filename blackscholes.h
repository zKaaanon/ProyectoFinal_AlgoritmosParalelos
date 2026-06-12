/*
 * blackscholes.h
 * Fórmula cerrada de Black-Scholes para opciones europeas.
 * Se usa ÚNICAMENTE para verificar que Monte Carlo converge
 * al valor teórico correcto. No forma parte del algoritmo principal.
 *
 * Proyecto: Monte Carlo con OpenMP — Fase 1 (Secuencial)
 * Materia:  Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
 * Entrega:  Viernes 12 de junio de 2026
 */

#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

#include <math.h>

/*
 * norm_cdf — distribución normal acumulada usando erfc de <math.h>.
 * erfc(x) = 1 - erf(x), con erf de alta precisión (error < 1e-15).
 * Fórmula: N(x) = erfc(-x / sqrt(2)) / 2
 */
static double norm_cdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

/*
 * bs_call_price — precio de una opción call europea (Black-Scholes).
 *
 *   S0    precio actual del subyacente
 *   K     precio de ejercicio (strike)
 *   T     tiempo al vencimiento (años)
 *   r     tasa libre de riesgo continua
 *   sigma volatilidad anualizada
 */
static double bs_call_price(double S0, double K, double T,
                             double r,  double sigma) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T)
                / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
}

#endif /* BLACKSCHOLES_H */
