#!/usr/bin/env bash

#  benchmark.sh — Benchmarking sistemático: Monte Carlo secuencial vs OpenMP
#
#  Qué hace:
#    1. Compila ambas versiones (make all)
#    2. Para cada combinación (N, hilos) ejecuta REPS veces
#    3. Calcula promedio, desviación estándar, speedup y eficiencia
#    4. Guarda resultados en benchmarks.csv
#
#  Uso:
#    chmod +x benchmark.sh
#    ./benchmark.sh
#
#  Salida:
#    benchmarks.csv  
#  Materia:  Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón

set -euo pipefail   # Salir si cualquier comando falla

# Configuración del experimento 
REPS=5  # Cuantas veces repetimos cada prueba
TAMANIOS=(100000 1000000 10000000)   #  Tamaños de entrada N: 100K, 1M, 10M trayectorias
HILOS=(1 2 4 8)                 # Números de hilos a probar
SEQ_BIN="./montecarlo_seq"
OMP_BIN="./montecarlo_omp"
CSV_OUT="benchmarks.csv"
SEED=12345

# Colores para que la salida en consola sea legible
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# Funciones auxiliares 

# Extrae el tiempo de ejecución del output del programa
# El programa imprime: "    Tiempo de ejecución : X.XXXXXX s"
extraer_tiempo() {
    grep "Tiempo de ejecuci" | grep -oP '[0-9]+\.[0-9]+'
}

# Calcula promedio de N números pasados como argumentos
promedio() {
    local vals=("$@")
    local suma=0
    for v in "${vals[@]}"; do
        suma=$(echo "$suma + $v" | bc -l)
    done
    echo "scale=6; $suma / ${#vals[@]}" | bc -l
}

# Calcula desviación estándar poblacional para ver que tan estables son los tiempos
desv_std() {
    local vals=("$@")
    local media
    media=$(promedio "${vals[@]}")
    local suma_sq=0
    for v in "${vals[@]}"; do
        local diff
        diff=$(echo "$v - $media" | bc -l)
        suma_sq=$(echo "$suma_sq + ($diff * $diff)" | bc -l)
    done
    echo "scale=6; sqrt($suma_sq / ${#vals[@]})" | bc -l
}

# Verificar dependencias 
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   Benchmark: Monte Carlo secuencial vs OpenMP${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"

for cmd in bc gcc awk; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}[ERROR] Dependencia no encontrada: $cmd${NC}"
        exit 1
    fi
done

# Compilar
echo -e "\n${CYAN}[1/4] Compilando...${NC}"
make all 2>&1 | sed 's/^/  /'
echo -e "${GREEN}  Compilación exitosa${NC}"

# Verificar que los binarios existen
if [[ ! -x "$SEQ_BIN" || ! -x "$OMP_BIN" ]]; then
    echo -e "${RED}[ERROR] Binarios no encontrados. Ejecuta 'make all' primero.${NC}"
    exit 1
fi

# Detectar núcleos disponibles
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "?")
echo -e "\n${CYAN}[2/4] Sistema detectado: ${NCORES} núcleos lógicos${NC}"
echo -e "  Tamaños N : ${TAMANIOS[*]}"
echo -e "  Hilos     : ${HILOS[*]}"
echo -e "  Repeticiones por config: ${REPS}"
TOTAL_RUNS=$(( (${#TAMANIOS[@]} + ${#TAMANIOS[@]} * ${#HILOS[@]}) * REPS ))
echo -e "  Total de ejecuciones: ${TOTAL_RUNS}"

# Crear CSV con su encabezado
echo -e "\n${CYAN}[3/4] Ejecutando experimentos...${NC}"
echo "version,N,hilos,rep,tiempo_s" > "$CSV_OUT"

# Guardamos tiempos base secuenciales por N (para speedup)
declare -A T_SEQ_BASE   # T_SEQ_BASE[N] = tiempo promedio secuencial

RUN=0


#  BLOQUE 1: Versión SECUENCIAL (hilos = 1, sin OpenMP)

echo -e "\n  ${BOLD}── Secuencial ──${NC}"

for N in "${TAMANIOS[@]}"; do
    tiempos=()
    printf "  N=%-10s [" "$N"

    for rep in $(seq 1 $REPS); do
        t=$("$SEQ_BIN" "$N" "$SEED" 2>/dev/null | extraer_tiempo)
        tiempos+=("$t")
        echo "secuencial,$N,1,$rep,$t" >> "$CSV_OUT"
        printf "."
        RUN=$((RUN + 1))
    done
    printf "] "

    prom=$(promedio "${tiempos[@]}")
    std=$(desv_std "${tiempos[@]}")
    T_SEQ_BASE[$N]=$prom
    printf "${GREEN}avg=%.3fs  std=%.3fs${NC}\n" "$prom" "$std"
done


#  BLOQUE 2: Versión PARALELA (distintos hilos)

echo -e "\n  ${BOLD}── Paralelo (OpenMP) ──${NC}"

for N in "${TAMANIOS[@]}"; do
    echo -e "  N=${N}"
    for P in "${HILOS[@]}"; do
        tiempos=()
        printf "    p=%-3s [" "$P"

        for rep in $(seq 1 $REPS); do
            t=$("$OMP_BIN" "$N" "$P" "$SEED" 2>/dev/null | extraer_tiempo)
            tiempos+=("$t")
            echo "paralelo,$N,$P,$rep,$t" >> "$CSV_OUT"
            printf "."
            RUN=$((RUN + 1))
        done
        printf "] "

        prom=$(promedio "${tiempos[@]}")
        std=$(desv_std "${tiempos[@]}")
        t_seq="${T_SEQ_BASE[$N]}"
        speedup=$(echo "scale=4; $t_seq / $prom" | bc -l)
        eficiencia=$(echo "scale=4; $speedup / $P" | bc -l)

        printf "${GREEN}avg=%.3fs  std=%.3fs  S=%.2f  E=%.2f${NC}\n" \
               "$prom" "$std" "$speedup" "$eficiencia"
    done
done


#  BLOQUE 3: Generar CSV de resumen con métricas calculadas

SUMMARY_CSV="benchmarks_resumen.csv"
echo "version,N,hilos,tiempo_avg_s,tiempo_std_s,speedup,eficiencia" > "$SUMMARY_CSV"

echo -e "\n${CYAN}[4/4] Calculando métricas de resumen...${NC}"

# Procesar los datos de la version secuencial
for N in "${TAMANIOS[@]}"; do
    # Filtrar filas del csv crudo que corresponden a este N y version secuencial
    tiempos=()
    while IFS=',' read -r ver n p rep t; do
        [[ "$ver" == "secuencial" && "$n" == "$N" ]] && tiempos+=("$t")
    done < <(tail -n +2 "$CSV_OUT")

    prom=$(promedio "${tiempos[@]}")
    std=$(desv_std "${tiempos[@]}")
    echo "secuencial,$N,1,${prom},${std},1.0000,1.0000" >> "$SUMMARY_CSV"
done

# Procesar los datos en paralelo y calcular speedup y eficiencia
for N in "${TAMANIOS[@]}"; do
    t_seq="${T_SEQ_BASE[$N]}"
    for P in "${HILOS[@]}"; do
        tiempos=()
        while IFS=',' read -r ver n p rep t; do
            [[ "$ver" == "paralelo" && "$n" == "$N" && "$p" == "$P" ]] && tiempos+=("$t")
        done < <(tail -n +2 "$CSV_OUT")

        prom=$(promedio "${tiempos[@]}")
        std=$(desv_std "${tiempos[@]}")
        speedup=$(echo "scale=6; $t_seq / $prom" | bc -l)
        eficiencia=$(echo "scale=6; $speedup / $P" | bc -l)
        echo "paralelo,$N,$P,${prom},${std},${speedup},${eficiencia}" >> "$SUMMARY_CSV"
    done
done

# Imprimir tabla de resumen
echo -e "\n${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   RESUMEN DE RESULTADOS${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
printf "%-12s %-12s %-6s %-10s %-10s %-8s %-8s\n" \
       "Version" "N" "Hilos" "Avg(s)" "Std(s)" "Speedup" "Efic."
echo "──────────────────────────────────────────────────────"

while IFS=',' read -r ver N p avg std spd efi; do
    [[ "$ver" == "version" ]] && continue   # skip header
    printf "%-12s %-12s %-6s %-10s %-10s %-8s %-8s\n" \
           "$ver" "$N" "$p" \
           "$(printf '%.4f' $avg)" \
           "$(printf '%.4f' $std)" \
           "$(printf '%.3f' $spd)" \
           "$(printf '%.3f' $efi)"
done < "$SUMMARY_CSV"

echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] CSVs generados:${NC}"
echo -e "     ${CYAN}${CSV_OUT}${NC}          — datos crudos (todas las repeticiones)"
echo -e "     ${CYAN}${SUMMARY_CSV}${NC}  — métricas agregadas (para Power BI)"
echo ""
echo -e "${YELLOW}Nota: El speedup real depende de los núcleos físicos de tu CPU.${NC}"
echo -e "${YELLOW}En sistemas con 1 núcleo (WSL sin afinidad) S(p) ≈ 1.${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
