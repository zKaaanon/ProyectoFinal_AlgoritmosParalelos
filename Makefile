# ═══════════════════════════════════════════════════════════════════
#  Makefile — Monte Carlo con OpenMP
#  Proyecto: Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
#  Entrega: Viernes 12 de junio de 2026
# ═══════════════════════════════════════════════════════════════════

CC      = gcc
CFLAGS  = -O2 -std=c11 -Wall -Wextra
OMPFLAG = -fopenmp
LIBS    = -lm

# Archivos fuente
SEQ_SRC = montecarlo_seq.c
OMP_SRC = montecarlo_omp.c

# Binarios
SEQ_BIN = montecarlo_seq
OMP_BIN = montecarlo_omp

# ── Targets principales ────────────────────────────────────────────
.PHONY: all seq omp clean test

all: seq omp
	@echo ""
	@echo "  [OK] Compilación exitosa: $(SEQ_BIN) y $(OMP_BIN)"

seq: $(SEQ_BIN)

omp: $(OMP_BIN)

$(SEQ_BIN): $(SEQ_SRC) blackscholes.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
	@echo "  [SEQ] Compilado: $@"

$(OMP_BIN): $(OMP_SRC) blackscholes.h
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $< $(LIBS)
	@echo "  [OMP] Compilado: $@ (con OpenMP)"

# ── Test rápido: compara precios seq vs paralelo ───────────────────
test: all
	@echo ""
	@echo "─── Test: versión secuencial (N=100000) ───"
	./$(SEQ_BIN) 100000
	@echo ""
	@echo "─── Test: versión paralela 1 hilo (N=100000) ───"
	./$(OMP_BIN) 100000 1
	@echo ""
	@echo "─── Test: versión paralela 4 hilos (N=100000) ───"
	./$(OMP_BIN) 100000 4

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) resultados_seq.csv resultados_omp.csv
	@echo "  [CLEAN] Binarios y CSVs eliminados"
