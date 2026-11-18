#!/usr/bin/env bash
# Throughput com M modelos independentes: GPU (1 bloco por modelo, um lancamento)
# contra CPU (M processos ./barenn em paralelo, um por nucleo).
# Uso: scripts/escala.sh "1 8 18 36 72 108 144 288"
set -euo pipefail
export LC_ALL=C
NUCLEOS=${NUCLEOS:-$(nproc)}
echo "modelos gpu_treino_s cpu_parede_s (cpu com $NUCLEOS processos simultaneos)"
for M in ${1:-1 8 18 36 72 108 144 288}; do
    g=$(BARENN_MODELOS=$M ./barenn_cuda 2>/dev/null | awk '/Tempo de treino/ {print $4}')
    ini=$(date +%s.%N)
    seq "$M" | OMP_NUM_THREADS=1 xargs -P "$NUCLEOS" -I{} ./barenn > /dev/null
    fim=$(date +%s.%N)
    echo "$M $g $(awk -v a="$ini" -v b="$fim" "BEGIN {printf \"%.4f\", b - a}")"
done
