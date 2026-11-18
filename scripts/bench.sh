#!/usr/bin/env bash
# Executa o binario N vezes e resume um tempo impresso (media, desvio, min, max).
# Uso: scripts/bench.sh ./barenn [N] [etapa]   (etapa: treino, leitura, avaliacao, total)
set -euo pipefail
export LC_ALL=C
BIN=${1:-./barenn}
N=${2:-10}
ETAPA=${3:-treino}
for _ in $(seq "$N"); do
    "$BIN" | awk -v e="$ETAPA" '/^Tempo/ && $(NF-1) ~ /^[0-9.]+$/ && $0 ~ e {print $(NF-1)}'
done | awk -v n="$N" '
    { t[NR] = $1; s += $1; if (NR == 1 || $1 < mn) mn = $1; if ($1 > mx) mx = $1 }
    END {
        m = s / NR
        for (i = 1; i <= NR; i++) v += (t[i] - m) ^ 2
        dp = NR > 1 ? sqrt(v / (NR - 1)) : 0
        printf "runs=%d media=%.4f s desvio=%.4f s min=%.4f s max=%.4f s\n", NR, m, dp, mn, mx
    }'
