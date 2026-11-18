#!/usr/bin/env bash
# Compara a saida do binario com a referencia do codigo legado.
# Perdas por epoca devem ficar dentro de TOL e os acertos dentro de ACC_TOL imagens.
# Uso: scripts/validate.sh ./barenn [referencia]
set -euo pipefail
export LC_ALL=C
BIN=${1:-./barenn}
REF=${2:-$(dirname "$0")/../ref/saida_referencia.txt}
TOL=${TOL:-1e-3}
ACC_TOL=${ACC_TOL:-9}

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
"$BIN" > "$OUT"

if diff <(grep -v 'Tempo de treino' "$REF") <(grep -v 'Tempo de treino' "$OUT") > /dev/null; then
    echo "OK: saida identica a referencia (bit a bit)"
    exit 0
fi

awk -v tol="$TOL" -v acc_tol="$ACC_TOL" '
    FNR == 1 { arq++ }
    /perda media/ { perda[arq, ++np[arq]] = $NF }
    /^Acuracia/  { acertos[arq] = $4 }
    END {
        ok = 1
        if (np[1] != np[2]) { print "FALHA: numero de epocas difere"; exit 1 }
        for (i = 1; i <= np[1]; i++) {
            d = perda[1, i] - perda[2, i]; if (d < 0) d = -d
            if (d > maxd) maxd = d
            if (d > tol) ok = 0
        }
        da = acertos[1] - acertos[2]; if (da < 0) da = -da
        if (da > acc_tol) ok = 0
        printf "%s: max |dperda| = %.6f (tol %s), acertos %d vs ref %d (tol %d)\n",
               ok ? "OK" : "FALHA", maxd, tol, acertos[2], acertos[1], acc_tol
        exit !ok
    }' "$REF" "$OUT"
