#!/usr/bin/env bash
# Compila no head-node do Cluster Franky e roda um comando no no de GPU.
# Os nos de computacao nao tem headers do sistema, entao a compilacao fica no head-node.
# Uso (no head-node, na raiz do repo): scripts/franky.sh './scripts/bench.sh ./barenn_cuda 10'
set -euo pipefail
CUDA_BIN=/opt/ohpc/pub/spack/opt/spack/linux-skylake/cuda-12.8.1-hk7c4l6fpeucmtbivmr7kn73dnkarzyt/bin
module load cuda/12.8.1
# Os nos de GPU tem i9-14900 (Raptor Lake); o head-node tem outra CPU.
make -s ARCH=raptorlake all cuda
srun --partition="${PARTICAO:-sunny_gpu}" --gres=gpu:1 --mem=2G --cpus-per-task=8 \
    bash -c "export PATH=$CUDA_BIN:\$PATH OMP_NUM_THREADS=8; $1"
