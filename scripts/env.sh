#!/usr/bin/env bash
# Resume o ambiente de teste para colar nas issues.
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p') ($(nproc) threads)"
echo "RAM: $(free -g | awk '/Mem/ {print $2}') GB"
echo "SO: $(uname -sr)"
echo "Compilador: $(${CXX:-g++} --version | head -1)"
if command -v nvidia-smi > /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    echo "CUDA: $(nvcc --version | tail -1)"
fi
