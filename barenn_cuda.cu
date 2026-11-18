// Versão GPU do treino. Leitura dos dados, inicialização dos pesos e
// avaliação reaproveitam o código da CPU (barenn.hpp); só o laço de treino
// roda na GPU.
#include "barenn.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(chamada)                                                    \
    do {                                                                       \
        cudaError_t erro_cuda = (chamada);                                     \
        if (erro_cuda != cudaSuccess) {                                        \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro_cuda) + \
                                " em " #chamada);                              \
        }                                                                      \
    } while (0)

namespace BarennGPU {

using Barenn::classes;
using Barenn::entradas;

// Uma camada na GPU. Os pesos usam o mesmo layout transposto da CPU:
// w[j * out + i] liga a entrada j ao neurônio i.
struct CamadaGPU {
    int in, out;
    bool final;
    float* w;      // in * out
    float* b;      // out
    float* z;      // soma antes da ativação (out)
    float* a;      // saída após a ativação (out)
    float* grad;   // gradiente em relação à entrada (in)
    const float* x = nullptr;  // entrada do último forward
};

// Um thread por neurônio de saída: z = b + soma em j de x[j] * w[j][i].
// A soma segue a mesma ordem da CPU.
__global__ void forward_kernel(const float* x, const float* w, const float* b,
                               float* z, float* a, int in, int out, bool final) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out) {
        return;
    }
    float soma = b[i];
    for (int j = 0; j < in; ++j) {
        soma += x[j] * w[j * out + i];
    }
    z[i] = soma;
    // Mesmo critério de std::max(0.0f, soma).
    a[i] = final ? soma : (0.0f < soma ? soma : 0.0f);
}

// Um thread por entrada j: calcula grad_in[j] com os pesos antigos da linha j
// e depois aplica o SGD nessa linha. Cada thread só escreve na própria linha,
// então não há condição de corrida. Os threads j < out atualizam o viés.
__global__ void backward_kernel(const float* grad_out, const float* z,
                                const float* x, float* w, float* b,
                                float* grad_in, int in, int out, bool final,
                                bool calcular_grad_in, float taxa) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= in) {
        return;
    }
    float* linha = w + j * out;
    if (calcular_grad_in) {
        float soma = 0;
        for (int i = 0; i < out; ++i) {
            const float d = final ? grad_out[i]
                                  : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
            soma += d * linha[i];
        }
        grad_in[j] = soma;
    }
    const float xj = x[j];
    for (int i = 0; i < out; ++i) {
        const float d = final ? grad_out[i]
                              : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        linha[i] -= taxa * (d * xj);
    }
    if (j < out) {
        const float d = final ? grad_out[j]
                              : grad_out[j] * (z[j] > 0.0f ? 1.0f : 0.0f);
        b[j] -= taxa * d;
    }
}

// Perda e gradiente de uma amostra, calculados na GPU. Um thread por classe
// calcula o erro; o thread 0 soma os termos na mesma ordem da CPU.
__global__ void perda_kernel(const float* escores, const int* rotulos, int amostra,
                             float* gradiente, float* perdas) {
    __shared__ float termos[classes];
    const int i = threadIdx.x;
    if (i < classes) {
        const float erro = escores[i] - (i == rotulos[amostra] ? 1.0f : 0.0f);
        gradiente[i] = erro;
        termos[i] = 0.5f * erro * erro;
    }
    __syncthreads();
    if (i == 0) {
        float total = 0;
        for (int c = 0; c < classes; ++c) {
            total += termos[c];
        }
        perdas[amostra] = total;
    }
}

constexpr int threads_por_bloco = 64;

int blocos(int n) {
    return (n + threads_por_bloco - 1) / threads_por_bloco;
}

CamadaGPU criar(int in, int out, bool final, const float* w_host, const float* b_host) {
    CamadaGPU c{in, out, final, nullptr, nullptr, nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&c.w, sizeof(float) * in * out));
    CUDA_CHECK(cudaMalloc(&c.b, sizeof(float) * out));
    CUDA_CHECK(cudaMalloc(&c.z, sizeof(float) * out));
    CUDA_CHECK(cudaMalloc(&c.a, sizeof(float) * out));
    CUDA_CHECK(cudaMalloc(&c.grad, sizeof(float) * in));
    CUDA_CHECK(cudaMemcpy(c.w, w_host, sizeof(float) * in * out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.b, b_host, sizeof(float) * out, cudaMemcpyHostToDevice));
    return c;
}

void baixar(const CamadaGPU& c, float* w_host, float* b_host) {
    CUDA_CHECK(cudaMemcpy(w_host, c.w, sizeof(float) * c.in * c.out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_host, c.b, sizeof(float) * c.out, cudaMemcpyDeviceToHost));
}

void liberar(CamadaGPU& c) {
    cudaFree(c.w);
    cudaFree(c.b);
    cudaFree(c.z);
    cudaFree(c.a);
    cudaFree(c.grad);
}

const float* forward(CamadaGPU& c, const float* x) {
    c.x = x;
    forward_kernel<<<blocos(c.out), threads_por_bloco>>>(x, c.w, c.b, c.z, c.a,
                                                         c.in, c.out, c.final);
    CUDA_CHECK(cudaGetLastError());
    return c.a;
}

const float* backward(CamadaGPU& c, const float* grad_out, bool calcular_grad_in = true) {
    backward_kernel<<<blocos(c.in), threads_por_bloco>>>(
        grad_out, c.z, c.x, c.w, c.b, c.grad, c.in, c.out, c.final,
        calcular_grad_in, Barenn::taxa);
    CUDA_CHECK(cudaGetLastError());
    return c.grad;
}

} // namespace BarennGPU

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        using namespace BarennGPU;
        using Barenn::epocas;

        const auto inicio_total = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_total;

        // Os pesos iniciais vêm das camadas da CPU (mesma semente e ordem).
        auto l1 = make_unique<Barenn::Camada1>();
        auto l2 = make_unique<Barenn::Camada2>();
        auto l3 = make_unique<Barenn::Camada3>();

        // Inicializa o contexto CUDA fora da medição do treino.
        CUDA_CHECK(cudaFree(nullptr));
        CamadaGPU g1 = criar(entradas, 32, false, l1->pesos(), l1->vies());
        CamadaGPU g2 = criar(32, 16, false, l2->pesos(), l2->vies());
        CamadaGPU g3 = criar(16, classes, true, l3->pesos(), l3->vies());
        // Conjunto de treino inteiro na GPU, copiado uma única vez.
        const int n = static_cast<int>(treino.size());
        vector<float> imagens(static_cast<size_t>(n) * entradas);
        vector<int> rotulos(n);
        for (int k = 0; k < n; ++k) {
            copy(treino[k].imagem.begin(), treino[k].imagem.end(),
                 imagens.begin() + static_cast<size_t>(k) * entradas);
            rotulos[k] = treino[k].rotulo;
        }
        float* d_imagens;
        int* d_rotulos;
        float* d_gradiente;
        float* d_perdas;
        CUDA_CHECK(cudaMalloc(&d_imagens, sizeof(float) * imagens.size()));
        CUDA_CHECK(cudaMalloc(&d_rotulos, sizeof(int) * n));
        CUDA_CHECK(cudaMalloc(&d_gradiente, sizeof(float) * classes));
        CUDA_CHECK(cudaMalloc(&d_perdas, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(d_imagens, imagens.data(), sizeof(float) * imagens.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rotulos, rotulos.data(), sizeof(int) * n,
                              cudaMemcpyHostToDevice));

        vector<float> perdas_amostra(n);
        array<double, epocas> perdas{};

        // Nenhuma cópia dentro do laço de amostras: o host só enfileira os
        // kernels no stream e a GPU executa na ordem. A perda de cada amostra
        // fica no device e volta uma vez por época.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            for (int k = 0; k < n; ++k) {
                const float* x = d_imagens + static_cast<size_t>(k) * entradas;
                const float* saida = forward(g3, forward(g2, forward(g1, x)));
                perda_kernel<<<1, 32>>>(saida, d_rotulos, k, d_gradiente, d_perdas);
                CUDA_CHECK(cudaGetLastError());

                const float* grad3 = backward(g3, d_gradiente);
                const float* grad2 = backward(g2, grad3);
                backward(g1, grad2, false);
            }
            // Soma em double e na ordem das amostras, como na CPU.
            CUDA_CHECK(cudaMemcpy(perdas_amostra.data(), d_perdas, sizeof(float) * n,
                                  cudaMemcpyDeviceToHost));
            double perda_total = 0;
            for (int k = 0; k < n; ++k) {
                perda_total += perdas_amostra[k];
            }
            perdas[epoca] = perda_total / n;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Traz os pesos treinados de volta e avalia com o código da CPU.
        baixar(g1, l1->pesos(), l1->vies());
        baixar(g2, l2->pesos(), l2->vies());
        baixar(g3, l3->pesos(), l3->vies());
        liberar(g1);
        liberar(g2);
        liberar(g3);
        cudaFree(d_imagens);
        cudaFree(d_rotulos);
        cudaFree(d_gradiente);
        cudaFree(d_perdas);

        Barenn::imprimir_treino(treino.size(), perdas, tempo.count());
        const auto inicio_avaliacao = chrono::steady_clock::now();
        Barenn::avaliar(*l1, *l2, *l3, teste);
        const auto fim = chrono::steady_clock::now();
        const chrono::duration<double> tempo_avaliacao = fim - inicio_avaliacao;
        const chrono::duration<double> tempo_total = fim - inicio_total;

        cout << fixed << setprecision(4)
             << "\nTempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n"
             << "Tempo total " << tempo_total.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
