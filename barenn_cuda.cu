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
        float* d_imagem;
        float* d_gradiente;
        CUDA_CHECK(cudaMalloc(&d_imagem, sizeof(float) * entradas));
        CUDA_CHECK(cudaMalloc(&d_gradiente, sizeof(float) * classes));

        array<float, classes> escores{};
        array<float, classes> gradiente{};
        array<double, epocas> perdas{};

        // Port direto do laço da CPU: cada amostra envia a imagem, roda os
        // kernels camada a camada, traz os escores para calcular a perda no
        // host e devolve o gradiente para a GPU.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (const auto& amostra : treino) {
                CUDA_CHECK(cudaMemcpy(d_imagem, amostra.imagem.data(),
                                      sizeof(float) * entradas, cudaMemcpyHostToDevice));
                const float* saida = forward(g3, forward(g2, forward(g1, d_imagem)));
                CUDA_CHECK(cudaMemcpy(escores.data(), saida, sizeof(float) * classes,
                                      cudaMemcpyDeviceToHost));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);
                CUDA_CHECK(cudaMemcpy(d_gradiente, gradiente.data(),
                                      sizeof(float) * classes, cudaMemcpyHostToDevice));

                const float* grad3 = backward(g3, d_gradiente);
                const float* grad2 = backward(g2, grad3);
                backward(g1, grad2, false);
            }
            perdas[epoca] = perda_total / treino.size();
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
        cudaFree(d_imagem);
        cudaFree(d_gradiente);

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
