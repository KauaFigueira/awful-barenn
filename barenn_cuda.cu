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

constexpr int oculta1 = 32;
constexpr int oculta2 = 16;
// Um thread por entrada da maior camada.
constexpr int threads = entradas;

// Distância entre linhas de pesos na shared memory (layout transposto da
// CPU: w[j * Stride + i] liga a entrada j ao neurônio i). O stride ímpar
// (padding de uma palavra) faz os threads do backward, que percorrem linhas
// diferentes, caírem em bancos diferentes da shared memory.
constexpr int stride1 = oculta1 + 1;
constexpr int stride2 = oculta2 + 1;
constexpr int stride3 = classes + 1;

// Um thread por neurônio: z = b + soma em j de x[j] * w[j][i], na ordem da CPU.
template <int In, int Out, int Stride, bool Final>
__device__ void forward_camada(const float* x, const float* w, const float* b,
                               float* z, float* a) {
    const int i = threadIdx.x;
    if (i < Out) {
        float soma = b[i];
        #pragma unroll
        for (int j = 0; j < In; ++j) {
            soma += x[j] * w[j * Stride + i];
        }
        z[i] = soma;
        // Mesmo critério de std::max(0.0f, soma).
        a[i] = Final ? soma : (0.0f < soma ? soma : 0.0f);
    }
}

// Um thread por entrada j: calcula grad_in[j] com os pesos antigos da linha j
// e já devolve o delta da camada anterior (grad_in * ReLU'(z_anterior)).
// Depois aplica o SGD na própria linha; threads j < Out atualizam o viés.
template <int In, int Out, int Stride, bool CalcularGradIn>
__device__ void backward_camada(const float* delta, const float* x, float* w,
                                float* b, const float* z_anterior,
                                float* delta_anterior, float taxa) {
    const int j = threadIdx.x;
    if (j < In) {
        float* linha = w + j * Stride;
        if (CalcularGradIn) {
            float soma = 0;
            #pragma unroll
            for (int i = 0; i < Out; ++i) {
                soma += delta[i] * linha[i];
            }
            delta_anterior[j] = soma * (z_anterior[j] > 0.0f ? 1.0f : 0.0f);
        }
        const float xj = x[j];
        #pragma unroll
        for (int i = 0; i < Out; ++i) {
            linha[i] -= taxa * (delta[i] * xj);
        }
    }
    if (j < Out) {
        b[j] -= taxa * delta[j];
    }
}

// Copia uma matriz [In][Out] contígua da memória global para a shared
// memory com distância Stride entre linhas (e de volta, se Ida = false).
template <int In, int Out, int Stride, bool Ida>
__device__ void copiar_pesos(float* global, float* shared) {
    for (int k = threadIdx.x; k < In * Out; k += blockDim.x) {
        const int j = k / Out;
        const int i = k % Out;
        if (Ida) {
            shared[j * Stride + i] = global[k];
        } else {
            global[k] = shared[j * Stride + i];
        }
    }
}

// Com -DBARENN_PERFIL, o thread 0 acumula os ciclos (clock64) gastos em cada
// fase do treino. Como cada fase termina num __syncthreads(), o tempo medido
// pelo thread 0 é o tempo da fase para o bloco inteiro.
constexpr int fases = 8;
#ifdef BARENN_PERFIL
#define MARCA(fase)                                                            \
    if (t == 0) {                                                              \
        const long long agora = clock64();                                     \
        ciclos[fase] += agora - antes;                                         \
        antes = agora;                                                         \
    }
#else
#define MARCA(fase)
#endif

struct Parametros {
    float* w1; float* b1;
    float* w2; float* b2;
    float* w3; float* b3;
};

// Treino inteiro em um único lançamento de 1 bloco. Toda a rede fica na
// shared memory; as fases de cada amostra são separadas por __syncthreads().
__global__ void __launch_bounds__(threads, 1)
treinar_kernel(Parametros p, const float* imagens, const int* rotulos, int n,
               int epocas, float taxa, float* perdas, long long* perfil) {
    __shared__ float w1[entradas * stride1], b1[oculta1];
    __shared__ float w2[oculta1 * stride2], b2[oculta2];
    __shared__ float w3[oculta2 * stride3], b3[classes];
    __shared__ float x[entradas];
    __shared__ float z1[oculta1], a1[oculta1], delta1[oculta1];
    __shared__ float z2[oculta2], a2[oculta2], delta2[oculta2];
    __shared__ float z3[classes], a3[classes], delta3[classes];
    __shared__ float termos[classes];

    const int t = threadIdx.x;
    copiar_pesos<entradas, oculta1, stride1, true>(p.w1, w1);
    copiar_pesos<oculta1, oculta2, stride2, true>(p.w2, w2);
    copiar_pesos<oculta2, classes, stride3, true>(p.w3, w3);
    if (t < oculta1) b1[t] = p.b1[t];
    if (t < oculta2) b2[t] = p.b2[t];
    if (t < classes) b3[t] = p.b3[t];
    __syncthreads();

#ifdef BARENN_PERFIL
    long long ciclos[fases] = {};
    long long antes = clock64();
#endif
    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int k = 0; k < n; ++k) {
            x[t] = imagens[static_cast<size_t>(k) * entradas + t];
            __syncthreads();
            MARCA(0)

            forward_camada<entradas, oculta1, stride1, false>(x, w1, b1, z1, a1);
            __syncthreads();
            MARCA(1)
            forward_camada<oculta1, oculta2, stride2, false>(a1, w2, b2, z2, a2);
            __syncthreads();
            MARCA(2)
            forward_camada<oculta2, classes, stride3, true>(a2, w3, b3, z3, a3);
            __syncthreads();
            MARCA(3)

            // Perda: o erro de cada classe é o próprio delta da última camada.
            if (t < classes) {
                const float erro = a3[t] - (t == rotulos[k] ? 1.0f : 0.0f);
                delta3[t] = erro;
                termos[t] = 0.5f * erro * erro;
            }
            __syncthreads();
            MARCA(4)
            // O thread 0 soma na mesma ordem da CPU; ninguém mais usa termos
            // até a próxima amostra.
            if (t == 0) {
                float total = 0;
                for (int c = 0; c < classes; ++c) {
                    total += termos[c];
                }
                perdas[static_cast<size_t>(epoca) * n + k] = total;
            }

            backward_camada<oculta2, classes, stride3, true>(delta3, a2, w3, b3, z2, delta2, taxa);
            __syncthreads();
            MARCA(5)
            backward_camada<oculta1, oculta2, stride2, true>(delta2, a1, w2, b2, z1, delta1, taxa);
            __syncthreads();
            MARCA(6)
            backward_camada<entradas, oculta1, stride1, false>(delta1, x, w1, b1, nullptr, nullptr, taxa);
            __syncthreads();
            MARCA(7)
        }
    }

#ifdef BARENN_PERFIL
    if (t == 0) {
        for (int f = 0; f < fases; ++f) {
            perfil[f] = ciclos[f];
        }
    }
#endif
    copiar_pesos<entradas, oculta1, stride1, false>(p.w1, w1);
    copiar_pesos<oculta1, oculta2, stride2, false>(p.w2, w2);
    copiar_pesos<oculta2, classes, stride3, false>(p.w3, w3);
    if (t < oculta1) p.b1[t] = b1[t];
    if (t < oculta2) p.b2[t] = b2[t];
    if (t < classes) p.b3[t] = b3[t];
}

// Reserva um vetor na GPU e copia o conteúdo do host.
template <typename T>
T* enviar(const T* dados, size_t n) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * n));
    CUDA_CHECK(cudaMemcpy(ptr, dados, sizeof(T) * n, cudaMemcpyHostToDevice));
    return ptr;
}

template <typename T>
void receber(T* destino, const T* ptr, size_t n) {
    CUDA_CHECK(cudaMemcpy(destino, ptr, sizeof(T) * n, cudaMemcpyDeviceToHost));
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
        Parametros p{
            enviar(l1->pesos(), entradas * oculta1), enviar(l1->vies(), oculta1),
            enviar(l2->pesos(), oculta1 * oculta2), enviar(l2->vies(), oculta2),
            enviar(l3->pesos(), oculta2 * classes), enviar(l3->vies(), classes)};

        // Conjunto de treino inteiro na GPU, copiado uma única vez.
        const int n = static_cast<int>(treino.size());
        vector<float> imagens(static_cast<size_t>(n) * entradas);
        vector<int> rotulos(n);
        for (int k = 0; k < n; ++k) {
            copy(treino[k].imagem.begin(), treino[k].imagem.end(),
                 imagens.begin() + static_cast<size_t>(k) * entradas);
            rotulos[k] = treino[k].rotulo;
        }
        float* d_imagens = enviar(imagens.data(), imagens.size());
        int* d_rotulos = enviar(rotulos.data(), rotulos.size());
        float* d_perdas;
        CUDA_CHECK(cudaMalloc(&d_perdas, sizeof(float) * epocas * n));
        long long* d_perfil;
        CUDA_CHECK(cudaMalloc(&d_perfil, sizeof(long long) * fases));
        CUDA_CHECK(cudaMemset(d_perfil, 0, sizeof(long long) * fases));

        // Um único lançamento treina todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        treinar_kernel<<<1, threads>>>(p, d_imagens, d_rotulos, n, epocas,
                                       Barenn::taxa, d_perdas, d_perfil);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

#ifdef BARENN_PERFIL
        // Ciclos por amostra em cada fase (stderr, para não alterar a saída).
        array<long long, fases> perfil{};
        receber(perfil.data(), d_perfil, fases);
        const char* nomes[fases] = {"carga x", "forward 1", "forward 2", "forward 3",
                                    "perda", "backward 3", "backward 2", "backward 1"};
        long long total_ciclos = 0;
        for (long long c : perfil) {
            total_ciclos += c;
        }
        const double amostras = static_cast<double>(epocas) * n;
        cerr << fixed << setprecision(1) << "PERFIL (ciclos por amostra)\n";
        for (int f = 0; f < fases; ++f) {
            cerr << setw(12) << nomes[f] << " " << setw(8) << perfil[f] / amostras
                 << "  " << setw(5) << 100.0 * perfil[f] / total_ciclos << " %\n";
        }
        cerr << setw(12) << "total" << " " << setw(8) << total_ciclos / amostras
             << "  clock efetivo " << setprecision(0)
             << total_ciclos / tempo.count() / 1e6 << " MHz\n";
#endif
        cudaFree(d_perfil);

        // Soma as perdas em double e na ordem das amostras, como na CPU.
        vector<float> perdas_amostra(static_cast<size_t>(epocas) * n);
        receber(perdas_amostra.data(), d_perdas, perdas_amostra.size());
        array<double, epocas> perdas{};
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (int k = 0; k < n; ++k) {
                perda_total += perdas_amostra[static_cast<size_t>(epoca) * n + k];
            }
            perdas[epoca] = perda_total / n;
        }

        // Traz os pesos treinados de volta e avalia com o código da CPU.
        receber(l1->pesos(), p.w1, entradas * oculta1);
        receber(l1->vies(), p.b1, oculta1);
        receber(l2->pesos(), p.w2, oculta1 * oculta2);
        receber(l2->vies(), p.b2, oculta2);
        receber(l3->pesos(), p.w3, oculta2 * classes);
        receber(l3->vies(), p.b3, classes);
        for (float* ptr : {p.w1, p.b1, p.w2, p.b2, p.w3, p.b3, d_imagens, d_perdas}) {
            cudaFree(ptr);
        }
        cudaFree(d_rotulos);

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
