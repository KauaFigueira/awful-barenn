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
// Threads do bloco. Forward e grad_in usam um thread por neurônio/entrada
// (no máximo 64); o update dos pesos é distribuído entre todos os threads,
// o que dá mais warps para esconder a latência da shared memory.
#ifndef BARENN_THREADS
#define BARENN_THREADS 256
#endif
constexpr int threads = BARENN_THREADS;
static_assert(threads > 2 * 32 && threads >= entradas && threads % 32 == 0,
              "bloco pequeno demais");

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
// e devolve o delta da camada anterior (grad_in * ReLU'(z_anterior)).
// Precisa terminar (barreira) antes do update, que mexe em todas as linhas.
template <int In, int Out, int Stride>
__device__ void gradiente_entrada(const float* delta, const float* w,
                                  const float* z_anterior, float* delta_anterior) {
    const int j = threadIdx.x;
    if (j < In) {
        const float* linha = w + j * Stride;
        float soma = 0;
        #pragma unroll
        for (int i = 0; i < Out; ++i) {
            soma += delta[i] * linha[i];
        }
        delta_anterior[j] = soma * (z_anterior[j] > 0.0f ? 1.0f : 0.0f);
    }
}

// SGD distribuído: o peso (j, i) fica com o thread k = j * Out + i (mod
// blockDim). Threads vizinhos pegam colunas vizinhas da mesma linha, então
// não há conflito de banco; x[j] é broadcast.
template <int In, int Out, int Stride>
__device__ void atualizar_pesos(const float* delta, const float* x, float* w,
                                float* b, float taxa) {
    const int t = threadIdx.x;
    if constexpr (threads % Out == 0) {
        // A coluna i de cada thread é fixa: delta[i] fica num registrador e
        // o laço só anda pelas linhas.
        const int i = t % Out;
        const float d = delta[i];
        #pragma unroll
        for (int j = t / Out; j < In; j += threads / Out) {
            w[j * Stride + i] -= taxa * (d * x[j]);
        }
    } else {
        #pragma unroll 4
        for (int k = t; k < In * Out; k += threads) {
            const int j = k / Out;
            const int i = k % Out;
            w[j * Stride + i] -= taxa * (delta[i] * x[j]);
        }
    }
    if (t < Out) {
        b[t] -= taxa * delta[t];
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
constexpr int fases = 6;
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

constexpr int tamanho_w1 = entradas * oculta1;
constexpr int tamanho_w2 = oculta1 * oculta2;
constexpr int tamanho_w3 = oculta2 * classes;

// Treino inteiro em um único lançamento. Cada bloco treina um modelo
// independente (bloco m usa taxas[m] e o m-ésimo conjunto de parâmetros);
// toda a rede do bloco fica na shared memory e as fases de cada amostra são
// separadas por __syncthreads().
__global__ void __launch_bounds__(threads)
treinar_kernel(Parametros p, const float* imagens, const int* rotulos, int n,
               int epocas, const float* taxas, float* perdas, long long* perfil) {
    __shared__ float w1[entradas * stride1], b1[oculta1];
    __shared__ float w2[oculta1 * stride2], b2[oculta2];
    __shared__ float w3[oculta2 * stride3], b3[classes];
    // Duas cópias de x: enquanto a amostra atual usa uma, a próxima é
    // escrita na outra durante a fase de update.
    __shared__ float x[2][entradas];
    __shared__ float z1[oculta1], a1[oculta1], delta1[oculta1];
    __shared__ float z2[oculta2], a2[oculta2], delta2[oculta2];
    __shared__ float z3[classes], a3[classes], delta3[classes];
    __shared__ float termos[classes];

    const int t = threadIdx.x;
    const int m = blockIdx.x;
    const float taxa = taxas[m];
    float* gw1 = p.w1 + m * tamanho_w1;
    float* gw2 = p.w2 + m * tamanho_w2;
    float* gw3 = p.w3 + m * tamanho_w3;
    float* gb1 = p.b1 + m * oculta1;
    float* gb2 = p.b2 + m * oculta2;
    float* gb3 = p.b3 + m * classes;
    perdas += static_cast<size_t>(m) * epocas * n;

    copiar_pesos<entradas, oculta1, stride1, true>(gw1, w1);
    copiar_pesos<oculta1, oculta2, stride2, true>(gw2, w2);
    copiar_pesos<oculta2, classes, stride3, true>(gw3, w3);
    if (t < oculta1) b1[t] = gb1[t];
    if (t < oculta2) b2[t] = gb2[t];
    if (t < classes) b3[t] = gb3[t];
    if (t < entradas) x[0][t] = imagens[t];
    // Rótulo da amostra atual, mantido em registrador pelos threads da perda.
    int rotulo = rotulos[0];
    __syncthreads();

#ifdef BARENN_PERFIL
    long long ciclos[fases] = {};
    long long antes = clock64();
#endif
    int atual = 0;
    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int k = 0; k < n; ++k) {
            const float* xa = x[atual];
            // Prefetch: a leitura da próxima amostra (memória global, ~300
            // ciclos) é disparada agora e só é usada no fim da amostra, então
            // a latência fica escondida atrás das fases de cálculo.
            const int seguinte = k + 1 < n ? k + 1 : 0;
            const float x_seguinte =
                t < entradas ? imagens[static_cast<size_t>(seguinte) * entradas + t] : 0.0f;
            const int rotulo_seguinte = rotulos[seguinte];

            forward_camada<entradas, oculta1, stride1, false>(xa, w1, b1, z1, a1);
            __syncthreads();
            MARCA(0)
            forward_camada<oculta1, oculta2, stride2, false>(a1, w2, b2, z2, a2);
            __syncthreads();
            MARCA(1)
            // Forward 3 e perda na mesma fase: o thread t usa só o a3[t] que
            // ele mesmo acabou de calcular. O erro de cada classe é o delta da
            // última camada.
            forward_camada<oculta2, classes, stride3, true>(a2, w3, b3, z3, a3);
            if (t < classes) {
                const float erro = a3[t] - (t == rotulo ? 1.0f : 0.0f);
                delta3[t] = erro;
                termos[t] = 0.5f * erro * erro;
            }
            __syncthreads();
            MARCA(2)
            // Um thread que não participa do grad_in 3 soma a perda, na mesma
            // ordem da CPU; ninguém mais usa termos até a próxima amostra.
            if (t == 2 * 32) {
                float total = 0;
                for (int c = 0; c < classes; ++c) {
                    total += termos[c];
                }
                perdas[static_cast<size_t>(epoca) * n + k] = total;
            }

            // Os grad_in das camadas 3 e 2 usam os pesos antigos; os updates
            // só começam depois. Cada update mexe numa matriz diferente, então
            // os três podem rodar na mesma fase.
            gradiente_entrada<oculta2, classes, stride3>(delta3, w3, z2, delta2);
            __syncthreads();
            MARCA(3)
            gradiente_entrada<oculta1, oculta2, stride2>(delta2, w2, z1, delta1);
            __syncthreads();
            MARCA(4)
            atualizar_pesos<oculta2, classes, stride3>(delta3, a2, w3, b3, taxa);
            atualizar_pesos<oculta1, oculta2, stride2>(delta2, a1, w2, b2, taxa);
            atualizar_pesos<entradas, oculta1, stride1>(delta1, xa, w1, b1, taxa);
            if (t < entradas) {
                x[atual ^ 1][t] = x_seguinte;
            }
            rotulo = rotulo_seguinte;
            atual ^= 1;
            __syncthreads();
            MARCA(5)
        }
    }

#ifdef BARENN_PERFIL
    if (t == 0 && m == 0) {
        for (int f = 0; f < fases; ++f) {
            perfil[f] = ciclos[f];
        }
    }
#endif
    copiar_pesos<entradas, oculta1, stride1, false>(gw1, w1);
    copiar_pesos<oculta1, oculta2, stride2, false>(gw2, w2);
    copiar_pesos<oculta2, classes, stride3, false>(gw3, w3);
    if (t < oculta1) gb1[t] = b1[t];
    if (t < oculta2) gb2[t] = b2[t];
    if (t < classes) gb3[t] = b3[t];
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

// Repete os parâmetros iniciais para cada modelo.
vector<float> replicar(const float* dados, size_t n, int modelos) {
    vector<float> v(n * modelos);
    for (int m = 0; m < modelos; ++m) {
        copy(dados, dados + n, v.begin() + m * n);
    }
    return v;
}

// Número de modelos treinados ao mesmo tempo (variável BARENN_MODELOS).
// O modelo 0 usa a taxa original e produz a saída padrão; os demais varrem
// taxas de taxa/4 a 4·taxa em escala logarítmica.
int ler_modelos() {
    const char* valor = getenv("BARENN_MODELOS");
    const int modelos = valor ? atoi(valor) : 1;
    if (modelos < 1 || modelos > 4096) {
        throw runtime_error("BARENN_MODELOS deve estar entre 1 e 4096");
    }
    return modelos;
}

vector<float> criar_taxas(int modelos) {
    vector<float> taxas(modelos, Barenn::taxa);
    for (int m = 1; m < modelos; ++m) {
        const double expoente = modelos > 2 ? -2.0 + 4.0 * (m - 1) / (modelos - 2) : 1.0;
        taxas[m] = static_cast<float>(Barenn::taxa * pow(2.0, expoente));
    }
    return taxas;
}

// Acertos no conjunto de teste, usando o forward da CPU.
size_t contar_acertos(Barenn::Camada1& l1, Barenn::Camada2& l2, Barenn::Camada3& l3,
                      const vector<Barenn::Amostra>& teste) {
    size_t acertos = 0;
    for (const auto& amostra : teste) {
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem.data()).data()).data());
        const int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        acertos += previsto == amostra.rotulo;
    }
    return acertos;
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
        const int modelos = ler_modelos();
        const vector<float> taxas = criar_taxas(modelos);
        // Todos os modelos partem dos mesmos pesos iniciais.
        auto enviar_todos = [&](const float* dados, size_t tamanho) {
            const vector<float> v = replicar(dados, tamanho, modelos);
            return enviar(v.data(), v.size());
        };
        Parametros p{
            enviar_todos(l1->pesos(), tamanho_w1), enviar_todos(l1->vies(), oculta1),
            enviar_todos(l2->pesos(), tamanho_w2), enviar_todos(l2->vies(), oculta2),
            enviar_todos(l3->pesos(), tamanho_w3), enviar_todos(l3->vies(), classes)};
        float* d_taxas = enviar(taxas.data(), taxas.size());

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
        CUDA_CHECK(cudaMalloc(&d_perdas, sizeof(float) * modelos * epocas * n));
        long long* d_perfil;
        CUDA_CHECK(cudaMalloc(&d_perfil, sizeof(long long) * fases));
        CUDA_CHECK(cudaMemset(d_perfil, 0, sizeof(long long) * fases));

        // Um único lançamento treina todas as épocas de todos os modelos.
        const auto inicio = chrono::steady_clock::now();
        treinar_kernel<<<modelos, threads>>>(p, d_imagens, d_rotulos, n, epocas,
                                             d_taxas, d_perdas, d_perfil);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

#ifdef BARENN_PERFIL
        // Ciclos por amostra em cada fase (stderr, para não alterar a saída).
        array<long long, fases> perfil{};
        receber(perfil.data(), d_perfil, fases);
        const char* nomes[fases] = {"forward 1", "forward 2", "fwd 3+perda",
                                    "grad_in 3", "grad_in 2", "updates"};
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

        // Com vários modelos, resume cada um no stderr (a saída padrão continua
        // sendo a do modelo 0).
        if (modelos > 1) {
            auto m1 = make_unique<Barenn::Camada1>();
            auto m2 = make_unique<Barenn::Camada2>();
            auto m3 = make_unique<Barenn::Camada3>();
            vector<float> ultima(n);
            cerr << fixed << "MODELOS " << modelos << " | treino " << setprecision(4)
                 << tempo.count() << " s\n";
            for (int m = 0; m < modelos; ++m) {
                receber(m1->pesos(), p.w1 + m * tamanho_w1, tamanho_w1);
                receber(m1->vies(), p.b1 + m * oculta1, oculta1);
                receber(m2->pesos(), p.w2 + m * tamanho_w2, tamanho_w2);
                receber(m2->vies(), p.b2 + m * oculta2, oculta2);
                receber(m3->pesos(), p.w3 + m * tamanho_w3, tamanho_w3);
                receber(m3->vies(), p.b3 + m * classes, classes);
                receber(ultima.data(),
                        d_perdas + (static_cast<size_t>(m) * epocas + epocas - 1) * n, n);
                double perda_final = 0;
                for (float v : ultima) {
                    perda_final += v;
                }
                const size_t acertos = contar_acertos(*m1, *m2, *m3, teste);
                cerr << "modelo " << setw(4) << m << " taxa " << setprecision(5) << taxas[m]
                     << " perda final " << setprecision(4) << perda_final / n
                     << " acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
                     << "%\n";
            }
        }

        // Traz os pesos treinados do modelo 0 e avalia com o código da CPU.
        receber(l1->pesos(), p.w1, tamanho_w1);
        receber(l1->vies(), p.b1, oculta1);
        receber(l2->pesos(), p.w2, tamanho_w2);
        receber(l2->vies(), p.b2, oculta2);
        receber(l3->pesos(), p.w3, tamanho_w3);
        receber(l3->vies(), p.b3, classes);
        for (float* ptr : {p.w1, p.b1, p.w2, p.b2, p.w3, p.b3, d_imagens, d_perdas, d_taxas}) {
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
