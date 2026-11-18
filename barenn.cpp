#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// As dimensões são parâmetros de template: o compilador conhece o tamanho de
// todos os laços, desenrola e vetoriza sem testes de resto ou de aliasing.
template <int In, int Out, bool Final = false>
class DenseLayer {
    // Matriz contígua transposta: weights[j * Out + i] liga a entrada j ao
    // neurônio de saída i. Cada linha guarda os pesos de uma entrada para
    // todos os neurônios, então o forward vira AXPY e vetoriza sem reordenar
    // as somas. Alinhada a 32 bytes para cargas AVX.
    alignas(32) array<float, In * Out> weights;
    alignas(32) array<float, Out> bias{};
    // Dados do último forward usados para calcular os gradientes.
    const float* last_input = nullptr;
    alignas(32) array<float, Out> last_z{};
    // Buffers reaproveitados entre chamadas para evitar alocações.
    alignas(32) array<float, Out> out{};
    alignas(32) array<float, Out> delta{};
    alignas(32) array<float, In> grad_in{};

public:
    DenseLayer() {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // Sorteia na ordem neurônio -> entrada, a mesma do código original.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < Out; ++i) {
            for (int j = 0; j < In; ++j) {
                weights[j * Out + i] = distribution(gen);
            }
        }
    }

    // A entrada precisa continuar viva até o Backward correspondente.
    const array<float, Out>& Forward(const float* __restrict x) {
        last_input = x;
        // z = bias + soma em j de x[j] * w[j][:], acumulando entrada por entrada.
        // Cada neurônio soma na mesma ordem do produto escalar original.
        float* __restrict z = last_z.data();
        for (int i = 0; i < Out; ++i) {
            z[i] = bias[i];
        }
        for (int j = 0; j < In; ++j) {
            const float* __restrict row = &weights[j * Out];
            const float xj = x[j];
            for (int i = 0; i < Out; ++i) {
                z[i] += xj * row[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < Out; ++i) {
            out[i] = Final ? z[i] : max(0.0f, z[i]);
        }
        return out;
    }

    // A primeira camada não precisa do gradiente da entrada (CalcularGradIn = false).
    template <bool CalcularGradIn = true>
    const array<float, In>& Backward(const float* __restrict grad_out, float learning_rate) {
        const float* __restrict x = last_input;
        float* __restrict d = delta.data();
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < Out; ++i) {
            d[i] = Final ? grad_out[i]
                         : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // Uma única passada por linha (entrada j): primeiro calcula grad_in[j]
        // com os pesos antigos, depois aplica o SGD na linha.
        for (int j = 0; j < In; ++j) {
            float* __restrict row = &weights[j * Out];
            if constexpr (CalcularGradIn) {
                float sum = 0;
                for (int i = 0; i < Out; ++i) {
                    sum += d[i] * row[i];
                }
                grad_in[j] = sum;
            }
            // SGD por amostra. O gradiente do peso é delta vezes a entrada.
            const float xj = x[j];
            for (int i = 0; i < Out; ++i) {
                row[i] -= learning_rate * (d[i] * xj);
            }
        }
        // O gradiente do viés é o próprio delta.
        for (int i = 0; i < Out; ++i) {
            bias[i] -= learning_rate * d[i];
        }
        return grad_in;
    }
};

// Rede densa com dimensões 64 -> 32 -> 16 -> 10.
using Camada1 = DenseLayer<entradas, 32>;
using Camada2 = DenseLayer<32, 16>;
using Camada3 = DenseLayer<16, classes, true>;

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

// Converte uma linha do arquivo. Devolve false se o formato for inválido.
bool ler_linha(const string& linha, Amostra& amostra) {
    istringstream campos(linha);
    amostra.imagem.clear();
    amostra.imagem.reserve(entradas);
    // Cada atributo deve ser seguido por vírgula, inclusive o último.
    for (int i = 0; i < entradas; ++i) {
        float valor;
        char separador;
        if (!(campos >> valor >> separador) || separador != ',' ||
            !isfinite(valor) || valor < 0 || valor > 16) {
            break;
        }
        // Converte os valores de [0, 16] para [0, 1].
        amostra.imagem.push_back(valor / 16.0f);
    }
    // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
    return amostra.imagem.size() == entradas &&
           (campos >> amostra.rotulo) &&
           amostra.rotulo >= 0 && amostra.rotulo < classes &&
           (campos >> ws).eof();
}

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    // A leitura do disco é sequencial; a conversão de cada linha é
    // independente e roda em paralelo.
    vector<string> linhas;
    string linha;
    while (getline(arquivo, linha)) {
        linhas.push_back(move(linha));
    }
    if (arquivo.bad() || linhas.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }

    const long n = static_cast<long>(linhas.size());
    vector<Amostra> dados(n);
    // Guarda a primeira linha inválida para reportar o mesmo erro da versão serial.
    long primeira_invalida = n;
    #pragma omp parallel for schedule(static) reduction(min : primeira_invalida)
    for (long i = 0; i < n; ++i) {
        if (!ler_linha(linhas[i], dados[i])) {
            primeira_invalida = min(primeira_invalida, i);
        }
    }
    if (primeira_invalida < n) {
        throw runtime_error(caminho + ":" + to_string(primeira_invalida + 1) +
            " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const array<float, classes>& escores, int rotulo,
            array<float, classes>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(Camada1& l1, Camada2& l2, Camada3& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem.data()).data()).data());
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto inicio_total = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_total;

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // As camadas ficam no heap: juntas têm ~11 KB de pesos.
        auto l1 = make_unique<Barenn::Camada1>();
        auto l2 = make_unique<Barenn::Camada2>();
        auto l3 = make_unique<Barenn::Camada3>();
        array<float, Barenn::classes> gradiente{};
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores =
                    l3->Forward(l2->Forward(l1->Forward(amostra.imagem.data()).data()).data());
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const auto& grad3 = l3->Backward(gradiente.data(), taxa);
                const auto& grad2 = l2->Backward(grad3.data(), taxa);
                l1->Backward<false>(grad2.data(), taxa);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        const auto inicio_avaliacao = chrono::steady_clock::now();
        Barenn::avaliar(*l1, *l2, *l3, teste);
        const auto fim = chrono::steady_clock::now();
        const chrono::duration<double> tempo_avaliacao = fim - inicio_avaliacao;
        const chrono::duration<double> tempo_total = fim - inicio_total;

        // Tempos das outras etapas, para enxergar o que não é treino.
        cout << fixed << setprecision(4)
             << "\nTempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n"
             << "Tempo total " << tempo_total.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
