#include "barenn.hpp"

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

        using Barenn::epocas;
        using Barenn::taxa;
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
        Barenn::imprimir_treino(treino.size(), perdas, tempo.count());
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
