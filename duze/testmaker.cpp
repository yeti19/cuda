#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tuple>

/*
    Test maker: program tworzący przykładowe dane dla dużego zadania
    z CUDA. Wejście:
    -   n - liczba obiektów
    -   k - liczba zmiennych opisowych na obiekt
    -   c - liczba klas każdej zmiennej opisowej
    -   p - liczba zmiennych które chcemy w wyniku
    -   s - liczba związków pojedyńczych
    -   d - liczba związków podwójnych
    -   każdy związek pojedyńczy:
        - a - zmienna której dotyczy
        - b - wartość zmiennej której dotyczy
        - p - zmiana +- szansy na pozytywną zmienną decyzyjną (1)
    -   każdy związek podwójny:
        - a1 a2 - zmienne których dotyczy
        - b1 b2 - wartości zmiennych
        - p - zmiana j/w

    Format wyjściowy:
    n k p a
    n razy
        zmienna decyzyjna [0, 1]
        k liczb z zakresu [0, c)

    p - P z zadania
    a - pr. a priori z zadania (% chorych, na razie zaślepka 0.5)
*/

int main()
{
    int n, k, c, s, d, p;
    srand(time(NULL));
    std::cin >> n >> k >> c >> p >> s >> d;

    std::tuple<int, int, float> *singles = new std::tuple<int, int, float>[s];
    for (int i = 0; i < s; i++)
        std::cin >> std::get<0>(singles[i]) >> std::get<1>(singles[i]) >> std::get<2>(singles[i]);
    std::tuple<int, int, int, int, float> *doubles = new std::tuple<int, int, int, int, float>[d];
    for (int i = 0; i < d; i++)
        std::cin >> std::get<0>(doubles[i]) >> std::get<1>(doubles[i]) >> std::get<2>(doubles[i]) >>
                    std::get<3>(doubles[i]) >> std::get<4>(doubles[i]);

    int *vars = new int[n * k];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            vars[i * k + j] = rand() % c;

    std::cout << n << " " << k << " " << p << " 0.5" << std::endl;
    for (int i = 0; i < n; i++) {
        float chance = 0.5f;

        for (int j = 0; j < s; j++)
            if (vars[i * k + std::get<0>(singles[j])] == std::get<1>(singles[j]))
                if  (std::get<2>(singles[j]) > 0.0f) chance += std::get<2>(singles[j]) * (1.0f - chance);
                else                                 chance -= std::get<2>(singles[j]) * chance;

        for (int j = 0; j < d; j++)
            if (vars[i * k + std::get<0>(doubles[j])] == std::get<2>(doubles[j]) &&
                vars[i * k + std::get<1>(doubles[j])] == std::get<3>(doubles[j]))
                if  (std::get<4>(doubles[j]) > 0.0f) chance += std::get<4>(doubles[j]) * (1.0f - chance);
                else                                 chance -= std::get<4>(doubles[j]) * chance;

        std::cout << (((static_cast<float>(rand() % 10000) * 0.0001f) > chance) ? "1" : "0");

        for (int j = 0; j < k; j++)
            std::cout << " " << vars[i * k + j];
        std::cout << std::endl;
    }
    return 0;
}
