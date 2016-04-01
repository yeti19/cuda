#include <iostream>
#include <cstdlib>
#include <ctime>

/*
    Test maker: program tworzący przykładowe dane dla dużego zadania
    z CUDA. Wejście:
    -   n - liczba obiektów
    -   k - liczba zmiennych opisowych na obiekt
    -   c - liczba klas każdej zmiennej opisowej
    -   związki między zmiennymi(TODO)

    Format wyjściowy:
    n k c
    n razy
        zmienna decyzyjna [0, 1]
        k liczb z zakresu [0, c)
*/

int main()
{
    int n, k, c;
    srand(time(NULL));
    std::cin >> n >> k >> c;
    std::cout << n << " " << k << " " << c << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << rand() % 2;
        for (int j = 0; j < k; j++) {
            std::cout << " " << rand() % c;
        }
        std::cout << std::endl;
    }
    return 0;
}
