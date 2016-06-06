#include <iostream>
#include <exception>
#include <cassert>
#include <cmath>

#define SHOULD_BE(a, b) { \
    int va = (a), vb = (b); \
    if (va != vb) \
        std::cout << "Line " << __LINE__ << ": " << #a << " should be " << b << ", but is " << va << std::endl; \
    }(void)0

template<typename T>
class ZeroVector {
    T* memory;
public:
    ZeroVector(int n) { memory = new T[n]; }
    ~ZeroVector() { delete[] memory; }
    T& operator[](int n) { return memory[n]; }
};

template<int N>
class IterableTuple {
    const int maximum;
    int vars[N];
public:
    IterableTuple(int max) : maximum(max) {
        static_assert(N > 0, "N must be > 0");
        if (max <= 0) throw std::exception();
        for (int i = 0; i < N; i++)
            vars[i] = 0;
    }
    IterableTuple(const IterableTuple<N>& it) :
        maximum(it.getMaximum())
    {
        for (int i = 0; i < N; i++)
            vars[i] = it[i];
    }
    IterableTuple(const IterableTuple<N + 1>& it, int except) :
        maximum(it.getMaximum())
    {
        for (int i = 0, j = 0; i < N + 1; i++)
            if (i != except)
                vars[j++] = it[i];
    }
    int getMaximum() const { return maximum; }

    operator bool() const {
        return vars[0] < maximum;
    }
    const IterableTuple<N>& operator++() {
        if (!(*this)) throw std::exception();

        int i = N - 1; vars[i]++;
        while (i > 0 && vars[i] == maximum) {
            vars[i] = 0;
            vars[--i]++;
        }
        return *this;
    }
    int operator[](int i) const {
        if (i >= N) throw std::exception();
        if (!(*this)) throw std::exception();
        return vars[i];
    }
};

bool test_IterableTuple()
{
    //IterableTuple<0> it(100); //Should not compile
    try {
        IterableTuple<1> it(0); //Should throw
        assert(0);
    } catch (...) { }
    try {
        IterableTuple<23> it(-2); //Should throw
        assert(0);
    } catch (...) { }

    {
        IterableTuple<1> it(3);
        assert(it);
        assert(it[0] == 0);
        try {
            std::cout << it[1]; //Should throw, bad index
            assert(0);
        } catch (...) { }
        ++it; ++it;
        assert(it);
        assert(it[0] == 2);

        ++it;
        assert(!it);
        try {
            std::cout << it[0]; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
        try {
            ++it; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
        try {
            std::cout << it[0]; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
    }

    {
        IterableTuple<3> it(3);
        for (int i = 0; i < 10; i++)
            ++it;
        assert(it);
        assert(it[0] == 1);
        assert(it[1] == 0);
        assert(it[2] == 1);
        try {
            std::cout << it[3]; //Should throw, bad index
            assert(0);
        } catch (...) { }

        for (int i = 10; i < 27; i++)
            ++it;
        assert(!it);
    }

    {
        IterableTuple<3> it(3);
        for (int i = 0; i < 10; i++)
            ++it;
        IterableTuple<3> it2(it);
        assert(it2);
        assert(it2[0] == 1);
        assert(it2[1] == 0);
        assert(it2[2] == 1);
        try {
            std::cout << it2[3]; //Should throw, bad index
            assert(0);
        } catch (...) { }
    }

    {
        IterableTuple<3> it(3);
        for (int i = 0; i < 10; i++)
            ++it;
        IterableTuple<2> it2(it, 1);
        assert(it2);
        assert(it2[0] == 1);
        assert(it2[1] == 1);
        try {
            std::cout << it2[2]; //Should throw, bad index
            assert(0);
        } catch (...) { }
    }

    return true;
}

template<int N>
class AscTuple {
    const int vars[N];
public:
    AscTuple(const int *v) {
        static_assert(N > 0, "N must be > 0");
        for (int i = 0; i < N; i++)
            vars[i] = v[i];
    }
    AscTuple(const AscTuple<N>& it) {
        for (int i = 0; i < N; i++)
            vars[i] = it[i];
    }
    AscTuple(const AscTuple<N + 1>& it, int except) {
        for (int i = 0, j = 0; i < N + 1; i++)
            if (i != except)
                vars[j++] = it[i];
    }

    AscTuple<N - 1> getExcept(int except) {
        AscTuple<N - 1> result(*this, except);
        return result;
    }
    int operator[](int i) const {
        if (i >= N) throw std::exception();
        return vars[i];
    }
};

template<int N>
class IterableAscTupleFactory {
    const int maximum, minimum;
    int counter, max_counter;
public:
    IterableAscTupleFactory(int max, int min) : maximum(max), minimum(min) {
        static_assert(N > 0, "N must be > 0");
        if (max <= 0) throw std::exception();
        if (max < N) throw std::exception();
        if (min < 0) throw std::exception();
        if (min > N + max) throw std::exception();

        counter = 0;
    }
    //int getMaximum() const { return maximum; }

    operator bool() const {
        return vars[0] <= maximum - N;
    }

    int operator[](int i) const {
        if (i >= N) throw std::exception();
        if (!(*this)) throw std::exception();
        return vars[i];
    }

    static AscTuple<N> getNextAfter(AscTuple<N> &at, int maximum) {
        int vars[N];
        for (int i = 0; i < N; ++i)
            vars[i] = at[i];

        if (vars[0] > maximum - N) throw std::exception();

        if (vars[N - 1] < maximum - 1)
            ++vars[N - 1];
        else {
            int i = 2;
            for (; i <= N && vars[N - i] == maximum - i; ++i);
            if (i <= N) {
                vars[N - i]++;
                for (--i; i > 0; --i)
                    vars[N - i] = vars[N - i - 1] + 1;
            } else
                ++vars[0];
        }

        return AscTuple<N>(vars);
    }
};

bool test_IterableAscTuple()
{
    //IterableTuple<0> it(100); //Should not compile
    try {
        IterableAscTuple<1> it(0); //Should throw
        assert(0);
    } catch (...) { }
    try {
        IterableAscTuple<23> it(-2); //Should throw
        assert(0);
    } catch (...) { }
    try {
        IterableAscTuple<3> it(2); //Should throw
        assert(0);
    } catch (...) { }

    {
        IterableAscTuple<1> it(3);
        assert(it);
        assert(it[0] == 0);
        try {
            std::cout << it[1]; //Should throw, bad index
            assert(0);
        } catch (...) { }
        ++it; ++it;
        assert(it);
        assert(it[0] == 2);

        ++it;
        assert(!it);
        try {
            std::cout << it[0]; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
        try {
            ++it; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
        try {
            std::cout << it[0]; //Should throw, it is invalid
            assert(0);
        } catch (...) { }
    }

    {
        IterableAscTuple<3> it(5);
        for (int i = 0; i < 4; i++)
            ++it;
        assert(it);
        SHOULD_BE(it[0], 0);
        SHOULD_BE(it[1], 2);
        SHOULD_BE(it[2], 4);
        try {
            std::cout << it[3]; //Should throw, bad index
            assert(0);
        } catch (...) { }

        for (int i = 4; i < 10; i++)
            ++it;
        assert(!it);
    }

    {
        IterableAscTuple<3> it(5);
        for (int i = 0; i < 4; i++)
            ++it;
        IterableAscTuple<3> it2(it);
        assert(it2);
        SHOULD_BE(it2[0], 0);
        SHOULD_BE(it2[1], 2);
        SHOULD_BE(it2[2], 4);
        try {
            std::cout << it2[3]; //Should throw, bad index
            assert(0);
        } catch (...) { }
    }

    {
        IterableAscTuple<3> it(5);
        for (int i = 0; i < 6; i++)
            ++it;
        IterableAscTuple<2> it2(it, 1);
        assert(it2);
        SHOULD_BE(it2[0], 1);
        SHOULD_BE(it2[1], 3);
        try {
            std::cout << it2[2]; //Should throw, bad index
            assert(0);
        } catch (...) { }
    }

    return true;
}

class Data {
    int numObjects, numVars, numClasses;
    int *classes;
public:
    Data(std::istream& ist) {
        ist >> numObjects >> numVars >> numClasses;
        numVars++;
        classes = new int[numObjects * numVars];
        for (int i = 0; i < numObjects; ++i) {
            for (int j = 0; j < numVars; ++j)
                ist >> classes[i * numVars + j];
        }
    }
    Data(int numObj, int numVar, int numClas, int *clas) :
        numObjects(numObj),
        numVars(numVar),
        numClasses(numClas)
    {
        classes = new int[numObjects * numVars];
        for (int i = 0, k = 0; i < numObjects; ++i) {
            for (int j = 0; j < numVars; ++j)
                classes[i * numVars + j] = clas[k++];
        }
    }
    ~Data() { delete[] classes; }

    int getNumObjects() const { return numObjects; }
    int getNumVars() const { return numVars; }
    int getNumClasses(int varNum) const {
        if (varNum == 0) return 2;
        return numClasses;
    }
    int getClass(int objNum, int varNum) const {
        return classes[objNum * numVars + varNum];
    }
};

/* TODO: Test Data class?? */

template<int N>
class VariableTuple {
    AscTuple<N> vars;
    const Data &data;

    int numClasses() const {
        int ret = 1;
        for (int i = 0; i < N; i++) {
            ret *= data.getNumClasses(vars[i]);
        }
        return ret;
    }
    int whichClass(int obj) const {
        int ret = 0;
        for (int i = 0; i < N; i++) {
            ret *= data.getNumClasses(vars[i]);
            ret += data.getClass(obj, vars[i]);
        }
        return ret;
    }

public:
    VariableTuple(Data &data, AscTuple &vars) : data(data), vars(vars) { }

    float H() const {
        ZeroVector<int> Count(numClasses());

        for (int i = 0; i < data.getNumObjects(); i++)
            Count[whichClass(i)]++;

        float ret = 0.0f;
        for (int i = 0; i < numClasses(); ++i) {
            float prob = static_cast<float>(Count[i]) /
                         static_cast<float>(data.getNumObjects());
            ret -= prob * log2(prob);
        }

        return ret;
    }

    float IG() {
        VariableTuple<1> decisive(data, AscTuple({0}));
        VariableTuple<N + 1> together(data, AscTuple({0}) + vars);
        return decisive.H() + H() - together.H();
    }

    float GIG() {
        float maxig = 0.0f;
        for (int i = 0; i < N; ++i) {
            VariableTuple<N - 1> vt(data, vars.getExcept(i));
            float ig = vt.IG();
            if (ig > maxig)
                maxig = ig;
        }
        return IG() - maxig;
    }

    const VariableTuple<N>& operator++() {
        ++vars;
        return *this;
    }

    operator bool() { return (bool)vars; }

    friend bool test_VariableTuple();
};

bool test_VariableTuple()
{
    int da[] = {
        0, 0, 0,
        0, 1, 0,
        0, 2, 0,
        1, 0, 1,
        1, 1, 1,
        1, 2, 2,
        1, 1, 2,
        1, 2, 2
    };
    Data test_data(8, 3, 3, da);

    { // Test test_data
        SHOULD_BE(test_data.getNumObjects(), 8);
        SHOULD_BE(test_data.getNumClasses(0), 2);
        SHOULD_BE(test_data.getNumClasses(1), 3);
        SHOULD_BE(test_data.getNumClasses(2), 3);
        SHOULD_BE(test_data.getClass(0, 0), 0);
        SHOULD_BE(test_data.getClass(0, 2), 0);
        SHOULD_BE(test_data.getClass(1, 1), 1);
        SHOULD_BE(test_data.getClass(5, 2), 2);
        // TODO: Throw when out of bounds!!!
    }

    VariableTuple<2> vt(test_data);
    SHOULD_BE(vt.numClasses(), 6);
    SHOULD_BE(vt.whichClass(0), 0);
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            assert(vt.whichClass(i) != vt.whichClass(j));

    if (vt.H() != 2.5f) {
        std::cout << "Expected H(): -4(1/8 * log2(1/8)) - 2(1/4 * log2(1/4)) = " <<
                    -4.0f * (1.0f/8.0f) * log2(1.0f/8.0f) - 2.0f * (1.0f/4.0f) * log2(1.0f/4.0f) << std::endl;
        std::cout << "Calculated: " << vt.H() << std::endl;
    }

    return true;
}

int main()
{
    if (test_IterableTuple()) std::cout << "IterableTuple OK" << std::endl;
    if (test_IterableAscTuple()) std::cout << "IterableAscTuple OK" << std::endl;
    if (test_VariableTuple()) std::cout << "VariableTuple OK" << std::endl;
    /*
    Data data(std::cin);
    std::vector<std::pair<VariableTuple<2>, float>> results;
    for (VariableTuple<2> vt(data); vt; vt++)
        results.push_back({vt, vt.GIG()});
    std::sort(results);
    for (std::pair<VariableTuple<2>, float> p: results)
        std::cout << p.second() << std::endl;
    */
}
