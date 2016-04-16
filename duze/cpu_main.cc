#include <iostream>
#include <exception>
#include <cassert>
#include <cmath>

#define SHOULD_BE(a, b) { \
    int va = (a), vb = (b); \
    if (va != vb) \
        std::cout << "Line " << __LINE__ << ": " << #a << " should be " << b << ", but is " << va << std::endl; \
    }(void)0

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
class IterableAscTuple {
    const int maximum;
    int vars[N];
public:
    IterableAscTuple(int max) : maximum(max) {
        static_assert(N > 0, "N must be > 0");
        if (max <= 0) throw std::exception();
        if (max < N) throw std::exception();
        for (int i = 0; i < N; i++)
            vars[i] = i;
    }
    IterableAscTuple(const IterableAscTuple<N>& it) :
        maximum(it.getMaximum())
    {
        for (int i = 0; i < N; i++)
            vars[i] = it[i];
    }
    IterableAscTuple(const IterableAscTuple<N + 1>& it, int except) :
        maximum(it.getMaximum())
    {
        for (int i = 0, j = 0; i < N + 1; i++)
            if (i != except)
                vars[j++] = it[i];
    }
    int getMaximum() const { return maximum; }

    operator bool() const {
        return vars[0] <= maximum - N;
    }
    const IterableAscTuple<N>& operator++() {
        if (!(*this)) throw std::exception();

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

        return *this;
    }
    int operator[](int i) const {
        if (i >= N) throw std::exception();
        if (!(*this)) throw std::exception();
        return vars[i];
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
                ist >> classes[i * numObjects + j];
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
                classes[i * numObjects + j] = clas[k++];
        }
    }
    ~Data() { delete[] classes; }

    int getNumObjects() const { return numObjects; }
    int getNumVars() const { return numVars; }
    int getNumClasses(int varNum) const {
        if (varNum == 0) return 2;
        return numClasses;
    }
    int getClass(int varNum, int objNum) const {
        return classes[objNum * numObjects + varNum];
    }
};

/* TODO: Test Data class?? */

template<int N>
class VariableTuple {
    IterableAscTuple<N> vars;
    Data &data;

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
            ret += data.getClass(vars[i], obj);
        }
        return ret;
    }

public:
    VariableTuple(Data &data) : data(data), vars(data.getNumVars()) { }

    float H() const {
        int *Count = new int[numClasses()];
        for (int i = 0; i < numClasses(); ++i)
            Count[i] = 0;
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

#if 0
    float IG() { }

    float GIG() {
        return IG(vt) - maximum((IG(ldksf));)
    }
#endif

    const VariableTuple<N>& operator++() {
        vars++;
        return *this;
    }
    /* Warning: bool() returns false if first variable is not 0-th one. */
    operator bool() { return vars[0] == 0; }

    friend bool test_VariableTuple();
};

bool test_VariableTuple()
{
    int da[] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 2, 2, 2, 2
    };
    Data test_data(5, 5, 3, da);

    { // Test test_data
        SHOULD_BE(test_data.getNumObjects(), 5);
        SHOULD_BE(test_data.getNumClasses(0), 2);
        SHOULD_BE(test_data.getNumClasses(1), 3);
        SHOULD_BE(test_data.getNumClasses(2), 3);
        SHOULD_BE(test_data.getNumClasses(4), 3);
        SHOULD_BE(test_data.getClass(0, 0), 0);
        SHOULD_BE(test_data.getClass(0, 4), 1);
        SHOULD_BE(test_data.getClass(1, 4), 2);
        SHOULD_BE(test_data.getClass(4, 4), 2);
        // TODO: Throw when out of bounds!!!
    }

    VariableTuple<2> vt(test_data);
    SHOULD_BE(vt.numClasses(), 6);
}

#if 0
class Obj {
}

class Variable {
}

template<int N, int K>
class InformationEntropy {
    int n_classes = K ^ N;
public:
    InformationEntropy(int 
    float get() {
        float res = 0.0f;
        for (int i = 0; i < n_classes; i++) {
            res -= get_prob(i) * logf(get_prob(i));
        }
    }
}
#endif

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
