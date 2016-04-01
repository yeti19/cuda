#include <iostream>
#include <exception>
#include <cassert>

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

class Data {
    int numObjects, numVars, numClasses;
    int *classes;
public:
    Data(std::ostream& ost) {
        ost >> numObjects >> numVars >> numClasses;
        numVars++;
        classes = new int[numObjects * numVars];
        for (int i = 0; i < numObjects; ++i) {
            for (int j = 0; j < numVars; ++j)
                ost >> classes[i * numObjects + j];
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
    int getNumClasses(int varNum) const {
        if (varNum == 0) return 2;
        return numClasses;
    }
    int getClass(int varNum, int objNum) const {
        return classes[varNum * numObjects + objNum];
    }
}

/* TODO: Test Data class?? */

template<int N>
class VariableTuple {
    IterableTuple<N> vars;
    Data &data;
public:
    VariableTuple(Data &data) : data(data), vars(data.getNumVars()) { }

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
            ret -= prob * logf(prob);
        }
        return ret;
    }

#if 0
    float IG() {
        
    }

    float GIG() {
        return IG(vt) - maximum((IG(ldksf));)
    }
#endif

    const VariableTuple<N>& operator++() {
        vars++;
        return *this;
    }
    operator bool() { return vars[0] == 0; }
}

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
        assert(test_data.getNumObjects() == 5);
        assert(test_data.getNumClasses(0) == 2);
        assert(test_data.getNumClasses(1) == 3);
        assert(test_data.getNumClasses(2) == 3);
        assert(test_data.getNumClasses(4) == 3);
        assert(test_data.getClass(0, 0) == 0);
        assert(test_data.getClass(
    }
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
