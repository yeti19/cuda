#include "util.h"
#include <stdio.h>

int main() {
    SyncBitArray test(10);
    test.print();
    test.setHost(0, 1);
    test.print();
    test.setHost(4, 1);
    test.print();
    test.setHost(3, 1);
    test.print();
    test.setHost(4, 0);
    test.print();

    Sync2BitArray test2(10);
    test2.print();
    test2.setHost(0, 1);
    test2.print();
    test2.setHost(0, 2);
    test2.print();
    test2.setHost(0, 0);
    test2.print();
    test2.setHost(3, 1);
    test2.print();
    test2.setHost(3, 0);
    test2.print();
    test2.setHost(9, 2);
    test2.print();
}