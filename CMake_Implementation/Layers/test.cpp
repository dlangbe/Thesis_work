#include <stdio.h>
#include <stdlib.h>
#include "secure_float.hpp"

int main() 
{
    int size = 10;
    sfloat *a, *b, *c, *d;
    a = new sfloat[size];
    b = new sfloat[size];
    c = new sfloat[size];
    d = new sfloat[size];

    for (int i = 0; i < size; i++) {
        a[i].convert_in_place(i);
        b[i].convert_in_place(float(i + 1));
        c[i] = a[i] + b[i];
        d[i] = c[i] + (a[i] * b[i]);
    }

    float *e, *f, *g, *h;
    e = (float *) calloc(size, sizeof(float));
    f = (float *) calloc(size, sizeof(float));
    g = (float *) calloc(size, sizeof(float));
    h = (float *) calloc(size, sizeof(float));

    for (int i = 0; i < size; i++) {
        e[i] = a[i].reconstruct();
        f[i] = b[i].reconstruct();
        g[i] = c[i].reconstruct();
        h[i] = d[i].reconstruct();

        printf("%f\t%f\t%f\t%f\n", e[i], f[i], g[i], h[i]);
    }

    int j = 1;
    sfloat t;
    t.convert_in_place(float(j));
    t.print_values();

    sfloat test(0.111111111111);
    sfloat test2 = test;
    sfloat test3 = test * test2;
    float test_f = 0.111111111111;
    test_f *= test_f;
    printf("test = %0.12f\n", test.reconstruct());
    printf("test3 = %0.12f, real = %0.12f\n", test3.reconstruct(), test_f);
}