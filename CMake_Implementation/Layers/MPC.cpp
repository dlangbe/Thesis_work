#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <climits>
#include <bitset>
#include <iostream>

using namespace std;

// float quantizations
long long int F = 1000000;

// secret quantizations
long long int Q = LLONG_MAX;

// quantize a float from -0.5 to 0.5 to 0 to Q
long long int norm_to_int(float in) {
    //return (int) (Q * (in + 0.5));
    return (long long int) F * in;
}

float int_to_norm(long long int in) {
    //return (float) (in - Q2) / Q;
    return (float) in / F;
}

struct secret_struct {
    long long int share0, share1;
};

struct triple {
    long long int a, b, c;
};

class Secret {
    private:
        long long int share0, share1;

    public:
        Secret(int value) {
            share0 = rand() % Q;
            share1 = (value - share0) % Q;
        }

        Secret() {

        }
        secret_struct share(int in) {
            secret_struct secret;
            share0 = rand() % Q;
            share1 = (in - share0) % Q;
            secret.share0 = share0;
            secret.share1 = share1;
            return secret;
        }

        long long int reconstruct() {
            return (share0 + share1) % Q;
        }

        triple generate_mul_triple() {
            triple res;
            res.a = rand() % Q;
            res.b = rand() % Q;
            res.c = (res.a * res.b) % Q;
            return res;
        }
        // Secret + Secret
        Secret operator + (Secret const &obj) {
            Secret res;
            res.share0 = (share0 + obj.share0) % Q;
            res.share1 = (share1 + obj.share1) % Q;
            return res;
        }

        // Secret + int
        Secret operator + (int x) {
            Secret res;
            res.share0 = (share0 + x) % Q;
            res.share1 = share1;
            return res;
        }

        // Secret - Secret
        Secret operator - (Secret const &obj) {
            Secret res;
            res.share0 = (share0 - obj.share0) % Q;
            res.share1 = (share1 - obj.share1) % Q;
            return res;
        }

        // Secret - int
        Secret operator - (int x) {
            Secret res;
            res.share0 = (share0 - x) % Q;
            res.share1 = share1;
            return res;
        }

        // Secret * Secret
        Secret operator * (Secret const &obj) {
            Secret res;
            Secret temp = obj;
            triple trip = generate_mul_triple();
            long long int a = (*this - trip.a).reconstruct();
            long long int b = (temp - trip.b).reconstruct();
            res = (a*b) + (a*trip.b) + (trip.a*b) + trip.c;
            return res;
        }


        // Secret * int
        Secret operator * (int x) {
            Secret res;
            res.share0 = (share0 * x) % Q;
            res.share1 = (share1 * x) % Q;
            return res;
        }

        // Secret / Secret
        // Secret operator / (Secret const &obj) {
        //     Secret res;
        // }


        // Secret / int
        Secret operator / (int x) {
            Secret res;
            res.share0 = (share0 * 1/x) % Q;
            res.share1 = (share1 * 1/x) % Q;
            return res;
        }

};

int main() {
    Secret test;
    secret_struct test_struct;
    test_struct = test.share(5);

    printf("%lld\n%lld\n", test_struct.share0, test_struct.share1);

    test_struct = test.share(10);

    printf("%lld\n%lld\n", test_struct.share0, test_struct.share1);

    long long int test_out = test.reconstruct();

    printf("reconstructed: %lld\n", test_out);

    Secret t1(120000);
    Secret t2(250000);
    Secret t3;
    // secret_struct s1, s2, s3;
    // s1 = t1.share(25);
    // s2 = t2.share(12);

    t3 = t1 * t2;
    long long int out = t3.reconstruct();

    printf("%lld * %lld = %lld\n", t1.reconstruct(), t2.reconstruct(), out);

    Secret t4;
    secret_struct share4 = t4.share(100);

    long long int i1 = share4.share0;
    long long int i2 = share4.share1;

    long long int i3 = i1 * 10;
    long long int i4 = i2 * 10;
    printf("%lld\n", (i3 + i4) % Q);

    float f = 0.25136645;
    long long int i = norm_to_int(f);
    float g = int_to_norm(i);

    printf("%f => %lld => %f\n", f, i, g);

    float f1 = 0.25;
    float f2 = 0.025;
    Secret m1(norm_to_int(f1));
    Secret m2(norm_to_int(f2));
    Secret m3 = m1 * m2;
    float f3 = int_to_norm(m3.reconstruct());
    printf("%lld + %lld = %lld\n", m1.reconstruct(), m2.reconstruct(), m3.reconstruct());
    printf("%f + %f = %f\n", f1, f2, f3);

    float ft = 0.15625;
    
    char* bits = reinterpret_cast<char*>(&ft);
    for(std::size_t n = 0; n < sizeof ft; ++n)
            std::cout << std::bitset<8>(bits[n]);
    std::cout << '\n';

    int exponent;
    float base, base2;
    base = frexpf(ft, &exponent);
    std::cout << ft << " = " << base << " * 2^" << exponent << "\n";
    cout << ft << " = " << log2(base) << " ^ " << exponent << "\n";
}