#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <climits>
#include <bitset>
#include <iostream>

using namespace std;

class sfloat {
    /*  
        represents (1 - 2s) * (1 - z) * v * 2 ^ p
        v: base or significand 
        p: exponent
        z: zero bit
        s: sign bit
    */
    private:
        int vlen = 24;
        int plen = 8;
        
        int v, p, z, s;

    public:
        sfloat convert_float(float in) {
            sfloat ret;
            if (in < 0)
                ret.s = 1;
            else ret.s = 0;
            if (in == 0.0) {
                ret.v = 0;
                ret.p = 0;
                ret.z = 1;
            } else {
                //ret.v = frexpf(in, &ret.p);
                ret.p = (int) floor(log2(abs(in))) - vlen + 1;
                ret.v = (int) round(abs(in) * pow(2, (-1*ret.p)));

                if (ret.v == pow(2, vlen)) {
                    ret.p += 1;
                    ret.v /= 2;
                }
                ret.z = 0;
            }
            return ret;
        }

        float reconstruct() {
            return (float) (1 - 2 * s) * (1 - z) * v * pow(2, p);
        }

        // constructors
        sfloat(float in) {
            if (in < 0)
                s = 1;
            else s = 0;
            if (in == 0.0) {
                v = 0;
                p = 0;
                z = 1;
            } else {
                //ret.v = frexpf(in, &ret.p);
                p = (int) floor(log2(abs(in))) - vlen + 1;
                v = (int) round(abs(in) * pow(2, (-1*p)));

                if (v == pow(2, vlen)) {
                    p += 1;
                    v /= 2;
                }
                z = 0;
            };
        }

        sfloat(int in_v, int in_p, int in_z, int in_s) {
            v = in_v;
            p = in_p;
            z = in_z;
            s = in_s;
        }

        sfloat() {}

        void print_values() {
            printf("float = %f, v = %d, p = %d, z = %d, s = %d\n", reconstruct(), v, p, z, s);
        }

        // arithmetic operators
        sfloat operator + (sfloat const &obj) {
            sfloat ret, temp;

            // determine which value has a larger exponent
            if (p >= obj.p) {
                // shift the smaller value so they have the same exponents
                temp.p = p;
                temp.s = s;

                // check for integer overflow
                if (obj.v * pow(2, p - obj.p) < INT_MAX || obj.v * pow(2, p - obj.p) > INT_MIN)
                    temp.v = obj.v / pow(2, p - obj.p);
                else {
                    // figure out what to put here later
                }
                ret.v = abs((1 - 2*s) * v + (1 - 2*obj.s) * temp.v);
                ret.p = temp.p;
                ret.s = temp.s;
                ret.z = 0;
            }
            else {
                temp.p = obj.p;
                temp.s = obj.s;

                // check for integer overflow
                if (v * pow(2, obj.p - p) < INT_MAX || v * pow(2, obj.p - p) > INT_MIN)
                    temp.v = v / pow(2, obj.p - p);
                else {
                    // figure out what to put here later
                }
                ret.v = abs((1 - 2*obj.s) * obj.v + (1 - 2 * s) * temp.v);
                ret.p = temp.p;
                ret.s = temp.s;
                ret.z = 0;
            }
            return ret;
        }

        sfloat operator + (float obj) {
            sfloat ret, temp;
            temp = sfloat(obj);
            ret = *this + temp;
            return ret;
        }

        sfloat operator * (sfloat const &obj) {
            sfloat ret;
            long long int temp = (long long int) v * obj.v;
            
            ret.v = (int) (temp >> vlen);
            ret.p = p + obj.p + vlen;
            ret.s = s ^ obj.s;
            ret.z = z | obj.z;
            return ret;
        }

        sfloat operator * (float obj) {
            sfloat ret, temp;
            temp = sfloat(obj);
            ret = *this * temp;
            return ret;
        }


};

int main() {
    // sfloat test(1);
    // test.print_values();
    // float f = test.reconstruct();
    // printf("float = %0.9f\n", f);

    sfloat a(1000000.0), b(0.000001), c;
    c = a * 0.00125;
    a.print_values();
    b.print_values();
    c.print_values();
    printf("%0.9f + %0.9f = %0.9f", a.reconstruct(), b.reconstruct(), c.reconstruct());
}