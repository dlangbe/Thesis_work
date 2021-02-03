#include "secure_float.hpp"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <chrono>

struct RGB {
    unsigned char r, g, b;
};

class Conv_layer {
    private:
        int rows, cols, num_filters, filter_size;
        float learn_rate;
        double t_forward = 0.0;
        double t_back = 0.0;

    public:
        Conv_layer(int in_rows, int in_cols, int in_num_filters, int in_filter_size, float in_learn_rate);
        Conv_layer();
        void get_parameters(int *r, int *c, int *num, int *size, float *learn);
        void forward(float *dest, float *image, float *filters);
        void back(float *dest, float *gradient, float *last_input);
        void sback(sfloat *dest, sfloat *gradient,  sfloat *last_input);
        void update_duration(double t, bool forward);
        double get_duration(bool forward) {return forward ? t_forward : t_back;}
};

class Maxpool_layer {
    private:
        int rows, cols, num_filters;
        double duration;
        double t_forward = 0.0;
        double t_back = 0.0;

    public:
        Maxpool_layer(int in_rows, int in_cols, int in_num_filters);
        Maxpool_layer();
        void forward(float *dest, float *input);
        void back(float *dest, float *gradient, float *last_input);
        void update_duration(double t, bool forward);
        double get_duration(bool forward) {return forward ? t_forward : t_back;}
};

class Avgpool_layer {
    private:
        int rows, cols, num_filters;
        double t_forward = 0.0;
        double t_back = 0.0;

    public:
        Avgpool_layer(int in_rows, int in_cols, int in_num_filters);
        Avgpool_layer();
        void get_parameters(int *r, int *c);
        void forward(float *dest, float *input);
        void back(float *dest, float *gradient);
        void sback(sfloat *dest, sfloat *gradient);
        void update_duration(double t, bool forward);
        double get_duration(bool forward) {return forward ? t_forward : t_back;}
};

class Softmax_layer {
    private:
        int in_length, out_length;
        float learn_rate, sum;
        float *exp_holder;
        double t_forward = 0.0;
        double t_back = 0.0;
    
    public:
        Softmax_layer(int in_in_length, int in_out_length, float in_learn_rate);
        Softmax_layer();
        void get_parameters(int *in_len, int *out_len);
        void forward(float *dest, float *input, float *totals, float *weight, float *bias);
        void back(float *dest, float *gradient, float *last_input, float *totals, float *weights, float *bias);
        void sback(sfloat *dest, sfloat *gradient, sfloat *last_input, sfloat *weights, sfloat *bias);
        void update_duration(double t, bool forward);
        double get_duration(bool forward) {return forward ? t_forward : t_back;}
        
};

void forward(Conv_layer &conv, Avgpool_layer &maxpool, Softmax_layer &softmax, RGB *image, float *filters, 
    unsigned char label, float *out, float *loss, float *acc, float *last_pool_input, float *last_soft_input, 
    float *totals, float *soft_weight, float *soft_bias);

void train(Conv_layer &conv, Avgpool_layer &maxpool, Softmax_layer &softmax, RGB *image, float *filters, 
    unsigned char label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input);

void strain(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, RGB *image, float *filters, 
    unsigned char label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input);