#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include "layers.hpp"


void Conv_layer::get_parameters(int *r, int *c, int *num, int *size, float *learn) {
    *r = rows;
    *c = cols;
    *num = num_filters;
    *size = filter_size;
    *learn = learn_rate;
}

Conv_layer::Conv_layer(int in_rows, int in_cols, int in_num_filters, int in_filter_size, float in_learn_rate) {
    rows = in_rows;
    cols = in_cols;
    num_filters = in_num_filters;
    filter_size = in_filter_size;
    learn_rate = in_learn_rate;
}

void Conv_layer::update_duration(double t, bool forward) {
    if (forward) t_forward += t / 1000000000.0;
    else t_back += t / 1000000000.0;
    return;
}

void Conv_layer::forward(float *dest, float *image, float *filters) {
    int r, c, n, i, j;

    for (n = 0; n < num_filters; n++) {
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        dest[n * ((rows-2) * (cols-2)) + (r * (cols-2) + c)] += image[(r+i) * cols + (c+j)] *
                            filters[(n*filter_size*filter_size) + (i*filter_size + j)];
                    }
                }
            }
        }
    }

    return;
}

void Conv_layer::back(float *dest, float *gradient, float *last_input) {
    float *holder;
    holder = (float *) calloc(filter_size * filter_size, sizeof(float));
    int r, c, n, i, j;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                holder[i * filter_size + j] = 0;
            }
        }
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        holder[i * filter_size + j] += last_input[((r+i)*cols)+(c+j)] *
                            gradient[(n*(rows-2)*(cols-2))+(r*(cols-2)+c)];
                    }
                }
            }
        }
        // Write local var to output var
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                dest[(n*filter_size*filter_size) + (i * filter_size + j)] -=
                    learn_rate * holder[i * filter_size + j];

                //dest[(n*filter_size*filter_size) + (i * filter_size + j)] = holder[i * filter_size + j];
            }
        }
    }

    return;
}

void Conv_layer::sback(sfloat *dest, sfloat *gradient,  sfloat *last_input) {
    sfloat *holder;
    holder = new sfloat[filter_size*filter_size];
    int r, c, n, i, j;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                holder[i * filter_size + j].convert_in_place(0.0);
            }
        }
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        holder[i * filter_size + j] = holder[i * filter_size + j] +
                        (last_input[((r+i)*cols)+(c+j)] * gradient[(n*(rows-2)*(cols-2))+(r*(cols-2)+c)]);
                    }
                }
            }
        }
        // Write local var to output var
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                dest[(n*filter_size*filter_size) + (i * filter_size + j)] =
                    dest[(n*filter_size*filter_size) + (i * filter_size + j)] - (holder[i * filter_size + j] * learn_rate);

                //dest[(n*filter_size*filter_size) + (i * filter_size + j)] = holder[i * filter_size + j];
            }
        }
    }
    delete[] holder;

    return;
}


Maxpool_layer::Maxpool_layer(int in_rows, int in_cols, int in_num_filters) {
    rows = in_rows;
    cols = in_cols;
    num_filters = in_num_filters;
}

void Maxpool_layer::update_duration(double t, bool forward) {
    if (forward) t_forward += t / 1000000000.0;
    else t_back += t / 1000000000.0;
    return;
}

void Maxpool_layer::forward(float *dest, float *input) {
    int r, c, n, i, j;
    float holder;

    for (r = 0; r < rows/2; r++) {
        for (c = 0; c < cols/2; c++) {
            for (n = 0; n < num_filters; n++) {
                holder = -100.0;
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        if (input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] > holder)
                            holder = input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))];
                    }
                }
                dest[(n * (rows/2) * (cols/2)) + (r*(cols/2) + c)] = holder;
            }
        }
    }

    return;
}

void Maxpool_layer::back(float *dest, float *gradient, float *last_input) {
    int r, c, n, i, j;
    float holder;

    for (r = 0; r < rows/2; r++) {
        for (c = 0; c < cols/2; c++) {
            for (n = 0; n < num_filters; n++) {
                holder = -100.0;
                // find max
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        if (last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] > holder)
                            holder = last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))];
                        /*if (r == 0 && c == 2 && n == 0) {
                            printf("%.15f\t", last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))]);
                        }*/
                    }
                }
                /*if (r == 0 && c == 2 && n == 0) {
                    printf("\n");
                }*/

                // backprop max
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        if (last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] == holder)
                            dest[(n * rows * cols) + ((r*2+i)*cols+(c*2+j))] = gradient[(n*(rows/2)*(cols/2))+(r*(cols/2)+c)];
                    }
                }
            }
        }
    }

    return;
}

Avgpool_layer::Avgpool_layer(int in_rows, int in_cols, int in_num_filters) {
    rows = in_rows;
    cols = in_cols;
    num_filters = in_num_filters;
}

void Avgpool_layer::update_duration(double t, bool forward) {
    if (forward) t_forward += t / 1000000000.0;
    else t_back += t / 1000000000.0;
    return;
}

void Avgpool_layer::forward(float *dest, float *input) {
    int r, c, n, i, j;
    float holder;

    for (r = 0; r < rows/2; r++) {
        for (c = 0; c < cols/2; c++) {
            for (n = 0; n < num_filters; n++) {
                holder = 0.0;
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        holder += input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))];
                    }
                }
                dest[(n * (rows/2) * (cols/2)) + (r*(cols/2) + c)] = holder/4;
            }
        }
    }

    return;
}

void Avgpool_layer::back(float *dest, float *gradient) {
    int r, c, n, i, j;

    for (r = 0; r < rows/2; r++) {
        for (c = 0; c < cols/2; c++) {
            for (n = 0; n < num_filters; n++) {
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        dest[(n * rows * cols) + ((r*2+i)*cols+(c*2+j))] = gradient[(n*(rows/2)*(cols/2))+(r*(cols/2)+c)]/4;
                    }
                }
            }
        }
    }

    return;
}

void Avgpool_layer::sback(sfloat *dest, sfloat *gradient) {
    int r, c, n, i, j;

    for (r = 0; r < rows/2; r++) {
        for (c = 0; c < cols/2; c++) {
            for (n = 0; n < num_filters; n++) {
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        dest[(n * rows * cols) + ((r*2+i)*cols+(c*2+j))] = gradient[(n*(rows/2)*(cols/2))+(r*(cols/2)+c)]/4;
                    }
                }
            }
        }
    }

    return;
}


Softmax_layer::Softmax_layer(int in_in_length, int in_out_length, float in_learn_rate) {
    in_length = in_in_length;
    out_length = in_out_length;
    learn_rate = in_learn_rate;
    exp_holder = (float *) calloc(out_length, sizeof(float));
    //sum = 0.0;
}

void Softmax_layer::update_duration(double t, bool forward) {
    if (forward) t_forward += t / 1000000000.0;
    else t_back += t / 1000000000.0;
    return;
}

void Softmax_layer::forward(float *dest, float *input, float *totals, float *weight, float *bias) {
    int i, j;
    //float *loc_exp_holder = (float *) calloc(out_length, sizeof(float));
    //float loc_sum = 0.0;
    sum = 0.0;
    //printf("%0.6f\t%0.6f\t%0.6f\n\n", input[10], weight[100], input[10]*weight[100]);

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 1352; j++) {
            totals[i] += (float) input[j] * weight[j * 10 + i];
        }
        totals[i] += bias[i];
        exp_holder[i] = exp(totals[i]);
        sum += exp_holder[i];
    }
    //printf("%0.12f\t", sum);
    for (i = 0; i < out_length; i++) {
        dest[i] = (float) exp_holder[i] / sum;
        //printf("%0.6f\t%0.3f\n", totals[i], bias[i]);
    }
    //update(sum, loc_exp_holder);
    //free(loc_exp_holder);
    //printf("\n\n%0.6f\n", weight[101]);
    return;
}

void Softmax_layer::back(float *dest, float *gradient, float *last_input, float *totals, float *weights, float *bias) {
    int grad_length = out_length;
    int last_input_length = in_length;
    int i, j, a;
    
    float *d_out_d_t, *d_L_d_t, *d_L_d_w, *d_L_d_inputs;
    d_out_d_t = (float *) calloc(out_length, sizeof(float));
    //float d_t_d_w[1352]; <= last_input
    d_L_d_t = (float *) calloc(out_length, sizeof(float));
    d_L_d_w = (float *) calloc(in_length * out_length, sizeof(float));
    d_L_d_inputs = (float *) calloc(in_length, sizeof(float));

    // for (i = 0; i < grad_length; i++) {
    //     exp_holder[i] = exp(totals[i]);
    //     sum += exp_holder[i];
    // }
    
    // find index of gradient that != 0
    for (i = 0; i < grad_length; i++) {
        if (gradient[i] != 0) {
            a = i;
            break;
        }
    }

    // gradients of out[i] against totals
    for (i = 0; i < grad_length; i++) {
        d_out_d_t[i] = -1* exp_holder[a] * exp_holder[i] / (sum * sum);
    }
    d_out_d_t[a] = exp_holder[a] * (sum - exp_holder[a]) / (sum * sum);

    // gradients of totals against weights/ biases/ input

    // gradients of loss against totals
    for (i = 0; i < grad_length; i++) {
        d_L_d_t[i] = gradient[a] * d_out_d_t[i];
        bias[i] -= learn_rate * d_L_d_t[i];
    }

    // gradients of loss against weights/ biases/ input
    for (i = 0; i < last_input_length; i++) {
        for (j = 0; j < grad_length; j++) {
            d_L_d_w[i * grad_length + j] = last_input[i] * d_L_d_t[j];
            d_L_d_inputs[i] += weights[i * grad_length + j] * d_L_d_t[j];
            weights[i * grad_length + j] -= learn_rate * d_L_d_w[i * grad_length + j];
            //dest[i * grad_length + j] = d_L_d_w[i * grad_length + j];
        }
        dest[i] = d_L_d_inputs[i];
    }

    free(d_out_d_t);
    free(d_L_d_t);
    free(d_L_d_w);
    free(d_L_d_inputs);

    return;
}

void Softmax_layer::sback(sfloat *dest, sfloat *gradient, sfloat *last_input, sfloat *weights, sfloat *bias) {
    int grad_length = out_length;
    int last_input_length = in_length;
    int i, j, a;
    // ofstream fout;
    // fout.open("D:/Documents/Research/Palmetto/NIWC-Clemson/Layers/sback_print.txt");
    
    sfloat *d_out_d_t, *d_L_d_t, *d_L_d_w, *d_L_d_inputs;
    d_out_d_t = new sfloat[out_length];
    //float d_t_d_w[1352]; <= last_input
    d_L_d_t = new sfloat[out_length];
    d_L_d_w = new sfloat[in_length * out_length];
    d_L_d_inputs = new sfloat [in_length];
    
    // find index of gradient that != 0
    sfloat zero(0.0);
    for (i = 0; i < grad_length; i++) {
        if (gradient[i] != zero) {
            a = i;
            //printf("\t\ta = %d\n", a);
            break;
        }
    }

    // initialize d_L_d_inputs
    for (i = 0; i < last_input_length; i++) {
        d_L_d_inputs[i].convert_in_place(0.0);
    }


    // gradients of out[i] against totals
    for (i = 0; i < grad_length; i++) {
        d_out_d_t[i] = -1* exp_holder[a] * exp_holder[i] / (sum * sum);
    }
    d_out_d_t[a] = exp_holder[a] * (sum - exp_holder[a]) / (sum * sum);

    // gradients of totals against weights/ biases/ input

    // gradients of loss against totals
    for (i = 0; i < grad_length; i++) {
        d_L_d_t[i] = gradient[a] * d_out_d_t[i];
        bias[i] -= d_L_d_t[i] * learn_rate;
    }

    // gradients of loss against weights/ biases/ input
    for (i = 0; i < last_input_length; i++) {
        for (j = 0; j < grad_length; j++) {
            d_L_d_w[i * grad_length + j] = last_input[i] * d_L_d_t[j];
            // temp = (weights[i * grad_length + j] * d_L_d_t[j]);
            // d_L_d_inputs[i] = temp + 0.0;
            d_L_d_inputs[i] = d_L_d_inputs[i] + (weights[i * grad_length + j] * d_L_d_t[j]);
            weights[i * grad_length + j] -= d_L_d_w[i * grad_length + j] * learn_rate;
            // if (i == 0) {
            //     fout << j << '\t' << weights[i * grad_length + j].reconstruct() << '\t' << d_L_d_t[j].reconstruct() 
            //         << '\t' << temp.reconstruct() << '\n' << d_L_d_inputs[i].reconstruct() << '\n';
            //     d_L_d_inputs[i].print_values();
            // }
        }
        //fout << i << '\t' << d_L_d_inputs[i].reconstruct() << '\n';
        dest[i] = d_L_d_inputs[i];
    }

    delete[] d_out_d_t;
    delete[] d_L_d_t;
    delete[] d_L_d_w;
    delete[] d_L_d_inputs;

    return;
}

void forward(Conv_layer &conv, Maxpool_layer &maxpool, Softmax_layer &softmax, int *image, float *filters, 
    int label, float *out, float *loss, float *acc, float *last_pool_input, float *last_soft_input, 
    float *totals, float *soft_weight, float *soft_bias) {

    int rows, cols, num_filters, filter_size;
    float learn_rate;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &learn_rate);

    // allocate memory
    float *temp_image, *conv_out, *pool_out;
    temp_image = (float *) calloc(rows*cols, sizeof(float));
    conv_out = (float *) calloc((rows-2)*(cols-2)*num_filters, sizeof(float));
    pool_out = (float *) calloc(((rows-2)/2)*((cols-2)/2)*num_filters, sizeof(float));

    // normalize input image from -0.5 to 0.5
    int r, c;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            temp_image[r * cols + c] = ((float) image[r * cols + c] / 255.0) - 0.5;
        }
    }

    // link forward layers

    conv.forward(conv_out, temp_image, filters);
    maxpool.forward(pool_out, conv_out);
    softmax.forward(out, pool_out, totals, soft_weight, soft_bias);

    int i;
    for (i = 0; i < (rows-2)*(cols-2)*num_filters; i++) {
        last_pool_input[i] = conv_out[i];
    }
    for (i = 0; i < ((rows-2)/2)*((cols-2)/2)*num_filters; i++) {
        last_soft_input[i] = pool_out[i];
    }



    // calculate loss and accuracy
    //printf("%.3f, %.3f\n", out[label], log(out[label]));
    *loss = -1 * log(out[label]);
    float holder = -100.0;
    int index;
    for (i = 0; i < 10; i++) {
        if (out[i] > holder){
            holder = out[i];
            index = i;
        }
    }
    if (index == label)
        *acc = 1;
    else
        *acc = 0;

    //printf("%d\t%d\n", label, index);

    free(temp_image);
    free(conv_out);
    free(pool_out);
    return;
}

void forward2(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, int *image, float *filters, 
    int label, float *out, float *loss, float *acc, float *last_pool_input, float *last_soft_input, 
    float *totals, float *soft_weight, float *soft_bias) {

    int rows, cols, num_filters, filter_size;
    float learn_rate;
    double duration;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &learn_rate);

    // allocate memory
    float *temp_image, *conv_out, *pool_out;
    temp_image = (float *) calloc(rows*cols, sizeof(float));
    conv_out = (float *) calloc((rows-2)*(cols-2)*num_filters, sizeof(float));
    pool_out = (float *) calloc(((rows-2)/2)*((cols-2)/2)*num_filters, sizeof(float));

    // normalize input image from -0.5 to 0.5
    int r, c;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            temp_image[r * cols + c] = ((float) image[r * cols + c] / 255.0) - 0.5;
        }
    }

    // link forward layers
    auto t_start = std::chrono::high_resolution_clock::now();
    conv.forward(conv_out, temp_image, filters);
    auto t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    conv.update_duration(duration, true);

    t_start = std::chrono::high_resolution_clock::now();
    avgpool.forward(pool_out, conv_out);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    avgpool.update_duration(duration, true);

    t_start = std::chrono::high_resolution_clock::now();
    softmax.forward(out, pool_out, totals, soft_weight, soft_bias);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, true);

    int i;
    for (i = 0; i < (rows-2)*(cols-2)*num_filters; i++) {
        last_pool_input[i] = conv_out[i];
    }
    for (i = 0; i < ((rows-2)/2)*((cols-2)/2)*num_filters; i++) {
        last_soft_input[i] = pool_out[i];
    }



    // calculate loss and accuracy
    //printf("%.3f, %.3f\n", out[label], log(out[label]));
    *loss = -1 * log(out[label]);
    float holder = -100.0;
    int index;
    for (i = 0; i < 10; i++) {
        if (out[i] > holder){
            holder = out[i];
            index = i;
        }
    }
    if (index == label)
        *acc = 1;
    else
        *acc = 0;

    //printf("%d\t%d\n", label, index);

    free(temp_image);
    free(conv_out);
    free(pool_out);
    return;
}

void train(Conv_layer &conv, Maxpool_layer &maxpool, Softmax_layer &softmax, int *image, float *filters, 
    int label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    int rows, cols, num_filters, filter_size;
    float learn_rate;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &learn_rate);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(10, sizeof(float));
    pool_out = (float *) calloc(26*26*8, sizeof(float));
    totals = (float *) calloc(10, sizeof(float));

    // do forward propagation
    forward(conv, maxpool, softmax, image, filters, label, out, loss, acc,
        last_pool_input, last_soft_input, totals, soft_weight, soft_bias);

    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = (float *) calloc(rows*cols, sizeof(float));
    int r, c;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            temp_image[r * cols + c] = ((float) image[r * cols + c] / 255.0) - 0.5;
        }
    }

    // link backward layers

    softmax.back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias);
    maxpool.back(pool_out, soft_out, last_pool_input);
    conv.back(filters, pool_out, temp_image);


    free(temp_image);
    free(grad);
    free(pool_out);
    free(totals);
    return;
}

void train2(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, int *image, float *filters, 
    int label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    int rows, cols, num_filters, filter_size;
    float learn_rate;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &learn_rate);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(10, sizeof(float));
    pool_out = (float *) calloc(26*26*8, sizeof(float));
    totals = (float *) calloc(10, sizeof(float));

    // do forward propagation
    forward2(conv, avgpool, softmax, image, filters, label, out, loss, acc,
        last_pool_input, last_soft_input, totals, soft_weight, soft_bias);

    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = (float *) calloc(rows*cols, sizeof(float));
    int r, c;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            temp_image[r * cols + c] = ((float) image[r * cols + c] / 255.0) - 0.5;
        }
    }

    // link backward layers

    softmax.back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias);
    avgpool.back(pool_out, soft_out);
    conv.back(filters, pool_out, temp_image);


    free(temp_image);
    free(grad);
    free(pool_out);
    free(totals);
    return;
}

void strain(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, int *image, float *filters, 
    int label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    int rows, cols, num_filters, filter_size;
    float learn_rate;
    double duration;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &learn_rate);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(10, sizeof(float));
    pool_out = (float *) calloc(26*26*8, sizeof(float));
    totals = (float *) calloc(10, sizeof(float));

    // do forward propagation
    forward2(conv, avgpool, softmax, image, filters, label, out, loss, acc,
        last_pool_input, last_soft_input, totals, soft_weight, soft_bias);

    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = (float *) calloc(rows*cols, sizeof(float));
    int r, c;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            temp_image[r * cols + c] = ((float) image[r * cols + c] / 255.0) - 0.5;
        }
    }

    
    

    // sfloat conversions
    sfloat *sfilters, *spool_out, *stemp_image;
    sfloat *ssoft_out, *sgrad, *slast_soft_input, *ssoft_weight, *ssoft_bias;

    sfilters = new sfloat[filter_size*filter_size*num_filters];
    spool_out = new sfloat[26*26*num_filters];
    stemp_image = new sfloat[rows*cols];
    ssoft_out = new sfloat[13*13*num_filters];
    sgrad = new sfloat[10];
    slast_soft_input = new sfloat[13*13*num_filters];
    ssoft_weight = new sfloat[13*13*10*num_filters];
    ssoft_bias = new sfloat[10];

    for (int i = 0; i < filter_size*filter_size*num_filters; i++) {
        sfilters[i].convert_in_place(filters[i]);
    }
    // for (int i = 0; i < 26*26*num_filters; i++) {
    //     spool_out[i].convert_in_place(pool_out[i]);
    // }
    for (int i = 0; i < rows*cols; i++) {
        stemp_image[i].convert_in_place(temp_image[i]);
    }
    for (int i = 0; i < 13*13*10*num_filters; i++) {
        ssoft_weight[i].convert_in_place(soft_weight[i]);
    }
    for (int i = 0; i < 10; i++) {
        ssoft_bias[i].convert_in_place(soft_bias[i]);
    }
    for (int i = 0; i < 10; i++) {
        sgrad[i].convert_in_place(grad[i]);
    }
    for (int i = 0; i < 13*13*num_filters; i++) {
        slast_soft_input[i].convert_in_place(last_soft_input[i]);
    }

    // softmax.back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias);
    // for (int i = 0; i < 13*13*num_filters; i++) {
    //     ssoft_out[i].convert_in_place(soft_out[i]);
    // }

    // ofstream fout;
    // fout.open("D:/Documents/Research/Palmetto/NIWC-Clemson/Layers/test_print.txt");

    // link backward layers
    auto t_start = std::chrono::high_resolution_clock::now();
    softmax.sback(ssoft_out, sgrad, slast_soft_input, ssoft_weight, ssoft_bias);
    auto t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, false);

    t_start = std::chrono::high_resolution_clock::now();
    avgpool.sback(spool_out, ssoft_out);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    avgpool.update_duration(duration, false);

    t_start = std::chrono::high_resolution_clock::now();
    conv.sback(sfilters, spool_out, stemp_image);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    conv.update_duration(duration, false);

    // for (int i = 0; i < 13*13*num_filters; i++)
    //     fout << i << '\t' << ssoft_out[i].reconstruct() << '\n';

    // reconstruct sfloats
    for (int i = 0; i < filter_size*filter_size*num_filters; i++) {
        filters[i] = sfilters[i].reconstruct();
        //if (i >= 9 && i < 18) 
            //printf("%0.12f\n", filters[i]);
    }
    for (int i = 0; i < 13*13*10*num_filters; i++) {  
        soft_weight[i] = ssoft_weight[i].reconstruct();
        //fout << i << '\t' << soft_weight[i] << '\n';
    }
    //fout.close();
    for (int i = 0; i < 10; i++) {
        soft_bias[i] = ssoft_bias[i].reconstruct();
    }

    // for (int i = 0; i < 10; i++) {
    //     printf("%0.12f\n",soft_bias[i]);
    // }



    delete[] ssoft_weight;
    delete[] ssoft_bias;
    delete[] slast_soft_input;
    delete[] sgrad;
    delete[] ssoft_out;
    delete[] sfilters;
    delete[] spool_out;
    delete[] stemp_image;
    free(temp_image);
    free(grad);
    free(pool_out);
    free(totals);
    return;
}