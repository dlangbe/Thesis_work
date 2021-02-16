#include "layers.hpp"
#include <algorithm>

void Conv_layer::get_parameters(int *r, int *c, int *num, int *size, int *k, float *learn) {
    *r = rows;
    *c = cols;
    *num = num_filters;
    *size = filter_size;
    *k = colors;
    *learn = learn_rate;
}

Conv_layer::Conv_layer(int in_rows, int in_cols, int in_num_filters, int in_filter_size, int in_colors, float in_learn_rate) {
    rows = in_rows;
    cols = in_cols;
    num_filters = in_num_filters;
    filter_size = in_filter_size;
    colors = in_colors;
    learn_rate = in_learn_rate;
}

Conv_layer::Conv_layer() {}

void Conv_layer::update_duration(double t, bool forward) {
    if (forward) t_forward += t / 1000000000.0;
    else t_back += t / 1000000000.0;
    return;
}

void Conv_layer::forward(float *dest, float *image, float *filters) {
    int r, c, n, i, j, k;

    for (n = 0; n < num_filters; n++) {
        for (r = 0; r < rows-(filter_size-1); r++) {
            for (c = 0; c < cols-(filter_size-1); c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        for (k = 0; k < colors; k++){
                            dest[n * ((rows-(filter_size-1)) * (cols-(filter_size-1))) + (r * (cols-(filter_size-1)) + c)] += 
                                image[((r+i) * cols + (c+j))*colors + k] *
                                filters[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)];
                        }
                    }
                }
            }
        }
    }

    return;
}

void Conv_layer::forwardm(float *dest, float *image, float *filters) {
    int r, c, n, i, j, k;

    for (n = 0; n < num_filters; n++) {
        for (r = 0; r < rows-(filter_size-1); r++) {
            for (c = 0; c < cols-(filter_size-1); c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        for (k = 0; k < colors; k++){
                            dest[n * ((rows-(filter_size-1)) * (cols-(filter_size-1))) + (r * (cols-(filter_size-1)) + c)] += 
                                image[(k*rows*cols) + ((r+i) * cols + (c+j))] *
                                filters[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)];
                        }
                    }
                }
            }
        }
    }

    return;
}

void Conv_layer::back(float *dest, float *gradient, float *last_input) {
    float *holder;
    holder = new float[filter_size * filter_size * colors]();
    int r, c, n, i, j, k;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++) {
                    holder[(k*filter_size*filter_size) + (i * filter_size + j)] = 0;
                }
            }
        }
        for (r = 0; r < rows-(filter_size-1); r++) {
            for (c = 0; c < cols-(filter_size-1); c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        for (k = 0; k < colors; k++) {
                            holder[(k*filter_size*filter_size) + (i * filter_size + j)] += last_input[((r+i) * cols + (c+j))*colors + k] *
                                gradient[n * ((rows-(filter_size-1)) * (cols-(filter_size-1))) + (r * (cols-(filter_size-1)) + c)];
                        }
                    }
                }
            }
        }
        // Write local var to output var
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++){
                    dest[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)] -=
                        learn_rate * holder[(k*filter_size*filter_size) + (i * filter_size + j)];
                }

                //dest[(n*filter_size*filter_size) + (i * filter_size + j)] = holder[i * filter_size + j];
            }
        }
    }
    
    delete[] holder;
    return;
}

void Conv_layer::backm(float *dest, float *weights, float *gradient, float *last_input) {
    float *holder;
    holder = new float[filter_size * filter_size * colors]();
    int r, c, n, i, j, k;
    int filter_offset = filter_size - 1;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++) {
                    holder[(k*filter_size*filter_size) + (i * filter_size + j)] = 0;
                }
            }
        }
        for (r = 0; r < rows-filter_offset; r++) {
            for (c = 0; c < cols-filter_offset; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        for (k = 0; k < colors; k++) {
                            holder[(k*filter_size*filter_size) + (i * filter_size + j)] += last_input[(k*rows*cols) +((r+i) * cols + (c+j))] *
                                gradient[n * ((rows-filter_offset) * (cols-filter_offset)) + (r * (cols-filter_offset) + c)];
                        }
                    }
                }
            }
        }
        // update output gradient
        // for (r = 1-filter_size; r < rows + (filter_size-1); r++) {
        //     for (c = 1-filter_size; c < cols + (filter_size-1); c++) {
        //         for (i = 0; i < filter_size; i++) {
        //             for (j = 0;  j < filter_size; j++) {
        //                 for (k = 0; k < colors; k++) {
        //                     // dest = 8x12x12 (k, r, c), gradient = 8x8x8 (k, r, c), weights = 8x8x5x5 (n, k, r, c)
        //                     if (r+i >= 0 && r+i < rows && c+j >= 0 && c+j < cols) {
        //                         dest[(k*rows*cols) + ((r+i) * cols + (c+j))] += 
        //                             gradient[n * ((rows-(filter_size-1)) * (cols-(filter_size-1))) + ((r+i) * (cols-(filter_size-1)) + (c+j))] *
        //                             weights[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) 
        //                                 + ((filter_size-1-j)*filter_size + (filter_size-1-i))];
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        for (r = 0; r < rows; r++) {                        // rows = 12
            for (c = 0; c < cols; c++) {                    // cols = 12
                for (i = 0; i < filter_size; i++) {         //filter_size = 5
                    for (j = 0;  j < filter_size; j++) {
                        for (k = 0; k < colors; k++) {      // colors = 8
                            // dest = 8x12x12 (k, r, c), gradient = 8x8x8 (k, r, c), weights = 8x8x5x5 (n, k, r, c)
                            if (r+i-filter_offset >= 0 && r+i-filter_offset < (rows-filter_offset) && c+j-filter_offset >= 0 && c+j-filter_offset < (cols-filter_offset)) {
                                dest[(k*rows*cols) + (r * cols + c)] += 
                                    gradient[k * ((rows-filter_offset) * (cols-filter_offset)) + ((r+i-filter_offset) * (cols-filter_offset) + (c+j-filter_offset))] *
                                    weights[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) 
                                        + ((filter_offset-j)*filter_size + (filter_offset-i))];
                            }
                        }
                    }
                }
            }
        }

        // update filter weights
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++){
                    weights[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)] -=
                        learn_rate * holder[(k*filter_size*filter_size) + (i * filter_size + j)];
                }

                //dest[(n*filter_size*filter_size) + (i * filter_size + j)] = holder[i * filter_size + j];
            }
        }
    }
    
    delete[] holder;
    return;
}

void Conv_layer::sback(sfloat *dest, sfloat *gradient,  sfloat *last_input) {
    sfloat *holder;
    holder = new sfloat[filter_size*filter_size];
    int r, c, n, i, j, k;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++) {
                    holder[k * (i * filter_size + j)].convert_in_place(0.0);
                }
            }
        }
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        for (k = 0; k < colors; k++) {
                            holder[(k*filter_size*filter_size) + (i * filter_size + j)] = 
                                holder[(k*filter_size*filter_size) + (i * filter_size + j)] 
                                + (last_input[((r+i)*cols)+(c+j)] * gradient[(n*(rows-2)*(cols-2))+(r*(cols-2)+c)]);
                        }
                    }
                }
            }
        }
        // Write local var to output var
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                for (k = 0; k < colors; k++) {
                    dest[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)] =
                        dest[(n*filter_size*filter_size*colors) + (k*filter_size*filter_size) + (i*filter_size + j)]
                        - (holder[k*(i*filter_size+j)] * learn_rate);
                }
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

Avgpool_layer::Avgpool_layer() {}

void Avgpool_layer::get_parameters(int *r, int *c) {
    *r = rows;
    *c = cols;
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

void Softmax_layer::get_parameters(int *in_len, int*out_len) {
    *in_len = in_length;
    *out_len = out_length;
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

    for (i = 0; i < out_length; i++) {
        for (j = 0; j < in_length; j++) {
            totals[i] += (float) input[j] * weight[j * out_length + i];
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

void forward(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, unsigned char *image, float *filters, 
    unsigned char label, float *out, float *loss, float *acc, float *last_pool_input, float *last_soft_input, 
    float *totals, float *soft_weight, float *soft_bias) {

    int rows, cols, num_filters, filter_size, colors, avgpool_rows, avgpool_cols, softmax_in_length, softmax_out_length;
    float learn_rate;
    double duration;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &colors, &learn_rate);
    avgpool.get_parameters(&avgpool_rows, &avgpool_cols);
    softmax.get_parameters(&softmax_in_length, &softmax_out_length);

     // allocate memory
    float *temp_image, *conv_out, *pool_out;
    temp_image = new float[rows * cols * colors]();
    conv_out = new float[avgpool_rows*avgpool_cols*num_filters]();
    pool_out = new float[softmax_in_length]();

    // normalize input image from -0.5 to 0.5
    normalize_image(image, temp_image, rows, cols, colors);

    // link forward layers
    auto t_start = std::chrono::high_resolution_clock::now();
    conv.forward(conv_out, temp_image, filters);
    // relu(conv_out, avgpool_rows*avgpool_cols*num_filters);
    auto t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    conv.update_duration(duration, true);

    t_start = std::chrono::high_resolution_clock::now();
    avgpool.forward(pool_out, conv_out);
    //relu(pool_out, softmax_in_length);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    avgpool.update_duration(duration, true);

    t_start = std::chrono::high_resolution_clock::now();
    softmax.forward(out, pool_out, totals, soft_weight, soft_bias);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, true);
    
    int i;
    for (i = 0; i < avgpool_rows*avgpool_cols*num_filters; i++) {
        last_pool_input[i] = conv_out[i];
    }
    for (i = 0; i < softmax_in_length; i++) {
        last_soft_input[i] = pool_out[i];
    }
    

    // calculate loss and accuracy
    //printf("%.3f, %.3f\n", out[label], log(out[label]));
    *loss = -1 * log(out[label]);
    float holder = -100.0;
    unsigned char index;
    for (i = 0; i < softmax_out_length; i++) {
        if (out[i] > holder){
            holder = out[i];
            index = (unsigned char) i;
        }
    }
    if (index == label)
        *acc = 1;
    else
        *acc = 0;

    //printf("%d\t%d\n", label, index);

    delete[] temp_image;
    delete[] conv_out;
    delete[] pool_out;
    return;
}

void forwardm(Conv_layer *conv, Avgpool_layer *avgpool, Softmax_layer &softmax, unsigned char *image, float **filters, 
    unsigned char label, float *out, float *loss, float *acc, float **last_conv_input, float **last_pool_input, float *last_soft_input, 
    float *totals, float *soft_weight, float *soft_bias, int conv_layers) {

    int num_filters, filter_size, softmax_in_length, softmax_out_length;
    int *conv_rows, *conv_cols, *avgpool_rows, *avgpool_cols, *colors;
    float learn_rate;
    double duration;

    conv_rows = new int[conv_layers];
    conv_cols = new int[conv_layers];
    avgpool_rows = new int[conv_layers];
    avgpool_cols = new int[conv_layers];
    colors = new int[conv_layers];

     // allocate memory and get parameters
    float *temp_image, **conv_out, **pool_out;
    conv_out = new float * [conv_layers];
    pool_out = new float * [conv_layers];
    for (int i = 0; i < conv_layers; i++) {
        conv[i].get_parameters(&conv_rows[i], &conv_cols[i], &num_filters, &filter_size, &colors[i], &learn_rate);
        avgpool[i].get_parameters(&avgpool_rows[i], &avgpool_cols[i]);
        conv_out[i] = new float[avgpool_rows[i]*avgpool_cols[i]*num_filters]();
        pool_out[i] = new float[(avgpool_rows[i]/2)*(avgpool_cols[i]/2)*num_filters]();

    }
    softmax.get_parameters(&softmax_in_length, &softmax_out_length);
    
    // normalize input image from -0.5 to 0.5
    temp_image = new float[conv_rows[0] * conv_cols[0] * colors[0]]();
    normalize_image(image, temp_image, conv_rows[0], conv_cols[0], colors[0]);
    
    // link forward layers
    for (int i = 0; i < conv_layers; i++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            conv[i].forward(conv_out[i], temp_image, filters[i]);
            //normalize(conv_out[i], avgpool_rows[i]*avgpool_cols[i]*num_filters);
        }
            
        else
            conv[i].forwardm(conv_out[i], pool_out[i-1], filters[i]);
        
        // relu(conv_out[i], avgpool_rows[i]*avgpool_cols[i]*num_filters);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
        conv[i].update_duration(duration, true);
        
        t_start = std::chrono::high_resolution_clock::now();
        avgpool[i].forward(pool_out[i], conv_out[i]);
        // relu(pool_out, softmax_in_length);
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
        avgpool[i].update_duration(duration, true);
    }
    

    auto t_start = std::chrono::high_resolution_clock::now();
    softmax.forward(out, pool_out[conv_layers-1], totals, soft_weight, soft_bias);
    auto t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, true);
    
    for (int n = 0; n < conv_layers; n++) {
        for (int i = 0; i < avgpool_rows[n]*avgpool_cols[n]*num_filters; i++) {
            last_pool_input[n][i] = conv_out[n][i];
        }
        if (n != 0) {
            for (int i = 0; i < conv_rows[n]*conv_cols[n]*num_filters; i++) {
                last_conv_input[n-1][i] = pool_out[n-1][i];
            }
        }
    }
    
    for (int i = 0; i < softmax_in_length; i++) {
        last_soft_input[i] = pool_out[conv_layers-1][i];
    }
    
    
    // calculate loss and accuracy
    //printf("%.3f, %.3f\n", out[label], log(out[label]));
    *loss = -1 * log(out[label]);
    float holder = -100.0;
    unsigned char index;
    for (int i = 0; i < softmax_out_length; i++) {
        if (out[i] > holder){
            holder = out[i];
            index = (unsigned char) i;
        }
    }
    if (index == label)
        *acc = 1;
    else
        *acc = 0;

    //printf("%d\t%d\n", label, index);
    delete[] conv_rows;
    delete[] conv_cols;
    delete[] avgpool_rows;
    delete[] avgpool_cols;
    for (int i = 0; i < conv_layers; i++) {
        delete[] conv_out[i];
        delete[] pool_out[i];
    }
    delete[] conv_out;
    delete[] pool_out;
    delete[] temp_image;
    delete[] conv_out;
    delete[] pool_out;
    return;
}

void train(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, unsigned char *image, float *filters, 
    unsigned char label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    int rows, cols, num_filters, filter_size, colors, avgpool_rows, avgpool_cols, softmax_in_length, softmax_out_length;
    float learn_rate;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &colors, &learn_rate);
    avgpool.get_parameters(&avgpool_rows, &avgpool_cols);
    softmax.get_parameters(&softmax_in_length, &softmax_out_length);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(softmax_out_length, sizeof(float));
    pool_out = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
    totals = (float *) calloc(softmax_out_length, sizeof(float));

    // do forward propagation
    forward(conv, avgpool, softmax, image, filters, label, out, loss, acc,
        last_pool_input, last_soft_input, totals, soft_weight, soft_bias);
    
    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = new float [rows * cols * colors]();
    normalize_image(image, temp_image, rows, cols, colors);
    
    // link backward layers
    auto t_start = std::chrono::high_resolution_clock::now();
    softmax.back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias);
    // relu(soft_out, softmax_in_length);
    auto t_end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, false);

    t_start = std::chrono::high_resolution_clock::now();
    avgpool.back(pool_out, soft_out);
    // relu(pool_out, avgpool_rows*avgpool_cols*num_filters);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    avgpool.update_duration(duration, false);

    t_start = std::chrono::high_resolution_clock::now();
    conv.back(filters, pool_out, temp_image);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    conv.update_duration(duration, false);
    
    delete[] temp_image;
    free(grad);
    free(pool_out);
    free(totals);
    return;
}

void trainm(Conv_layer *conv, Avgpool_layer *avgpool, Softmax_layer &softmax, unsigned char *image, float **filters, 
    unsigned char label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float **last_conv_input, float **last_pool_input, float *last_soft_input, int conv_layers) {

    int rows, cols, num_filters, filter_size, colors, softmax_in_length, softmax_out_length;
    int *avgpool_rows = new int[conv_layers];
    int *avgpool_cols = new int[conv_layers];
    float learn_rate;
    double duration;

     // allocate memory and get parameters
    float **conv_out, **pool_out;
    conv_out = new float * [conv_layers - 1];
    pool_out = new float * [conv_layers];
    for (int i = 0; i < conv_layers; i++) {
        if (i == 0)
            conv[i].get_parameters(&rows, &cols, &num_filters, &filter_size, &colors, &learn_rate);
        avgpool[i].get_parameters(&avgpool_rows[i], &avgpool_cols[i]);
        pool_out[i] = new float[avgpool_rows[i]*avgpool_cols[i]*num_filters]();
        if (i != 0)
            conv_out[i-1] = new float[(avgpool_rows[i-1]*avgpool_cols[i-1]/4)*num_filters]();
    }
    softmax.get_parameters(&softmax_in_length, &softmax_out_length);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *totals;
    grad = (float *) calloc(softmax_out_length, sizeof(float));
    totals = (float *) calloc(softmax_out_length, sizeof(float));

    // do forward propagation
    forwardm(conv, avgpool, softmax, image, filters, label, out, loss, acc,
        last_conv_input, last_pool_input, last_soft_input, totals, soft_weight, soft_bias, conv_layers);
    
    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = new float [rows * cols * colors]();
    normalize_image(image, temp_image, rows, cols, colors);
    
    // link backward layers
    auto t_start = std::chrono::high_resolution_clock::now();
    softmax.back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias);
    // relu(soft_out, softmax_in_length);
    auto t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
    softmax.update_duration(duration, false);

    for (int i = conv_layers - 1; i >= 0; i--) {
        t_start = std::chrono::high_resolution_clock::now();
        if (i == conv_layers - 1)
            avgpool[i].back(pool_out[i], soft_out);
        else
            avgpool[i].back(pool_out[i], conv_out[i]);
        
        // relu(pool_out, avgpool_rows*avgpool_cols*num_filters);
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
        avgpool[i].update_duration(duration, false);

        t_start = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            //normalize(pool_out[i], avgpool_rows[i]*avgpool_cols[i]*num_filters);
            conv[i].back(filters[i], pool_out[i], temp_image);
        }
            
        else
            conv[i].backm(conv_out[i-1], filters[i], pool_out[i], last_conv_input[i-1]);
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count();
        conv[i].update_duration(duration, false);
    }
    
    for (int i = 0; i < conv_layers; i++) {
        if (i != 0)
            delete[] conv_out[i-1];
        delete[] pool_out[i];
    }
    delete[] conv_out;
    delete[] pool_out;
    delete[] temp_image;
    delete[] avgpool_rows;
    delete[] avgpool_cols;
    free(grad);
    free(totals);
    return;
}

void strain(Conv_layer &conv, Avgpool_layer &avgpool, Softmax_layer &softmax, unsigned char *image, float *filters, 
    unsigned char label, float *loss, float *acc, float *soft_weight, float *soft_bias,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    float learn_rate;
    double duration;
    int rows, cols, num_filters, filter_size, colors, avgpool_rows, avgpool_cols, softmax_in_length, softmax_out_length;
    conv.get_parameters(&rows, &cols, &num_filters, &filter_size, &colors, &learn_rate);
    avgpool.get_parameters(&avgpool_rows, &avgpool_cols);
    softmax.get_parameters(&softmax_in_length, &softmax_out_length);

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(softmax_out_length, sizeof(float));
    pool_out = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
    totals = (float *) calloc(softmax_out_length, sizeof(float));

    // do forward propagation
    forward(conv, avgpool, softmax, image, filters, label, out, loss, acc,
        last_pool_input, last_soft_input, totals, soft_weight, soft_bias);

    // set initial gradient
    grad[label] = -1 / out[label];

    // normalize image
    float *temp_image;
    temp_image = new float[rows*cols*colors]();
    normalize_image(image, temp_image, rows, cols, colors);    

    // sfloat conversions
    sfloat *sfilters, *spool_out, *stemp_image;
    sfloat *ssoft_out, *sgrad, *slast_soft_input, *ssoft_weight, *ssoft_bias;

    sfilters = new sfloat[filter_size*filter_size*num_filters];
    spool_out = new sfloat[26*26*num_filters];
    stemp_image = new sfloat[rows*cols*colors];
    ssoft_out = new sfloat[softmax_in_length];
    sgrad = new sfloat[softmax_out_length];
    slast_soft_input = new sfloat[softmax_in_length];
    ssoft_weight = new sfloat[softmax_in_length*softmax_out_length];
    ssoft_bias = new sfloat[softmax_out_length];

    for (int i = 0; i < filter_size*filter_size*num_filters; i++) {
        sfilters[i].convert_in_place(filters[i]);
    }
    // for (int i = 0; i < 26*26*num_filters; i++) {
    //     spool_out[i].convert_in_place(pool_out[i]);
    // }
    for (int i = 0; i < rows*cols; i++) {
        stemp_image[i].convert_in_place(temp_image[i]);
    }
    for (int i = 0; i < softmax_in_length*softmax_out_length; i++) {
        ssoft_weight[i].convert_in_place(soft_weight[i]);
    }
    for (int i = 0; i < softmax_out_length; i++) {
        ssoft_bias[i].convert_in_place(soft_bias[i]);
    }
    for (int i = 0; i < softmax_out_length; i++) {
        sgrad[i].convert_in_place(grad[i]);
    }
    for (int i = 0; i < softmax_in_length; i++) {
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
    for (int i = 0; i < softmax_in_length*softmax_out_length; i++) {  
        soft_weight[i] = ssoft_weight[i].reconstruct();
        //fout << i << '\t' << soft_weight[i] << '\n';
    }
    //fout.close();
    for (int i = 0; i < softmax_out_length; i++) {
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
    delete[] temp_image;
    free(grad);
    free(pool_out);
    free(totals);
    return;
}

void normalize_image(unsigned char *input, float *output, int rows, int cols, int colors) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            for (int k = 0; k < colors; k++) {
                output[(r * cols + c)*colors + k] = ((float) input[(r * cols + c)*colors + k] / 255.0)-0.5;
            }
        }
    }
    return;
}

void normalize(float *values, int size) {
    float max = *std::max_element(values, values+size);
    float min = *std::min_element(values, values+size);
    for (int i = 0; i < size; i++) {
        values[i] = (max - values[i]) / (max - min) - 0.5;
    }
}

// void normalize_back(float *values, float *last_values, int size) {
//     float max = *std::max_element(last_values, values+size);
//     float min = *std::min_element(last_values, values+size);
// }

void relu(float *values, int size) {
    for (int i = 0; i < size; i++) {
        values[i] = (values[i] > 0.0) ? values[i] : 0.000001 * values[i];
    }
    return;
}