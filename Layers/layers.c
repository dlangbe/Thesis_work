#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void conv_forward(float *dest, float *image, float *filters, int rows, int cols,
    int num_filters, int filter_size);

void conv_back(float *dest, float *last_input, float *gradient, int rows, int cols,
    int num_filters, int filter_size, float learn_rate);

void maxpool_forward(float *dest, float *input, int rows, int cols, int num_filters);

void maxpool_back(float *dest, float *input, float *last_input, int rows, int cols, int num_filters);

void softmax_forward(float *dest, float *totals, float *input, float *weight, float *bias,
    int in_length, int out_length);

void softmax_back(float *dest, float *gradient, float *last_input, float *totals, float *weights,
                    float *bias, int in_length, int out_length, float learn_rate);

void forward(int *image, float *filters, int label, float *out, float *loss, float *acc,
    int rows, int cols, int num_filters, int filter_size, float *last_pool_input,
    float *last_soft_input, float *totals, float *soft_weight, float *soft_bias);

void train(int *image, float *filters, int label, float *loss, float *acc,
    int rows, int cols, int num_filters, int filter_size,
    float *soft_weight, float *soft_bias, float learning_rate,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input);

clock_t t_conv_f, t_conv_b, t_pool_f, t_pool_b, t_soft_f, t_soft_b;
clock_t test_start, test_finish;

int main(int argc, char **argv) {
    FILE *fpt, *fpt2, *fout;
    char header[80];
    int rows, cols, max;
    int **images, *labels;
    unsigned char throwaway;
    int r, c, i, j, f;

    // hyperparameters
    int num_images = 60000;
    int num_train = 50000;
    float learning_rate = 0.005;
    int per_print = 1000;
    int num_epochs = 1;

    // timing variables
    clock_t start_t, end_t;
    t_conv_f = t_conv_b = t_pool_f = t_pool_b = t_soft_f = t_soft_b = 0.0;

    // allocate image and label arrays
    images = (int **) calloc(num_images, sizeof(int *));
    for (i = 0; i < num_images; i++) {
        images[i] = (int *) calloc(28*28, sizeof(int));
    }
    labels = (int *) calloc(num_images, sizeof(int));

    // read in images and labels
    fpt = fopen("mnist_images_full.txt", "rb");
    fpt2 = fopen("mnist_labels_full.txt", "rb");
    for (i = 0; i < num_images; i++) {
        fscanf(fpt2, "%d\n", &labels[i]);
        for (j = 0; j < 28*28; j++) {
            fscanf(fpt, "%d,", &images[i][j]);
        }
        fscanf(fpt, "%c", &throwaway);        
    }

    fclose(fpt);
    fclose(fpt2);

    // read in initial filter weights
    float *filters_init;
    filters_init = (float *) calloc(8 * 9, sizeof(float));
    fpt = fopen("filters_init.txt", "rb");
    for (i = 0; i < 8 * 9; i++) {
        fscanf(fpt, "%f\n", &filters_init[i]);
    }
    fclose(fpt);

    // read in initial softmax weights
    float *soft_weight_init;
    soft_weight_init = (float *) calloc(13*13*8*10, sizeof(float));
    fpt = fopen("soft_weights.txt", "rb");
    for (i = 0; i < 13*13*8*10; i++) {
        fscanf(fpt, "%f\n", &soft_weight_init[i]);
    }
    fclose(fpt);

    // set initial softmax biases
    float *soft_bias_init;
    soft_bias_init = (float *) calloc(10, sizeof(float));
    for (i = 0; i < 10; i++) {
        soft_bias_init[i] = 0.0;
    }

    /************************************************ training ************************************************/
    int epoch;

    // accuracy variables
    float *loss_p, *accuracy_p;
    float loss, accuracy;
    loss = 0;
    accuracy = 0;
    loss_p = &loss;
    accuracy_p = &accuracy;
    float num_correct = 0.0;
    float total_loss = 0.0;

    // declare variables that can be reused
    float *out, *soft_out, *last_pool_input, *last_soft_input;
    out = (float *) calloc(10, sizeof(float));
    soft_out = (float *) calloc(1352, sizeof(float));
    last_pool_input = (float *) calloc(26*26*8, sizeof(float));
    last_soft_input = (float *) calloc(13*13*8, sizeof(float));

    for (epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        start_t = clock();
        for (i = 0; i < num_train; i++) {
            if (i > 0 && i % per_print == (per_print-1)) {
                printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
                    total_loss/per_print, num_correct/per_print);
                total_loss = 0.0;
                num_correct = 0.0;
            }

            train(images[i], filters_init, labels[i], loss_p, accuracy_p, 28, 28, 8, 3,
                    soft_weight_init, soft_bias_init, learning_rate, out, soft_out,
                    last_pool_input, last_soft_input);

                total_loss += *loss_p;
                num_correct += *accuracy_p;

        }
        end_t = clock();
        printf("Epoch completed in: %.3f seconds\n", ((double) (end_t - start_t) / CLOCKS_PER_SEC));
    }
    printf("Duration of individual layer functions:\n");
    printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
        (double)t_conv_f/CLOCKS_PER_SEC, (double)t_pool_f/CLOCKS_PER_SEC, (double)t_soft_f/CLOCKS_PER_SEC);
    printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
        (double)t_conv_b/CLOCKS_PER_SEC, (double)t_pool_b/CLOCKS_PER_SEC, (double)t_soft_b/CLOCKS_PER_SEC);
    /*for (j = 0; j < 10; j++) {
        printf("%.8f\n", out[j]);
    }*/

    /************************************************ testing ************************************************/
    // reset accuracy variables
    total_loss = 0.0;
    num_correct = 0.0;
    float count = 0.0;
    *loss_p = 0.0;
    *accuracy_p = 0.0;

    printf("\nTesting:\n");
    test_start = clock();

    for (i = num_train; i < num_images; i++) {
        // re-initialize totals for each image
        float *totals;
        totals = (float *) calloc(10, sizeof(float));

        forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, 28, 28, 8, 3, last_pool_input,
            last_soft_input, totals, soft_weight_init, soft_bias_init);

        total_loss += *loss_p;
        num_correct += *accuracy_p;

        if (i > 0 && i % per_print == (per_print-1)) {
            printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
                (float) total_loss/count, (float) num_correct/count);
        }
        count += 1.0;
        free(totals);
    }

    test_finish = clock();
    printf("Testing completed in %.6lf seconds\n", ((double) (test_finish - test_start)/CLOCKS_PER_SEC));


    // print sample input image
    fout = fopen("test_out.ppm", "wb");
    fprintf(fout, "P5\n%d\n%d\n%d\n", 28, 28, 255);

    for (r = 0; r < 28; r++) {
        for (c = 0; c < 28; c++) {
            fprintf(fout, "%c", images[0][r * 28 + c]);
        }
    }

    fclose(fout);
    return 0;

}

void conv_forward(float *dest, float *image, float *filters, int rows, int cols,
    int num_filters, int filter_size) {

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

void conv_back(float *dest, float *last_input, float *gradient, int rows, int cols,
    int num_filters, int filter_size, float learn_rate) {

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

void maxpool_forward(float *dest, float *input, int rows, int cols, int num_filters) {
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

void maxpool_back(float *dest, float *input, float *last_input, int rows, int cols, int num_filters) {
    int r, c, n, i, j;
    float holder;
    /*for (i = 0; i < 1352; i++) {
        printf("%.8f\n", last_input[i]);
        if (i != 0 && i % 26 == 0)
            printf("\n");
    }*/

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
                            dest[(n * rows * cols) + ((r*2+i)*cols+(c*2+j))] = input[(n*(rows/2)*(cols/2))+(r*(cols/2)+c)];
                    }
                }
            }
        }
    }

    return;
}

void softmax_forward(float *dest, float *totals, float *input, float *weight, float *bias,
    int in_length, int out_length) {
    int i, j;
    float *exp_holder;
    exp_holder = (float *) calloc(10, sizeof(float));
    float sum = 0.0;
    //printf("%0.6f\t%0.6f\t%0.6f\n\n", input[10], weight[100], input[10]*weight[100]);

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 1352; j++) {
            totals[i] += (float) input[j] * weight[j * 10 + i];
        }
        totals[i] += bias[i];
        exp_holder[i] = exp(totals[i]);
        sum += exp_holder[i];
    }

    for (i = 0; i < out_length; i++) {
        dest[i] = (float) exp_holder[i] / sum;
        //printf("%0.6f\t%0.3f\n", totals[i], bias[i]);
    }

    //printf("\n\n%0.6f\n", weight[101]);
    free(exp_holder);
    return;
}

void softmax_back(float *dest, float *gradient, float *last_input, float *totals, float *weights,
                    float *bias, int in_length, int out_length, float learn_rate) {
    int grad_length = out_length;
    int last_input_length = in_length;
    int i, j, a;
    float *exp_holder;
    exp_holder = (float *) calloc(out_length, sizeof(float));
    float sum = 0.0;
    float *d_out_d_t, *d_L_d_t, *d_L_d_w, *d_L_d_inputs;
    d_out_d_t = (float *) calloc(out_length, sizeof(float));
    //float d_t_d_w[1352]; <= last_input
    d_L_d_t = (float *) calloc(out_length, sizeof(float));
    d_L_d_w = (float *) calloc(in_length * out_length, sizeof(float));
    d_L_d_inputs = (float *) calloc(in_length, sizeof(float));

    for (i = 0; i < grad_length; i++) {
        exp_holder[i] = exp(totals[i]);
        sum += exp_holder[i];
    }

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

    free(exp_holder);
    free(d_out_d_t);
    free(d_L_d_t);
    free(d_L_d_w);
    free(d_L_d_inputs);

    return;
}

void forward(int *image, float *filters, int label, float *out, float *loss, float *acc,
    int rows, int cols, int num_filters, int filter_size, float *last_pool_input,
    float *last_soft_input, float *totals, float *soft_weight, float *soft_bias) {
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
    clock_t start, end;
    start = clock();
    conv_forward(conv_out, temp_image, filters, rows, cols, num_filters, filter_size);
    end = clock();
    t_conv_f += end - start;

    start = clock();
    maxpool_forward(pool_out, conv_out, 26, 26, num_filters);
    end = clock();
    t_pool_f += end - start;

    start = clock();
    softmax_forward(out, totals, pool_out, soft_weight, soft_bias, 1352, 10);
    end = clock();
    t_soft_f += end - start;

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

void train(int *image, float *filters, int label, float *loss, float *acc,
    int rows, int cols, int num_filters, int filter_size,
    float *soft_weight, float *soft_bias, float learning_rate,
    float *out, float *soft_out, float *last_pool_input, float *last_soft_input) {

    // allocate arrays that need to be re-initialized for each image
    float *grad, *pool_out, *totals;
    grad = (float *) calloc(10, sizeof(float));
    pool_out = (float *) calloc(26*26*8, sizeof(float));
    totals = (float *) calloc(10, sizeof(float));

    // do forward propagation
    forward(image, filters, label, out, loss, acc, rows, cols, num_filters, filter_size,
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
    clock_t start, end;
    start = clock();
    softmax_back(soft_out, grad, last_soft_input, totals, soft_weight, soft_bias, 1352, 10, learning_rate);
    end = clock();
    t_soft_b += end - start;

    start = clock();
    maxpool_back(pool_out, soft_out, last_pool_input, rows-2, cols-2, num_filters);
    end = clock();
    t_pool_b += end - start;

    start = clock();
    conv_back(filters, temp_image, pool_out, rows, cols, num_filters, filter_size, learning_rate);
    end = clock();
    t_conv_b += end - start;

    // for (int i = 0; i < 9; i++) printf("%0.12f\n", filters[i]);
    free(temp_image);
    free(grad);
    free(pool_out);
    free(totals);
    return;
}
