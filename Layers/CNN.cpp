#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "layers.hpp"
#include <fstream>

int main(void) {
    FILE *fpt1, *fpt2, *fpt3;
    // if (fout.is_open()) printf("Is open\n");
    int **images, *labels;
    unsigned char throwaway;
    int i, j;

    // hyperparameters
    int num_images = 60000;
    int num_train = 50000;
    float learning_rate = 0.005;
    int per_print = 1000;
    int num_epochs = 1;
    int num_filters = 8;
    int filter_size = 3;

    /************************************************ allocate memory ************************************************/

    // allocate image and label arrays
    images = (int **) calloc(num_images, sizeof(int *));
    for (i = 0; i < num_images; i++) {
        images[i] = (int *) calloc(28*28, sizeof(int));
    }
    labels = (int *) calloc(num_images, sizeof(int));

    // read in images and labels
    printf("** Reading images and labels **\n");
    fpt1 = std::fopen("mnist_images_first.txt", "rb");
    fpt2 = std::fopen("mnist_images_second.txt", "rb");
    fpt3 = std::fopen("mnist_labels_full.txt", "rb");
    if (fpt1 == NULL || fpt2 == NULL) {
        printf("Image file == NULL\n");
        perror("fopen");
    } 
    if (fpt3 == NULL) {
        printf("Label file == NULL\n");
        perror("fopen");
    }
    for (i = 0; i < num_images; i++) {
        fscanf(fpt3, "%d\n", &labels[i]);
        for (j = 0; j < 28*28; j++) {
            if (i < num_images / 2)
                fscanf(fpt1, "%d,", &images[i][j]);
            else
                fscanf(fpt2, "%d,", &images[i][j]);
            
        }
        if (i < num_images / 2)
                fscanf(fpt1, "%c", &throwaway);
        else
            fscanf(fpt2, "%c", &throwaway);              
    }

    fclose(fpt1);
    fclose(fpt2);
    fclose(fpt3);

    // read in initial filter weights
    printf("** Reading initial filter weights **\n");
    float *filters_init;
    filters_init = (float *) calloc(8 * 9, sizeof(float));
    fpt1 = fopen("filters_init.txt", "rb");
    for (i = 0; i < 8 * 9; i++) {
        fscanf(fpt1, "%f\n", &filters_init[i]);
    }
    fclose(fpt1);

    // read in initial softmax weights
    printf("** Reading initial softmax weights **\n");
    float *soft_weight_init;
    soft_weight_init = (float *) calloc(13*13*8*10, sizeof(float));
    fpt1 = fopen("soft_weights.txt", "rb");
    for (i = 0; i < 13*13*8*10; i++) {
        fscanf(fpt1, "%f\n", &soft_weight_init[i]);
    }
    fclose(fpt1);

    // set initial softmax biases
    float *soft_bias_init;
    soft_bias_init = (float *) calloc(10, sizeof(float));
    for (i = 0; i < 10; i++) {
        soft_bias_init[i] = 0.0;
    }

    /************************************************ initialize layers ************************************************/

    Conv_layer conv(28, 28, num_filters, filter_size, learning_rate);
    Maxpool_layer maxpool(26, 26, num_filters);
    Avgpool_layer avgpool(26, 26, num_filters);
    Softmax_layer softmax(13*13*num_filters, 10, learning_rate);

    printf("** Layers initialized **\n");

    /************************************************ training ************************************************/

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

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        for (i = 0; i < num_train; i++) {
            if (i > 0 && i % per_print == (per_print-1)) {
                printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
                    total_loss/per_print, num_correct/per_print);
                total_loss = 0.0;
                num_correct = 0.0;
            }
            //printf("Label: %d\n", labels[i]);

            // train(conv, maxpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            // train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            strain(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                    soft_weight_init, soft_bias_init, out, soft_out,
                    last_pool_input, last_soft_input);

            total_loss += *loss_p;
            num_correct += *accuracy_p;

        }
        //printf("Epoch completed in: %.3f seconds\n", ((double) (end_t - start_t) / CLOCKS_PER_SEC));
    }

    /************************************************ testing ************************************************/
    // reset accuracy variables
    total_loss = 0.0;
    num_correct = 0.0;
    float count = 0.0;
    *loss_p = 0.0;
    *accuracy_p = 0.0;

    printf("\nTesting:\n");
    //test_start = clock();

    for (i = num_train; i < num_images; i++) {
        // re-initialize totals for each image
        float *totals;
        totals = (float *) calloc(10, sizeof(float));

        // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, 28, 28, 8, 3, last_pool_input,
        //     last_soft_input, totals, soft_weight_init, soft_bias_init);

        forward2(conv, avgpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p,
            last_pool_input, last_soft_input, totals, soft_weight_init, soft_bias_init);

        total_loss += *loss_p;
        num_correct += *accuracy_p;

        if (i > 0 && i % per_print == (per_print-1)) {
            printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
                (float) total_loss/count, (float) num_correct/count);
        }
        count += 1.0;
        free(totals);
    }

    //test_finish = clock();
    //printf("Testing completed in %.6lf seconds\n", ((double) (test_finish - test_start)/CLOCKS_PER_SEC));



    return 0;
}