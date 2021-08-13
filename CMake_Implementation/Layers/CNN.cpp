#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
//#include <cstdlib>
#include <time.h>
#include <math.h>
#include <fstream>
#include <chrono>
#include "layers.hpp"
#include "networks.hpp"
#include <fstream>

int main(void) {
    FILE **files;
    // if (fout.is_open()) printf("Is open\n");
    unsigned char **images;
    unsigned char *labels;
    int i, j;
    int z;

    // dataset parameters
    int num_images = 6000;
    int num_train = 5000;
    int per_print = 10;
    int images_per_file = num_images / 2;
    int image_size = 28;
    int num_classes = 10;
    int colors = 1;

    // hyperparameters
    float learning_rate = 0.005;
    int num_epochs = 4;
    int num_filters = 8;
    int filter_size = 5;
    int num_nodes = 1;
    int batch_size = 100;
    int conv_layers = 1;

    int soft_size = (image_size - (filter_size-1)) / 2;
    

    std::srand(1);

    /************************************************ allocate memory ************************************************/

    // allocate image and label arrays
    images = new unsigned char* [num_images];
    for (i = 0; i < num_images; i++) {
        images[i] = new unsigned char [image_size*image_size*colors];
    }
    labels = new unsigned char[num_images];

    // allocate files
    FILE *fpt1, *fpt2, *fpt3;
    unsigned char throwaway;
    int holder = 0;

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
        fscanf(fpt3, "%d\n", &holder);
        labels[i] = (unsigned char) holder;
        for (j = 0; j < 28*28; j++) {
            if (i < 30000) {
                fscanf(fpt1, "%d,", &holder);
                images[i][j] = (unsigned char) holder;
            }
            else {
                fscanf(fpt2, "%d,", &holder);
                images[i][j] = (unsigned char) holder;
            }
        }
        if (i < 30000)
                fscanf(fpt1, "%c", &throwaway);
        else
            fscanf(fpt2, "%c", &throwaway);              
    }

    fclose(fpt1);
    fclose(fpt2);
    fclose(fpt3);

    // create initial filter weights
    printf("** Setting initial filter weights **\n");
    float **filters_init;
    filters_init = new float* [conv_layers];
    for (int n = 0; n < conv_layers; n++) {
        if (n == 0) {
            filters_init[n] = new float[num_filters * filter_size * filter_size * colors];
            for (i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
                filters_init[n][i] = ((float) std::rand() / RAND_MAX) / (filter_size * filter_size);
            }
        }
        else {
            filters_init[n] = new float[num_filters * filter_size * filter_size * num_filters];
            for (i = 0; i < num_filters * filter_size * filter_size * num_filters; i++) {
                filters_init[n][i] = ((float) std::rand() / RAND_MAX) / (filter_size * filter_size);
            }
        }
        
    }

    //printf("test: %0.12f\n", ((float) std::rand() / RAND_MAX));

    // create initial softmax weights
    printf("** Setting initial softmax weights **\n");
    float *soft_weight_init;
    soft_weight_init = new float[soft_size*soft_size*num_filters*num_classes];
    for (i = 0; i < soft_size*soft_size*num_filters*num_classes; i++) {
        soft_weight_init[i] = ((float) std::rand() / RAND_MAX) / (soft_size*soft_size*num_filters);
    }

    // set initial softmax biases
    float *soft_bias_init;
    soft_bias_init = new float[num_classes];
    for (i = 0; i < num_classes; i++) {
        soft_bias_init[i] = 0.0;
    }

    // sfloat test(0.15625);
    // test.print_values();
    // printf("r = %d, g = %d, b = %d\n", (int) images[0][0].r, (int) images[0][0].g, (int) images[0][0].b);
    // printf("r = %d, g = %d, b = %d\n", 
    //     (int) images[59999][1023].r, (int) images[59999][1023].g, (int) images[59999][1023].b);
    // for (i = 0; i < 10; i++) printf("%d\t", (int) labels[i]);
    // printf("\n");
    // for (i = 59990; i < 60000; i++) printf("%d\t", (int) labels[i]);
    // printf("\n");

    run_sCNN(images, labels, num_images, image_size, image_size, num_classes, num_train, learning_rate, per_print, num_epochs, 
                num_filters, filter_size, filters_init[0], soft_weight_init, soft_bias_init, colors);

    // run_FedAvg(images, labels, num_images, num_train, learning_rate, per_print, num_epochs, num_filters, filter_size, 
    //     filters_init, soft_weight_init, soft_bias_init, num_nodes, batch_size);

    /************************************************ initialize layers ************************************************/

    // Conv_layer conv(28, 28, num_filters, filter_size, learning_rate);
    // Maxpool_layer maxpool(26, 26, num_filters);
    // Avgpool_layer avgpool(26, 26, num_filters);
    // Softmax_layer softmax(13*13*num_filters, 10, learning_rate);

    // printf("** Layers initialized **\n");

    // /************************************************ training ************************************************/

    // // accuracy variables
    // float *loss_p, *accuracy_p;
    // float loss, accuracy;
    // loss = 0;
    // accuracy = 0;
    // loss_p = &loss;
    // accuracy_p = &accuracy;
    // float num_correct = 0.0;
    // float total_loss = 0.0;

    // // declare variables that can be reused
    // float *out, *soft_out, *last_pool_input, *last_soft_input;
    // out = (float *) calloc(10, sizeof(float));
    // soft_out = (float *) calloc(1352, sizeof(float));
    // last_pool_input = (float *) calloc(26*26*8, sizeof(float));
    // last_soft_input = (float *) calloc(13*13*8, sizeof(float));

    // for (int epoch = 0; epoch < num_epochs; epoch++) {
    //     printf("--- Epoch %d ---\n", epoch+1);
    //     auto epoch_start = std::chrono::high_resolution_clock::now();

    //     for (i = 0; i < num_train; i++) {
    //         if (i > 0 && i % per_print == (per_print-1)) {
    //             printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
    //                 total_loss/per_print, num_correct/per_print);
    //             total_loss = 0.0;
    //             num_correct = 0.0;
    //         }
    //         //printf("Label: %d\n", labels[i]);

    //         train(conv, maxpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
    //                 soft_weight_init, soft_bias_init, out, soft_out,
    //                 last_pool_input, last_soft_input);

    //         // train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
    //         //         soft_weight_init, soft_bias_init, out, soft_out,
    //         //         last_pool_input, last_soft_input);

    //         // strain(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
    //         //         soft_weight_init, soft_bias_init, out, soft_out,
    //         //         last_pool_input, last_soft_input);

    //         total_loss += *loss_p;
    //         num_correct += *accuracy_p;

    //     }

    //     auto epoch_end = std::chrono::high_resolution_clock::now();
    //     double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
    //     printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
    //     printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
    //     conv.get_duration(true), avgpool.get_duration(true), softmax.get_duration(true));
    //     printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
    //     conv.get_duration(false), avgpool.get_duration(false), softmax.get_duration(false));

    //     // reset times
    //     epoch_duration = 0.0;
    //     conv.update_duration(0.0, true);
    //     conv.update_duration(0.0, false);
    //     avgpool.update_duration(0.0, true);
    //     avgpool.update_duration(0.0, false);
    //     softmax.update_duration(0.0, true);
    //     softmax.update_duration(0.0, false);
    // }

    // /************************************************ testing ************************************************/
    // // reset accuracy variables
    // total_loss = 0.0;
    // num_correct = 0.0;
    // float count = 0.0;
    // *loss_p = 0.0;
    // *accuracy_p = 0.0;

    // printf("\nTesting:\n");
    // auto test_start = std::chrono::high_resolution_clock::now();

    // for (i = num_train; i < num_images; i++) {
    //     // re-initialize totals for each image
    //     float *totals;
    //     totals = (float *) calloc(10, sizeof(float));

    //     forward(conv, maxpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p, last_pool_input,
    //         last_soft_input, totals, soft_weight_init, soft_bias_init);

    //     // forward2(conv, avgpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p,
    //     //     last_pool_input, last_soft_input, totals, soft_weight_init, soft_bias_init);

    //     total_loss += *loss_p;
    //     num_correct += *accuracy_p;

    //     if (i > 0 && i % per_print == (per_print-1)) {
    //         printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
    //             (float) total_loss/count, (float) num_correct/count);
    //     }
    //     count += 1.0;
    //     free(totals);
    // }

    // auto test_end = std::chrono::high_resolution_clock::now();
    // double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    // printf("Testing completed in: %.6lf seconds\n", test_duration);

    return 0;
}