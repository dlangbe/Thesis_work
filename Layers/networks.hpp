#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void run_CNN(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, float *soft_bias_init);

void run_sCNN(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, float *soft_bias_init);

void run_FedAvg(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, 
        float *soft_bias_init, int num_nodes, int batch_size);

void run_sFedAvg(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, 
        float *soft_bias_init, int num_nodes, int batch_size);

void average_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes);
    
void saverage_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes);