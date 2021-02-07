#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void run_CNN(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int colors);

void run_sCNN(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int colors);

void run_FedAvg(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int num_nodes, int batch_size, int colors);

void run_sFedAvg(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int num_nodes, int batch_size, int colors);

void average_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes, int num_classes, int softmax_in_len, int softmax_out_len, int colors);
    
void saverage_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes, int num_classes, int softmax_in_len, int softmax_out_len, int colors);