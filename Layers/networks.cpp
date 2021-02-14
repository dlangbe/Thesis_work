#include "layers.hpp"
#include "networks.hpp"
//#include "secure_float.hpp"

void run_CNN(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int colors) {
    
    int avgpool_rows, avgpool_cols, softmax_in_len, softmax_out_len;
    avgpool_rows = image_rows - (filter_size - 1);
    avgpool_cols = image_cols - (filter_size - 1);
    softmax_in_len = (avgpool_rows/2) * (avgpool_cols/2) * num_filters;
    softmax_out_len = num_classes;

    /************************************************ initialize layers ************************************************/

    Conv_layer conv(image_rows, image_cols, num_filters, filter_size, colors, learning_rate);
    //Maxpool_layer maxpool(avgpool_rows, avgpool_cols, num_filters);
    Avgpool_layer avgpool(avgpool_rows, avgpool_cols, num_filters);
    Softmax_layer softmax(softmax_in_len, softmax_out_len, learning_rate);

    printf("** Layers initialized **\n");

    /************************************************ training ************************************************/

    // accuracy variables
    int i;
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
    out = (float *) calloc(num_classes, sizeof(float));
    soft_out = (float *) calloc(softmax_in_len, sizeof(float));
    last_pool_input = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
    last_soft_input = (float *) calloc(softmax_in_len, sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();

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

            // train(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            train(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                    soft_weight_init, soft_bias_init, out, soft_out,
                    last_pool_input, last_soft_input);
            total_loss += *loss_p;
            num_correct += *accuracy_p;

        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
        conv.get_duration(true), avgpool.get_duration(true), softmax.get_duration(true));
        printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
        conv.get_duration(false), avgpool.get_duration(false), softmax.get_duration(false));

        // reset times
        epoch_duration = 0.0;
        conv.update_duration(0.0, true);
        conv.update_duration(0.0, false);
        avgpool.update_duration(0.0, true);
        avgpool.update_duration(0.0, false);
        softmax.update_duration(0.0, true);
        softmax.update_duration(0.0, false);
    }

    /************************************************ testing ************************************************/
    // reset accuracy variables
    total_loss = 0.0;
    num_correct = 0.0;
    float count = 0.0;
    *loss_p = 0.0;
    *accuracy_p = 0.0;

    printf("\nTesting:\n");
    auto test_start = std::chrono::high_resolution_clock::now();

    for (i = num_train; i < num_images; i++) {
        // re-initialize totals for each image
        float *totals;
        totals = (float *) calloc(num_classes, sizeof(float));

        // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, image_rows, image_cols, 8, 3, last_pool_input,
        //     last_soft_input, totals, soft_weight_init, soft_bias_init);

        forward(conv, avgpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p,
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

    auto test_end = std::chrono::high_resolution_clock::now();
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    return;
}

void run_mCNN(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float **filters_init, 
    float *soft_weight_init, float *soft_bias_init, int colors, int conv_layers) {

    /* for 2 conv layers:
        conv[0] in = 32 * 32
        avgpool[0] in = 28 * 28 * 8
        conv[1] in = 14 * 14 * 8
        avgpool[1] in = 10 * 10 * 8
        soft in = 5 * 5 * 8
    */

    int *conv_rows = new int[conv_layers];
    int *conv_cols = new int[conv_layers];
    int *avgpool_rows = new int[conv_layers];
    int *avgpool_cols = new int[conv_layers];
    int softmax_in_len, softmax_out_len;

    for (int i = 0; i < conv_layers; i++) {
        if (i == 0) {
            conv_rows[i] = image_rows;
            conv_cols[i] = image_cols;
            avgpool_rows[i] = image_rows - (filter_size - 1);
            avgpool_cols[i] = image_cols - (filter_size - 1);
        }
        else {
            conv_rows[i] = avgpool_rows[i-1] / 2;
            conv_cols[i] = avgpool_cols[i-1] / 2;
            avgpool_rows[i] = conv_rows[i] - (filter_size - 1);
            avgpool_cols[i] = conv_cols[i] - (filter_size - 1);
        }
    }
    softmax_in_len = (avgpool_rows[conv_layers-1]/2) * (avgpool_cols[conv_layers-1]/2) * num_filters;
    softmax_out_len = num_classes;

    /************************************************ initialize layers ************************************************/

    Conv_layer *conv = new Conv_layer[conv_layers];
    Avgpool_layer *avgpool = new Avgpool_layer[conv_layers];
    for (int i = 0; i < conv_layers; i++) {
        if (i == 0)
            conv[i] = Conv_layer(conv_rows[i], conv_cols[i], num_filters, filter_size, colors, learning_rate);
        else
            conv[i] = Conv_layer(conv_rows[i], conv_cols[i], num_filters, filter_size, num_filters, learning_rate);
        avgpool[i] = Avgpool_layer(avgpool_rows[i], avgpool_cols[i], num_filters);
    }
    
    //Maxpool_layer maxpool(avgpool_rows, avgpool_cols, num_filters);
    Softmax_layer softmax(softmax_in_len, softmax_out_len, learning_rate);

    printf("** Layers initialized **\n");

    /************************************************ training ************************************************/

    // accuracy variables
    int i;
    float *loss_p, *accuracy_p;
    float loss, accuracy;
    loss = 0;
    accuracy = 0;
    loss_p = &loss;
    accuracy_p = &accuracy;
    float num_correct = 0.0;
    float total_loss = 0.0;

    // declare variables that can be reused
    float *out, *soft_out, **last_pool_input, **last_conv_input, *last_soft_input;
    out = (float *) calloc(num_classes, sizeof(float));
    soft_out = (float *) calloc(softmax_in_len, sizeof(float));
    last_pool_input = new float *[conv_layers];
    last_conv_input = new float *[conv_layers - 1];
    for (i = 0; i < conv_layers; i++) {
        last_pool_input[i] = new float[avgpool_rows[i]*avgpool_cols[i]*num_filters]();
        if (i != 0)
            last_conv_input[i-1] = new float[conv_rows[i-1]*conv_cols[i-1]*num_filters]();
    }
    last_soft_input = (float *) calloc(softmax_in_len, sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();

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

            // train(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            trainm(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                    soft_weight_init, soft_bias_init, out, soft_out,
                    last_conv_input, last_pool_input, last_soft_input, conv_layers);
            total_loss += *loss_p;
            num_correct += *accuracy_p;

        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        double conv_ft, conv_bt, avgpool_ft, avgpool_bt;
        conv_ft = conv_bt = avgpool_ft = avgpool_bt = 0.0;
        for (i = 0; i < conv_layers; i++) {
            conv_ft += conv[i].get_duration(true);
            conv_bt += conv[i].get_duration(false);
            avgpool_ft += avgpool[i].get_duration(true);
            avgpool_bt += avgpool[i].get_duration(false);

            conv[i].update_duration(0.0, true);
            conv[i].update_duration(0.0, false);
            avgpool[i].update_duration(0.0, true);
            avgpool[i].update_duration(0.0, false);
        }
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
        conv_ft, avgpool_ft, softmax.get_duration(true));
        printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
        conv_bt, avgpool_bt, softmax.get_duration(false));

        // reset times
        epoch_duration = 0.0;
        softmax.update_duration(0.0, true);
        softmax.update_duration(0.0, false);
    }

    /************************************************ testing ************************************************/
    // reset accuracy variables
    total_loss = 0.0;
    num_correct = 0.0;
    float count = 0.0;
    *loss_p = 0.0;
    *accuracy_p = 0.0;

    printf("\nTesting:\n");
    auto test_start = std::chrono::high_resolution_clock::now();

    for (i = num_train; i < num_images; i++) {
        // re-initialize totals for each image
        float *totals;
        totals = (float *) calloc(num_classes, sizeof(float));

        // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, image_rows, image_cols, 8, 3, last_pool_input,
        //     last_soft_input, totals, soft_weight_init, soft_bias_init);

        forwardm(conv, avgpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p,
            last_conv_input, last_pool_input, last_soft_input, totals, soft_weight_init, soft_bias_init, conv_layers);

        total_loss += *loss_p;
        num_correct += *accuracy_p;

        if (i > 0 && i % per_print == (per_print-1)) {
            printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
                (float) total_loss/count, (float) num_correct/count);
        }
        count += 1.0;
        free(totals);
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    return;
}

void run_sCNN(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int colors) {

    int avgpool_rows, avgpool_cols, softmax_in_len, softmax_out_len;
    avgpool_rows = image_rows - (filter_size - 1);
    avgpool_cols = image_cols - (filter_size - 1);
    softmax_in_len = (avgpool_rows/2) * (avgpool_cols/2) * num_filters;
    softmax_out_len = num_classes;
    
    /************************************************ initialize layers ************************************************/

    Conv_layer conv(image_rows, image_cols, num_filters, filter_size, colors, learning_rate);
    //Maxpool_layer maxpool(avgpool_rows, avgpool_cols, num_filters);
    Avgpool_layer avgpool(avgpool_rows, avgpool_cols, num_filters);
    Softmax_layer softmax(softmax_in_len, softmax_out_len, learning_rate);

    printf("** Layers initialized **\n");

    /************************************************ training ************************************************/

    // accuracy variables
    int i, j;
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
    out = (float *) calloc(num_classes, sizeof(float));
    soft_out = (float *) calloc(softmax_in_len, sizeof(float));
    last_pool_input = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
    last_soft_input = (float *) calloc(softmax_in_len, sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();

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

            // train(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            strain(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                    soft_weight_init, soft_bias_init, out, soft_out,
                    last_pool_input, last_soft_input);

            total_loss += *loss_p;
            num_correct += *accuracy_p;

        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
        conv.get_duration(true), avgpool.get_duration(true), softmax.get_duration(true));
        printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
        conv.get_duration(false), avgpool.get_duration(false), softmax.get_duration(false));

        // reset times
        epoch_duration = 0.0;
        conv.update_duration(0.0, true);
        conv.update_duration(0.0, false);
        avgpool.update_duration(0.0, true);
        avgpool.update_duration(0.0, false);
        softmax.update_duration(0.0, true);
        softmax.update_duration(0.0, false);
    }

    /************************************************ testing ************************************************/
    // reset accuracy variables
    total_loss = 0.0;
    num_correct = 0.0;
    float count = 0.0;
    *loss_p = 0.0;
    *accuracy_p = 0.0;

    printf("\nTesting:\n");
    auto test_start = std::chrono::high_resolution_clock::now();

    for (i = num_train; i < num_images; i++) {
        // re-initialize totals for each image
        float *totals;
        totals = (float *) calloc(num_classes, sizeof(float));

        // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, image_rows, image_cols, 8, 3, last_pool_input,
        //     last_soft_input, totals, soft_weight_init, soft_bias_init);

        forward(conv, avgpool, softmax, images[i], filters_init, labels[i], out, loss_p, accuracy_p,
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

    auto test_end = std::chrono::high_resolution_clock::now();
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    return;
}

void run_FedAvg(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int num_nodes, int batch_size, int colors) {

    int avgpool_rows, avgpool_cols, softmax_in_len, softmax_out_len;
    avgpool_rows = image_rows - (filter_size - 1);
    avgpool_cols = image_cols - (filter_size - 1);
    softmax_in_len = (avgpool_rows/2) * (avgpool_cols/2) * num_filters;
    softmax_out_len = num_classes;

    /************************************************ initialize layers ************************************************/
    Conv_layer *conv = (Conv_layer *) calloc(num_nodes, sizeof(Conv_layer));
    //Maxpool_layer *maxpool = (Maxpool_layer *) calloc(num_nodes, sizeof(Maxpool_layer));
    Avgpool_layer *avgpool = (Avgpool_layer *) calloc(num_nodes, sizeof(Avgpool_layer));
    Softmax_layer *softmax = (Softmax_layer *) calloc(num_nodes, sizeof(Softmax_layer));

    for (int i = 0; i < num_nodes; i++) {
        conv[i] = Conv_layer(image_rows, image_cols, num_filters, filter_size, colors, learning_rate);
        //maxpool[i] = Maxpool_layer(avgpool_rows, avgpool_cols, num_filters);
        avgpool[i] = Avgpool_layer(avgpool_rows, avgpool_cols, num_filters);
        softmax[i] = Softmax_layer(softmax_in_len, softmax_out_len, learning_rate);
    }

    printf("** Layers initialized **\n");

    // set initial weights for each node
    float **filters = new float*[num_nodes];
    float **soft_weights = new float*[num_nodes];
    float **soft_biases = new float*[num_nodes];

    for (int n = 0; n < num_nodes; n++) {
        filters[n] = new float[num_filters * filter_size * filter_size*colors];
        soft_weights[n] = new float[softmax_in_len*softmax_out_len];
        soft_biases[n] = new float[softmax_out_len];

        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            filters[n][i] = filters_init[i];
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            soft_weights[n][i] = soft_weight_init[i];
        }
        for (int i = 0; i < softmax_out_len; i++) {
            soft_biases[n][i] = soft_bias_init[i];
        }
    }

    /************************************************ training ************************************************/

    // accuracy variables
    int i, j;
    float count = 0.0;
    double average_duration = 0.0;
    float *loss = new float[num_nodes]();
    float *accuracy= new float[num_nodes]();
    float *num_correct = new float[num_nodes]();
    float *total_loss = new float[num_nodes]();

    // declare variables that can be reused
    float **out, **soft_out, **last_pool_input, **last_soft_input;
    out = (float **) calloc(num_nodes, sizeof(float *));
    soft_out = (float **) calloc(num_nodes, sizeof(float *));
    last_pool_input = (float **) calloc(num_nodes, sizeof(float *));
    last_soft_input = (float **) calloc(num_nodes, sizeof(float *));

    for (int n = 0; n < num_nodes; n++) {
        out[n] = (float *) calloc(num_classes, sizeof(float));
        soft_out[n] = (float *) calloc(softmax_in_len, sizeof(float));
        last_pool_input[n] = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
        last_soft_input[n] = (float *) calloc(softmax_in_len, sizeof(float));
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        for (j = 0; j < num_train/num_nodes; j++) {
            for (int n = 0; n < num_nodes; n++) {
                i = n * (num_train/num_nodes) + j;
                if (i > 0 && i % per_print == (per_print-1)) {
                    printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
                        total_loss[n]/per_print, num_correct[n]/per_print);
                    total_loss[n] = 0.0;
                    num_correct[n] = 0.0;
                }

                //printf("Label: %d\n", labels[i]);

                // reset weights for each batch
                if ((i % (num_train/num_nodes)) > 0 && 
                    i % batch_size == 0) {
                    //printf("\n\n\tHERE\n");
                    auto average_start = std::chrono::high_resolution_clock::now();
                    average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes, num_classes,
                    softmax_in_len, softmax_out_len, colors);
                    auto average_end = std::chrono::high_resolution_clock::now();
                    average_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(average_end-average_start).count()/1000000000.0;
                }

                train(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], &loss[n], &accuracy[n],
                        soft_weights[n], soft_biases[n], out[n], soft_out[n],
                        last_pool_input[n], last_soft_input[n]);

                total_loss[n] += loss[n];
                num_correct[n] += accuracy[n];

            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        printf("Federated Averaging completed in: %.6lf seconds\n\n", average_duration);

        // reset times
        for (int n = 0; n < num_nodes; n++) {
            printf("Node %d\n", n);
            printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
            conv[n].get_duration(true), avgpool[n].get_duration(true), softmax[n].get_duration(true));
            printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
            conv[n].get_duration(false), avgpool[n].get_duration(false), softmax[n].get_duration(false));

            conv[n].update_duration(0.0, true);
            conv[n].update_duration(0.0, false);
            avgpool[n].update_duration(0.0, true);
            avgpool[n].update_duration(0.0, false);
            softmax[n].update_duration(0.0, true);
            softmax[n].update_duration(0.0, false);
        }
        
        epoch_duration = 0.0;
        average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes, num_classes, 
            softmax_in_len, softmax_out_len, colors);
        
    }

    // set weights after training
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        filters_init[i] = filters[0][i];
    }
    for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
        soft_weight_init[i] = soft_weights[0][i];
    }
    for (int i = 0; i < softmax_out_len; i++) {
        soft_bias_init[i] = soft_biases[0][i];
    }


    /************************************************ testing ************************************************/
    // reset accuracy variables
    for (i = 0; i < num_nodes; i++) {
        loss[i] = 0.0;
        accuracy[i] = 0.0;
        total_loss[i] = 0.0;
        num_correct[i] = 0.0;
    }   

    printf("\nTesting:\n");
    auto test_start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < 1; n++) {
        for (i = num_train; i < num_images; i++) {
            // re-initialize totals for each image
            float *totals;
            totals = (float *) calloc(num_classes, sizeof(float));

            // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, image_rows, image_cols, 8, 3, last_pool_input,
            //     last_soft_input, totals, soft_weight_init, soft_bias_init);

            forward(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], out[n], &loss[n], &accuracy[n],
                last_pool_input[n], last_soft_input[n], totals, soft_weights[n], soft_biases[n]);

            total_loss[n] += loss[n];
            num_correct[n] += accuracy[n];

            if (i > 0 && i % per_print == (per_print-1)) {
                printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
                    (float) total_loss[n]/count, (float) num_correct[n]/count);
            }
            count += 1.0;
            free(totals);
        }
    }

    auto test_end = std::chrono::high_resolution_clock::now();    
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    delete[] loss;
    delete[] accuracy;
    delete[] total_loss;
    delete[] num_correct;
    return;
}

void run_sFedAvg(unsigned char **images, unsigned char *labels, int num_images, int image_rows, int image_cols, int num_classes, int num_train, 
    float learning_rate, int per_print, int num_epochs, int num_filters, int filter_size, float *filters_init, 
    float *soft_weight_init, float *soft_bias_init, int num_nodes, int batch_size, int colors) {

    int avgpool_rows, avgpool_cols, softmax_in_len, softmax_out_len;
    avgpool_rows = image_rows - (filter_size - 1);
    avgpool_cols = image_cols - (filter_size - 1);
    softmax_in_len = (avgpool_rows/2) * (avgpool_cols/2) * num_filters;
    softmax_out_len = num_classes;

    /************************************************ initialize layers ************************************************/
    Conv_layer *conv = (Conv_layer *) calloc(num_nodes, sizeof(Conv_layer));
    //Maxpool_layer *maxpool = (Maxpool_layer *) calloc(num_nodes, sizeof(Maxpool_layer));
    Avgpool_layer *avgpool = (Avgpool_layer *) calloc(num_nodes, sizeof(Avgpool_layer));
    Softmax_layer *softmax = (Softmax_layer *) calloc(num_nodes, sizeof(Softmax_layer));

    for (int i = 0; i < num_nodes; i++) {
        conv[i] = Conv_layer(image_rows, image_cols, num_filters, filter_size, colors, learning_rate);
        //maxpool[i] = Maxpool_layer(avgpool_rows, avgpool_cols, num_filters);
        avgpool[i] = Avgpool_layer(avgpool_rows, avgpool_cols, num_filters);
        softmax[i] = Softmax_layer(softmax_in_len, softmax_out_len, learning_rate);
    }

    printf("** Layers initialized **\n");

    // set initial weights for each node
    float **filters = new float*[num_nodes];
    float **soft_weights = new float*[num_nodes];
    float **soft_biases = new float*[num_nodes];

    for (int n = 0; n < num_nodes; n++) {
        filters[n] = new float[num_filters * filter_size * filter_size * colors];
        soft_weights[n] = new float[softmax_in_len*softmax_out_len];
        soft_biases[n] = new float[softmax_out_len];

        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            filters[n][i] = filters_init[i];
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            soft_weights[n][i] = soft_weight_init[i];
        }
        for (int i = 0; i < softmax_out_len; i++) {
            soft_biases[n][i] = soft_bias_init[i];
        }
    }

    /************************************************ training ************************************************/

    // accuracy variables
    int i, j;
    float count = 0.0;
    double average_duration = 0.0;
    float *loss = new float[num_nodes]();
    float *accuracy= new float[num_nodes]();
    float *num_correct = new float[num_nodes]();
    float *total_loss = new float[num_nodes]();

    // declare variables that can be reused
    float **out, **soft_out, **last_pool_input, **last_soft_input;
    out = (float **) calloc(num_nodes, sizeof(float *));
    soft_out = (float **) calloc(num_nodes, sizeof(float *));
    last_pool_input = (float **) calloc(num_nodes, sizeof(float *));
    last_soft_input = (float **) calloc(num_nodes, sizeof(float *));

    for (int n = 0; n < num_nodes; n++) {
        out[n] = (float *) calloc(num_classes, sizeof(float));
        soft_out[n] = (float *) calloc(softmax_in_len, sizeof(float));
        last_pool_input[n] = (float *) calloc(avgpool_rows*avgpool_cols*num_filters, sizeof(float));
        last_soft_input[n] = (float *) calloc(softmax_in_len, sizeof(float));
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        for (j = 0; j < num_train/num_nodes; j++) {
            for (int n = 0; n < num_nodes; n++) {
                i = n * (num_train/num_nodes) + j;
                if (i > 0 && i % per_print == (per_print-1)) {
                    printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
                        total_loss[n]/per_print, num_correct[n]/per_print);
                    total_loss[n] = 0.0;
                    num_correct[n] = 0.0;
                }

                //printf("Label: %d\n", labels[i]);

                // reset weights for each batch
                if ((i % (num_train/num_nodes)) > 0 && 
                    i % batch_size == 0) {
                    //printf("\n\n\tHERE\n");
                    auto average_start = std::chrono::high_resolution_clock::now();
                    saverage_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes, num_classes,
                    softmax_in_len, softmax_out_len, colors);
                    auto average_end = std::chrono::high_resolution_clock::now();
                    average_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(average_end-average_start).count()/1000000000.0;
                }

                strain(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], &loss[n], &accuracy[n],
                        soft_weights[n], soft_biases[n], out[n], soft_out[n],
                        last_pool_input[n], last_soft_input[n]);

                total_loss[n] += loss[n];
                num_correct[n] += accuracy[n];

            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        printf("Federated Averaging completed in: %.6lf seconds\n\n", average_duration);

        // reset times
        for (int n = 0; n < num_nodes; n++) {
            printf("Node %d\n", n);
            printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
            conv[n].get_duration(true), avgpool[n].get_duration(true), softmax[n].get_duration(true));
            printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
            conv[n].get_duration(false), avgpool[n].get_duration(false), softmax[n].get_duration(false));

            conv[n].update_duration(0.0, true);
            conv[n].update_duration(0.0, false);
            avgpool[n].update_duration(0.0, true);
            avgpool[n].update_duration(0.0, false);
            softmax[n].update_duration(0.0, true);
            softmax[n].update_duration(0.0, false);
        }
        
        epoch_duration = 0.0;
        average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes, num_classes, 
            softmax_in_len, softmax_out_len, colors);
        
    }

    // set weights after training
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        filters_init[i] = filters[0][i];
    }
    for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
        soft_weight_init[i] = soft_weights[0][i];
    }
    for (int i = 0; i < softmax_out_len; i++) {
        soft_bias_init[i] = soft_biases[0][i];
    }


    /************************************************ testing ************************************************/
    // reset accuracy variables
    for (i = 0; i < num_nodes; i++) {
        loss[i] = 0.0;
        accuracy[i] = 0.0;
        total_loss[i] = 0.0;
        num_correct[i] = 0.0;
    }   

    printf("\nTesting:\n");
    auto test_start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < 1; n++) {
        for (i = num_train; i < num_images; i++) {
            // re-initialize totals for each image
            float *totals;
            totals = (float *) calloc(num_classes, sizeof(float));

            // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, image_rows, image_cols, 8, 3, last_pool_input,
            //     last_soft_input, totals, soft_weight_init, soft_bias_init);

            forward(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], out[n], &loss[n], &accuracy[n],
                last_pool_input[n], last_soft_input[n], totals, soft_weights[n], soft_biases[n]);

            total_loss[n] += loss[n];
            num_correct[n] += accuracy[n];

            if (i > 0 && i % per_print == (per_print-1)) {
                printf("step %d: past %d steps test loss: %.3f | test accuracy: %.3f\n", i+1, per_print,
                    (float) total_loss[n]/count, (float) num_correct[n]/count);
            }
            count += 1.0;
            free(totals);
        }
    }

    auto test_end = std::chrono::high_resolution_clock::now();    
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    delete[] loss;
    delete[] accuracy;
    delete[] total_loss;
    delete[] num_correct;
    return;
}

void average_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes, int num_classes, int softmax_in_len, int softmax_out_len, int colors) {
    // average holders
    float *avg_filters = new float[num_filters * filter_size * filter_size * colors]();
    float *avg_soft_weights = new float[softmax_in_len*softmax_out_len]();
    float *avg_soft_biases = new float[softmax_out_len]();

    // average each node's weights and biases
    for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_filters[i] += filters[n][i];  
        }
        avg_filters[i] /= num_nodes;  
    }

    for (int i = 0; i < softmax_in_len*10; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_weights[i] += soft_weights[n][i];
        }
        avg_soft_weights[i] /= num_nodes;
    }

    for (int i = 0; i < 10; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_biases[i] += soft_biases[n][i];
        }
        avg_soft_biases[i] /= num_nodes;
    }        

    // update each node's weights and biases
    for (int n = 0; n < num_nodes; n++) {
        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            filters[n][i] = avg_filters[i];
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            soft_weights[n][i] = avg_soft_weights[i];
        }
        for (int i = 0; i < softmax_out_len; i++) {
            soft_biases[n][i] = avg_soft_biases[i];
        }
    }

    delete[] avg_filters;
    delete[] avg_soft_weights;
    delete[] avg_soft_biases;

    return;
}

void saverage_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, 
    int num_nodes, int num_classes, int softmax_in_len, int softmax_out_len, int colors) {
    // average holders
    sfloat *avg_filters = new sfloat[num_filters * filter_size * filter_size * colors]();
    sfloat *avg_soft_weights = new sfloat[softmax_in_len*softmax_out_len]();
    sfloat *avg_soft_biases = new sfloat[softmax_out_len]();

    // allocate secure values
    sfloat **sfilters = new sfloat*[num_nodes];
    sfloat **ssoft_weights = new sfloat*[num_nodes];
    sfloat **ssoft_biases = new sfloat*[num_nodes];
    
    for (int n = 0; n < num_nodes; n++) {
        sfilters[n] = new sfloat[num_filters * filter_size * filter_size *colors]();
        ssoft_weights[n] = new sfloat[softmax_in_len*softmax_out_len]();
        ssoft_biases[n] = new sfloat[softmax_out_len]();

        // set secure values
        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            sfilters[n][i].convert_in_place(filters[n][i]);  
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            ssoft_weights[n][i].convert_in_place(soft_weights[n][i]);  
        }
        for (int i = 0; i < softmax_out_len; i++) {
            ssoft_biases[n][i].convert_in_place(soft_biases[n][i]);
        }
    }   

    // initialize average values
    for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
        avg_filters[i].convert_in_place(0.0);  
    }
    for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
        avg_soft_weights[i].convert_in_place(0.0);  
    }
    for (int i = 0; i < softmax_out_len; i++) {
        avg_soft_biases[i].convert_in_place(0.0);
    }


    // average each node's weights and biases
    for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_filters[i] += sfilters[n][i];  
        }
        avg_filters[i] /= num_nodes;  
    }

    for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_weights[i] += ssoft_weights[n][i];
        }
        avg_soft_weights[i] /= num_nodes;
    }

    for (int i = 0; i < softmax_out_len; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_biases[i] += ssoft_biases[n][i];
        }
        avg_soft_biases[i] /= num_nodes;
    }        

    // update each node's weights and biases
    for (int n = 0; n < num_nodes; n++) {
        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            sfilters[n][i] = avg_filters[i];
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            ssoft_weights[n][i] = avg_soft_weights[i];
        }
        for (int i = 0; i < softmax_out_len; i++) {
            ssoft_biases[n][i] = avg_soft_biases[i];
        }
    }

    // return non-secure values
    for (int n = 0; n < num_nodes; n++) {
        for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
            filters[n][i] = sfilters[n][i].reconstruct();  
        }
        for (int i = 0; i < softmax_in_len*softmax_out_len; i++) {
            soft_weights[n][i] = ssoft_weights[n][i].reconstruct();  
        }
        for (int i = 0; i < softmax_out_len; i++) {
            soft_biases[n][i] = ssoft_biases[n][i].reconstruct();
        }
    }


    for (int n = 0; n < num_nodes; n++) {
        delete[] sfilters[n];
        delete[] ssoft_weights[n];
        delete[] ssoft_biases[n];
    }
    delete[] sfilters;
    delete[] ssoft_weights;
    delete[] ssoft_biases;
    delete[] avg_filters;
    delete[] avg_soft_weights;
    delete[] avg_soft_biases;

    return;
}