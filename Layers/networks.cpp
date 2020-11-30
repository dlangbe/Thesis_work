#include "layers.hpp"
#include "networks.hpp"
//#include "secure_float.hpp"

void run_CNN(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, float *soft_bias_init) {
    
    
    /************************************************ initialize layers ************************************************/

    Conv_layer conv(28, 28, num_filters, filter_size, learning_rate);
    //Maxpool_layer maxpool(26, 26, num_filters);
    Avgpool_layer avgpool(26, 26, num_filters);
    Softmax_layer softmax(13*13*num_filters, 10, learning_rate);

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
    out = (float *) calloc(10, sizeof(float));
    soft_out = (float *) calloc(1352, sizeof(float));
    last_pool_input = (float *) calloc(26*26*8, sizeof(float));
    last_soft_input = (float *) calloc(13*13*8, sizeof(float));

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

            // train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
            //         soft_weight_init, soft_bias_init, out, soft_out,
            //         last_pool_input, last_soft_input);

            train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
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

    auto test_end = std::chrono::high_resolution_clock::now();
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    return;
}

void run_sCNN(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, float *soft_bias_init) {
    
    
    /************************************************ initialize layers ************************************************/

    Conv_layer conv(28, 28, num_filters, filter_size, learning_rate);
    //Maxpool_layer maxpool(26, 26, num_filters);
    Avgpool_layer avgpool(26, 26, num_filters);
    Softmax_layer softmax(13*13*num_filters, 10, learning_rate);

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
    out = (float *) calloc(10, sizeof(float));
    soft_out = (float *) calloc(1352, sizeof(float));
    last_pool_input = (float *) calloc(26*26*8, sizeof(float));
    last_soft_input = (float *) calloc(13*13*8, sizeof(float));

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

            // train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
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

    auto test_end = std::chrono::high_resolution_clock::now();
    double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
    printf("Testing completed in: %.6lf seconds\n", test_duration);

    return;
}

void run_FedAvg(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, 
        float *soft_bias_init, int num_nodes, int batch_size) {

    /************************************************ initialize layers ************************************************/
    Conv_layer *conv = (Conv_layer *) calloc(num_nodes, sizeof(Conv_layer));
    //Maxpool_layer *maxpool = (Maxpool_layer *) calloc(num_nodes, sizeof(Maxpool_layer));
    Avgpool_layer *avgpool = (Avgpool_layer *) calloc(num_nodes, sizeof(Avgpool_layer));
    Softmax_layer *softmax = (Softmax_layer *) calloc(num_nodes, sizeof(Softmax_layer));

    for (int i = 0; i < num_nodes; i++) {
        conv[i] = Conv_layer(28, 28, num_filters, filter_size, learning_rate);
        //maxpool[i] = Maxpool_layer(26, 26, num_filters);
        avgpool[i] = Avgpool_layer(26, 26, num_filters);
        softmax[i] = Softmax_layer(13*13*num_filters, 10, learning_rate);
    }

    printf("** Layers initialized **\n");

    // set initial weights for each node
    float **filters = new float*[num_nodes];
    float **soft_weights = new float*[num_nodes];
    float **soft_biases = new float*[num_nodes];

    for (int n = 0; n < num_nodes; n++) {
        filters[n] = new float[num_filters * filter_size * filter_size];
        soft_weights[n] = new float[13*13*num_filters*10];
        soft_biases[n] = new float[10];

        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            filters[n][i] = filters_init[i];
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            soft_weights[n][i] = soft_weight_init[i];
        }
        for (int i = 0; i < 10; i++) {
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
        out[n] = (float *) calloc(10, sizeof(float));
        soft_out[n] = (float *) calloc(1352, sizeof(float));
        last_pool_input[n] = (float *) calloc(26*26*8, sizeof(float));
        last_soft_input[n] = (float *) calloc(13*13*8, sizeof(float));
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
                    average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes);
                    auto average_end = std::chrono::high_resolution_clock::now();
                    average_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(average_end-average_start).count()/1000000000.0;
                }

                train2(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], &loss[n], &accuracy[n],
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
        average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes);
        
    }

    // set weights after training
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        filters_init[i] = filters[0][i];
    }
    for (int i = 0; i < 13*13*num_filters*10; i++) {
        soft_weight_init[i] = soft_weights[0][i];
    }
    for (int i = 0; i < 10; i++) {
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
            totals = (float *) calloc(10, sizeof(float));

            // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, 28, 28, 8, 3, last_pool_input,
            //     last_soft_input, totals, soft_weight_init, soft_bias_init);

            forward2(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], out[n], &loss[n], &accuracy[n],
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

void run_sFedAvg(int **images, int *labels, int num_images, int num_train, float learning_rate, int per_print,
    int num_epochs, int num_filters, int filter_size, float *filters_init, float *soft_weight_init, 
        float *soft_bias_init, int num_nodes, int batch_size) {

    /************************************************ initialize layers ************************************************/
    Conv_layer *conv = (Conv_layer *) calloc(num_nodes, sizeof(Conv_layer));
    //Maxpool_layer *maxpool = (Maxpool_layer *) calloc(num_nodes, sizeof(Maxpool_layer));
    Avgpool_layer *avgpool = (Avgpool_layer *) calloc(num_nodes, sizeof(Avgpool_layer));
    Softmax_layer *softmax = (Softmax_layer *) calloc(num_nodes, sizeof(Softmax_layer));

    for (int i = 0; i < num_nodes; i++) {
        conv[i] = Conv_layer(28, 28, num_filters, filter_size, learning_rate);
        //maxpool[i] = Maxpool_layer(26, 26, num_filters);
        avgpool[i] = Avgpool_layer(26, 26, num_filters);
        softmax[i] = Softmax_layer(13*13*num_filters, 10, learning_rate);
    }

    printf("** Layers initialized **\n");

    // set initial weights for each node
    float **filters = new float*[num_nodes];
    float **soft_weights = new float*[num_nodes];
    float **soft_biases = new float*[num_nodes];

    for (int n = 0; n < num_nodes; n++) {
        filters[n] = new float[num_filters * filter_size * filter_size];
        soft_weights[n] = new float[13*13*num_filters*10];
        soft_biases[n] = new float[10];

        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            filters[n][i] = filters_init[i];
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            soft_weights[n][i] = soft_weight_init[i];
        }
        for (int i = 0; i < 10; i++) {
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
        out[n] = (float *) calloc(10, sizeof(float));
        soft_out[n] = (float *) calloc(1352, sizeof(float));
        last_pool_input[n] = (float *) calloc(26*26*8, sizeof(float));
        last_soft_input[n] = (float *) calloc(13*13*8, sizeof(float));
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
                    saverage_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes);
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
        average_weights(filters, soft_weights, soft_biases, num_filters, filter_size, num_nodes);
        
    }

    // set weights after training
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        filters_init[i] = filters[0][i];
    }
    for (int i = 0; i < 13*13*num_filters*10; i++) {
        soft_weight_init[i] = soft_weights[0][i];
    }
    for (int i = 0; i < 10; i++) {
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
            totals = (float *) calloc(10, sizeof(float));

            // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, 28, 28, 8, 3, last_pool_input,
            //     last_soft_input, totals, soft_weight_init, soft_bias_init);

            forward2(conv[n], avgpool[n], softmax[n], images[i], filters[n], labels[i], out[n], &loss[n], &accuracy[n],
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

void average_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, int num_nodes) {
    // average holders
    float *avg_filters = new float[num_filters * filter_size * filter_size]();
    float *avg_soft_weights = new float[13*13*num_filters*10]();
    float *avg_soft_biases = new float[10]();

    // average each node's weights and biases
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_filters[i] += filters[n][i];  
        }
        avg_filters[i] /= num_nodes;  
    }

    for (int i = 0; i < 13*13*num_filters*10; i++) {
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
        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            filters[n][i] = avg_filters[i];
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            soft_weights[n][i] = avg_soft_weights[i];
        }
        for (int i = 0; i < 10; i++) {
            soft_biases[n][i] = avg_soft_biases[i];
        }
    }

    delete[] avg_filters;
    delete[] avg_soft_weights;
    delete[] avg_soft_biases;

    return;
}

void saverage_weights(float **filters, float **soft_weights, float **soft_biases, int num_filters, int filter_size, int num_nodes) {
    // average holders
    sfloat *avg_filters = new sfloat[num_filters * filter_size * filter_size]();
    sfloat *avg_soft_weights = new sfloat[13*13*num_filters*10]();
    sfloat *avg_soft_biases = new sfloat[10]();

    // allocate secure values
    sfloat **sfilters = new sfloat*[num_nodes];
    sfloat **ssoft_weights = new sfloat*[num_nodes];
    sfloat **ssoft_biases = new sfloat*[num_nodes];
    
    for (int n = 0; n < num_nodes; n++) {
        sfilters[n] = new sfloat[num_filters * filter_size * filter_size]();
        ssoft_weights[n] = new sfloat[13*13*num_filters*10]();
        ssoft_biases[n] = new sfloat[10]();

        // set secure values
        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            sfilters[n][i].convert_in_place(filters[n][i]);  
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            ssoft_weights[n][i].convert_in_place(soft_weights[n][i]);  
        }
        for (int i = 0; i < 10; i++) {
            ssoft_biases[n][i].convert_in_place(soft_biases[n][i]);
        }
    }   

    // initialize average values
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        avg_filters[i].convert_in_place(0.0);  
    }
    for (int i = 0; i < 13*13*num_filters*10; i++) {
        avg_soft_weights[i].convert_in_place(0.0);  
    }
    for (int i = 0; i < 10; i++) {
        avg_soft_biases[i].convert_in_place(0.0);
    }


    // average each node's weights and biases
    for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_filters[i] += sfilters[n][i];  
        }
        avg_filters[i] /= num_nodes;  
    }

    for (int i = 0; i < 13*13*num_filters*10; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_weights[i] += ssoft_weights[n][i];
        }
        avg_soft_weights[i] /= num_nodes;
    }

    for (int i = 0; i < 10; i++) {
        for (int n = 0; n < num_nodes; n++) {
            avg_soft_biases[i] += ssoft_biases[n][i];
        }
        avg_soft_biases[i] /= num_nodes;
    }        

    // update each node's weights and biases
    for (int n = 0; n < num_nodes; n++) {
        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            sfilters[n][i] = avg_filters[i];
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            ssoft_weights[n][i] = avg_soft_weights[i];
        }
        for (int i = 0; i < 10; i++) {
            ssoft_biases[n][i] = avg_soft_biases[i];
        }
    }

    // return non-secure values
    for (int n = 0; n < num_nodes; n++) {
        for (int i = 0; i < num_filters * filter_size * filter_size; i++) {
            filters[n][i] = sfilters[n][i].reconstruct();  
        }
        for (int i = 0; i < 13*13*num_filters*10; i++) {
            soft_weights[n][i] = ssoft_weights[n][i].reconstruct();  
        }
        for (int i = 0; i < 10; i++) {
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