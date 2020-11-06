#include "layers.hpp"
#include "networks.hpp"

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
        float *soft_bias_init, int num_nodes) {
        /************************************************ initialize layers ************************************************/


    Conv_layer *conv = (Conv_layer *) calloc(num_nodes, sizeof(Conv_layer));
    Maxpool_layer *maxpool = (Maxpool_layer *) calloc(num_nodes, sizeof(Maxpool_layer));
    //Avgpool_layer *avgpool = new Avgpool_layer[num_nodes];
    Softmax_layer *softmax = (Softmax_layer *) calloc(num_nodes, sizeof(Softmax_layer));

    for (int i = 0; i < num_nodes; i++) {
        conv[i] = Conv_layer(28, 28, num_filters, filter_size, learning_rate);
        maxpool[i] = Maxpool_layer(26, 26, num_filters);
        //avgpool[i] = Avgpool_layer(26, 26, num_filters);
        softmax[i] = Softmax_layer(13*13*num_filters, 10, learning_rate);
    }

    printf("** Layers initialized **\n");

    /************************************************ training ************************************************/

    // accuracy variables
    int i, j;
    float count = 0.0;
    float *loss = new float[num_nodes]();
    float *accuracy= new float[num_nodes]();
    float *num_correct = new float[num_nodes]();
    float *total_loss = new float[num_nodes]();

    // declare variables that can be reused
    float *out, *soft_out, *last_pool_input, *last_soft_input;
    out = (float *) calloc(10, sizeof(float));
    soft_out = (float *) calloc(1352, sizeof(float));
    last_pool_input = (float *) calloc(26*26*8, sizeof(float));
    last_soft_input = (float *) calloc(13*13*8, sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("--- Epoch %d ---\n", epoch+1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < num_nodes; n++) {
            for (i = n*(num_train/num_nodes); i < num_train; i++) {
                if (i > 0 && i % per_print == (per_print-1)) {
                    printf("step %d: past %d steps avg loss: %.3f | accuracy: %.3f\n", i+1, per_print,
                        total_loss[n]/per_print, num_correct[n]/per_print);
                    total_loss[n] = 0.0;
                    num_correct[n] = 0.0;
                }
                //printf("Label: %d\n", labels[i]);

                // train(conv, maxpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                //         soft_weight_init, soft_bias_init, out, soft_out,
                //         last_pool_input, last_soft_input);

                // train2(conv, avgpool, softmax, images[i], filters_init, labels[i], loss_p, accuracy_p,
                //         soft_weight_init, soft_bias_init, out, soft_out,
                //         last_pool_input, last_soft_input);

                train(conv[n], maxpool[n], softmax[n], images[i], filters_init, labels[i], &loss[n], &accuracy[n],
                        soft_weight_init, soft_bias_init, out, soft_out,
                        last_pool_input, last_soft_input);

                total_loss[n] += loss[n];
                num_correct[n] += accuracy[n];

            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end-epoch_start).count()/1000000000.0;
        printf("Epoch completed in: %.6lf seconds\n", epoch_duration);
        for (int n = 0; n < num_nodes; n++) {
            printf("Node %d\n", n);
            printf("\tConv_forward: %.6lf sec, Maxpool_forward: %.6lf sec, Softmax_forward: %.6lf sec\n",
            conv[n].get_duration(true), maxpool[n].get_duration(true), softmax[n].get_duration(true));
            printf("\tConv_backward: %.6lf sec, Maxpool_backward: %.6lf sec, Softmax_backward: %.6lf sec\n\n",
            conv[n].get_duration(false), maxpool[n].get_duration(false), softmax[n].get_duration(false));

            conv[n].update_duration(0.0, true);
            conv[n].update_duration(0.0, false);
            maxpool[n].update_duration(0.0, true);
            maxpool[n].update_duration(0.0, false);
            softmax[n].update_duration(0.0, true);
            softmax[n].update_duration(0.0, false);
        }
        // reset times
        epoch_duration = 0.0;
        
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

    for (int n = 0; n < num_nodes; n++) {
        for (i = num_train; i < num_images; i++) {
            // re-initialize totals for each image
            float *totals;
            totals = (float *) calloc(10, sizeof(float));

            // forward(images[i], filters_init, labels[i], out, loss_p, accuracy_p, 28, 28, 8, 3, last_pool_input,
            //     last_soft_input, totals, soft_weight_init, soft_bias_init);

            forward(conv[n], maxpool[n], softmax[n], images[i], filters_init, labels[i], out, &loss[n], &accuracy[n],
                last_pool_input, last_soft_input, totals, soft_weight_init, soft_bias_init);

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

    for (int n = 0; n < num_nodes; n++) {
        auto test_end = std::chrono::high_resolution_clock::now();
        double test_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-test_start).count()/1000000000.0;
        printf("Testing for node %d completed in: %.6lf seconds\n", n, test_duration);
    }

    delete[] loss;
    delete[] accuracy;
    delete[] total_loss;
    delete[] num_correct;
    return;

    }