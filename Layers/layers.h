class Conv_layer {
    private:
        int rows, cols, num_filters, filter_size;
        float learn_rate;

    public:
        Conv_layer(int in_rows, int in_cols, int in_num_filters, int in_filter_size, float in_learn_rate);
        Conv_layer();
        void forward(float *dest, float *image, float *filters);
        void back(float *dest, float *gradient, float *last_input);
};

class Maxpool_layer {
    private:
        int rows, cols, num_filters;

    public:
        Maxpool_layer(int in_rows, int in_cols, int in_num_filters);
        Maxpool_layer();
        void forward(float *dest, float *input);
        void back(float *dest, float *gradient, float *last_input);
};

class Softmax_layer {
    private:
        int in_length, out_length;
        float learn_rate;
    
    public:
        Softmax_layer(int in_in_length, int in_out_length, float in_learn_rate);
        Softmax_layer();
        void forward(float *dest, float *input, float *totals, float *weight, float *bias);
        void back(float *dest, float *gradient, float *last_input, float *totals, float *weight, float *bias);
        
};