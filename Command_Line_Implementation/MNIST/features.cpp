#include <cstdio>

int main() {
    int filter_size = 5;
    int num_filters = 8;
    int colors = 1;
    int filters_per_row = 4;
    int filters_per_col = num_filters / filters_per_row;
    int rows = (filters_per_col * (filter_size + 1)) + 1;
    int cols = (filters_per_row * (filter_size + 1)) + 1;

    FILE *fpt, *fout;
    fpt = fopen("output_5x5.txt", "rb");
    fout = fopen("MNIST_feature_images_5x5.ppm", "wb");

    fprintf(fout, "P5 %d %d %d\n", cols, rows, 255);

    float *input = new float[num_filters * filter_size * filter_size * colors];
    unsigned char *data = new unsigned char[rows * cols * colors]();

    for (int i = 0; i < num_filters * filter_size * filter_size * colors; i++) {
        fscanf(fpt, "%f, ", &input[i]);
    }
    
    float *max = new float[num_filters]();
    float *min = new float[num_filters]();

    // find max and min values
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size * filter_size * colors; j++) {
            if (input[i *  filter_size * filter_size * colors + j] > max[i]) max[i] = input[i *  filter_size * filter_size * colors + j];
            if (input[i *  filter_size * filter_size * colors + j] < min[i]) min[i] = input[i *  filter_size * filter_size * colors + j];
        }
    }

    int i = 0;
    int data_index = 0;
    int filter_index = 0;

    for (int pr = 0; pr < filters_per_col; pr++) {
        for (int pc = 0; pc < filters_per_row; pc++) {
            // filter i
            i = pr * filters_per_row + pc;
            int start_r = pr * colors * (filter_size + 1) + 1;
            int start_c = pc * colors * (filter_size + 1) + 1;
            for (int r = 0; r < filter_size; r++) {
                for (int c = 0; c < filter_size; c++) {
                    for (int k = 0; k < colors; k++) {
                        data_index = (start_r + r) * cols + (start_c + c);
                        filter_index = (i * filter_size * filter_size * colors) + r * filter_size + c;
                        data[data_index] = (unsigned char) (max[i] * input[filter_index] / (max[i] + min[i]) * 255);
                    }
                }
            }
        }
    }

    printf("cols = %d, rows = %d\n", cols, rows);

    fwrite(data, rows * cols, 1, fout);
}