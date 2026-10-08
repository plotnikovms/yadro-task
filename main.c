#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>

enum {
    EXIT_INVALID_ARGUMENTS = 10,
    EXIT_FILE_ERROR = 11,
    EXIT_DATA_ERROR = 12
};

typedef struct {
    uint32_t height;
    uint32_t width;
    uint8_t *a;
    uint8_t *b;
    uint8_t *c;

    uint16_t d_height;
    uint16_t d_width;
    int8_t *d;
} InputData;

typedef struct {
    uint8_t *a;
    uint8_t *b;
    uint8_t *c;
} OutputData;

static void free_input_data(InputData *data) {
    free(data->a);
    free(data->b);
    free(data->c);
    free(data->d);
    data->a = NULL;
    data->b = NULL;
    data->c = NULL;
    data->d = NULL;
}

static void free_output_data(OutputData *data) {
    free(data->a);
    free(data->b);
    free(data->c);
    data->a = NULL;
    data->b = NULL;
    data->c = NULL;
}

static int read_exact_bytes(FILE *file, void *buffer, size_t size) {
    return size == 0 || fread(buffer, 1, size, file) == size;
}

static int write_exact_bytes(FILE *file, const void *buffer, size_t size) {
    return size == 0 || fwrite(buffer, 1, size, file) == size;
}

static int read_input_data(FILE *file, InputData *data) {
    size_t element_count;
    size_t d_element_count;

    if (!read_exact_bytes(file, &data->height, sizeof(data->height)) ||
        !read_exact_bytes(file, &data->width, sizeof(data->width))) {
        return 1;
    }

    if (data->width != 0 &&
        (size_t)data->height > SIZE_MAX / data->width) {
        return 1;
    }
    element_count = (size_t)data->height * data->width;

    if (element_count != 0) {
        data->a = malloc(element_count * sizeof(*data->a));
        data->b = malloc(element_count * sizeof(*data->b));
        data->c = malloc(element_count * sizeof(*data->c));
        if (data->a == NULL || data->b == NULL || data->c == NULL) {
            return 1;
        }

        uint8_t values[3];
        for (size_t i = 0; i < element_count; ++i) {
            if (!read_exact_bytes(file, values, 3)) {
                return 1;
            }
            data->a[i] = values[0];
            data->b[i] = values[1];
            data->c[i] = values[2];
        }
    }

    if (!read_exact_bytes(file, &data->d_height, sizeof(data->d_height)) ||
        !read_exact_bytes(file, &data->d_width, sizeof(data->d_width))) {
        return 1;
    }

    d_element_count = (size_t)data->d_height * data->d_width;
    if (d_element_count != 0) {
        data->d = malloc(d_element_count * sizeof(*data->d));
        if (data->d == NULL ||
            !read_exact_bytes(file, data->d,
                              d_element_count * sizeof(*data->d))) {
            return 1;
        }
    }

    return 0;
}

static uint8_t transform_convolution_result(int64_t value) {
    if (value > 255) {
        value %= 251;
    } else if (value < 0) {
        value = (-value) % 241;
    }

    return (uint8_t)value;
}

static int convolve_all_matrices(const InputData *input,
                                 OutputData *output) {
    size_t element_count = (size_t)input->height * input->width;

    if (element_count == 0) {
        return 0;
    }

    output->a = malloc(element_count * sizeof(*output->a));
    output->b = malloc(element_count * sizeof(*output->b));
    output->c = malloc(element_count * sizeof(*output->c));
    if (output->a == NULL || output->b == NULL || output->c == NULL) {
        return 1;
    }

    const size_t kernel_center_row = input->d_height / 2;
    const size_t kernel_center_column = input->d_width / 2;

    for (size_t row = 0; row < input->height; ++row) {
        size_t kernel_row_begin = row < kernel_center_row
                                  ? kernel_center_row - row : 0;
        size_t kernel_row_end = kernel_center_row + input->height - row;
        if (kernel_row_end > input->d_height) {
            kernel_row_end = input->d_height;
        }

        for (size_t column = 0; column < input->width; ++column) {
            size_t kernel_column_begin = column < kernel_center_column
                                         ? kernel_center_column - column : 0;
            size_t kernel_column_end = kernel_center_column
                                       + input->width - column;
            if (kernel_column_end > input->d_width) {
                kernel_column_end = input->d_width;
            }

            int64_t sum_a = 0;
            int64_t sum_b = 0;
            int64_t sum_c = 0;

            for (size_t kernel_row = kernel_row_begin;
                 kernel_row < kernel_row_end; ++kernel_row) {
                size_t source_row = row + kernel_row - kernel_center_row;
                size_t source_index = source_row * input->width
                                      + column + kernel_column_begin
                                      - kernel_center_column;
                size_t kernel_index = kernel_row * input->d_width
                                      + kernel_column_begin;

                for (size_t kernel_column = kernel_column_begin;
                     kernel_column < kernel_column_end; ++kernel_column) {
                    int64_t coefficient = input->d[kernel_index];
                    sum_a += (int64_t)input->a[source_index] * coefficient;
                    sum_b += (int64_t)input->b[source_index] * coefficient;
                    sum_c += (int64_t)input->c[source_index] * coefficient;
                    ++source_index;
                    ++kernel_index;
                }
            }

            size_t result_index = row * input->width + column;
            output->a[result_index] = transform_convolution_result(sum_a);
            output->b[result_index] = transform_convolution_result(sum_b);
            output->c[result_index] = transform_convolution_result(sum_c);
        }
    }

    return 0;
}

static int write_output_data(FILE *file, const InputData *input,
                             const OutputData *output) {
    if (!write_exact_bytes(file, &input->height, sizeof(input->height)) ||
        !write_exact_bytes(file, &input->width, sizeof(input->width))) {
        return 1;
    }

    size_t element_count = (size_t)input->height * input->width;
    for (size_t i = 0; i < element_count; ++i) {
        uint8_t values[3] = {output->a[i], output->b[i], output->c[i]};
        if (!write_exact_bytes(file, values, sizeof(values))) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char *argv[]) {
    const char* input_filename = NULL;
    const char* output_filename = NULL;
    int option;

    while ((option = getopt(argc, argv, "i:o:")) != -1) {
        switch (option) {
        case 'i':
            input_filename = optarg;
            break;
        case 'o':
            output_filename = optarg;
            break;
        default:
            return EXIT_INVALID_ARGUMENTS;
        }
    }

    if (input_filename == NULL || output_filename == NULL) {
        return EXIT_INVALID_ARGUMENTS;
    }

    if (optind != argc) {
        return EXIT_INVALID_ARGUMENTS;
    }

    FILE* input = fopen(input_filename, "rb");
    if (input == NULL) {
        perror(input_filename);
        return EXIT_FILE_ERROR;
    }

    InputData data = {0};
    if (read_input_data(input, &data) != 0) {
        fprintf(stderr, "Invalid or incomplete input file\n");
        free_input_data(&data);
        fclose(input);
        return EXIT_DATA_ERROR;
    }
    fclose(input);

    OutputData result = {0};
    if (convolve_all_matrices(&data, &result) != 0) {
        fprintf(stderr, "Failed to allocate memory for the result\n");
        free_output_data(&result);
        free_input_data(&data);
        return EXIT_DATA_ERROR;
    }

    FILE* output = fopen(output_filename, "wb");
    if (output == NULL) {
        perror(output_filename);
        free_output_data(&result);
        free_input_data(&data);
        return EXIT_FILE_ERROR;
    }

    int write_error = write_output_data(output, &data, &result);
    int close_error = (fclose(output) != 0);

    free_output_data(&result);
    free_input_data(&data);

    if (write_error || close_error) {
        fprintf(stderr, "Failed to write the output file\n");
        return EXIT_FILE_ERROR;
    }

    return 0;
}
