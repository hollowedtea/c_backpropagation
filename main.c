#include "lib/matrix.h"
#include "lib/activation_functions.h"
#include "lib/cost_functions.h"
#include "lib/perceptron.h"
#include "lib/idx_parser.c"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define MNIST_IMAGE_SIZE 784u
#define MNIST_CLASS_COUNT 10u

#define DEFAULT_IMAGES_PATH "mnist/train-images-idx3-ubyte"
#define DEFAULT_LABELS_PATH "mnist/train-labels-idx1-ubyte"

#define TRAIN_BATCH_SIZE 64u
#define TRAIN_EPOCHS 10u
#define TRAIN_LEARNING_RATE 0.01f

static void destroy_perceptron(perceptron *p) {
    if (p == NULL) {
        return;
    }

    if (p->matrices != NULL) {
        for (size_t i = 0; i + 1 < p->layers; ++i) {
            free(p->matrices[i].content);
        }
    }

    if (p->biases != NULL) {
        for (size_t i = 0; i + 1 < p->layers; ++i) {
            free(p->biases[i].content);
        }
    }

    free(p->matrices);
    free(p->biases);
    free(p->activation_functions);

    *p = (perceptron){0};
}

static size_t argmax(const float *values, size_t n) {
    size_t max_index = 0;

    for (size_t i = 1; i < n; i++) {
        if (values[i] > values[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

int main(int argc, char **argv) {
    const char *images_path = DEFAULT_IMAGES_PATH;
    const char *labels_path = DEFAULT_LABELS_PATH;

    if (argc == 3) {
        images_path = argv[1];
        labels_path = argv[2];
    } else if (argc != 1) {
        fprintf(stderr,
                "Usage: %s [train-images-idx3-ubyte train-labels-idx1-ubyte]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    matrix *image_samples = NULL;
    matrix *label_samples = NULL;
    size_t sample_count = 0;

    if (!load_mnist_train(
            images_path,
            labels_path,
            &image_samples,
            &label_samples,
            &sample_count)) {
        fprintf(stderr, "Could not load MNIST training data.\n");
        return EXIT_FAILURE;
    }

    if (sample_count == 0 || sample_count > (size_t)-1 / MNIST_IMAGE_SIZE) {
        fprintf(stderr, "Invalid MNIST sample count.\n");
        free_matrix_array(image_samples, sample_count);
        free_matrix_array(label_samples, sample_count);
        return EXIT_FAILURE;
    }

    // Store samples as columns, matching the network's column-vector convention:
    // input:  784 x sample_count
    // target:  10 x sample_count
    matrix train_data_input = {0};
    matrix train_data_target = {0};

    matrix_init(&train_data_input, MNIST_IMAGE_SIZE, sample_count);
    matrix_init(&train_data_target, MNIST_CLASS_COUNT, sample_count);

    if (train_data_input.content == NULL || train_data_target.content == NULL) {
        fprintf(stderr, "Could not allocate packed MNIST matrices.\n");

        free(train_data_input.content);
        free(train_data_target.content);
        free_matrix_array(image_samples, sample_count);
        free_matrix_array(label_samples, sample_count);

        return EXIT_FAILURE;
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        if (image_samples[sample].rows != MNIST_IMAGE_SIZE ||
            image_samples[sample].columns != 1 ||
            label_samples[sample].rows != MNIST_CLASS_COUNT ||
            label_samples[sample].columns != 1) {
            fprintf(stderr, "Unexpected dimensions in MNIST sample %zu.\n", sample);

            free(train_data_input.content);
            free(train_data_target.content);
            free_matrix_array(image_samples, sample_count);
            free_matrix_array(label_samples, sample_count);

            return EXIT_FAILURE;
        }

        for (size_t pixel = 0; pixel < MNIST_IMAGE_SIZE; ++pixel) {
            train_data_input.content[pixel * sample_count + sample] =
                image_samples[sample].content[pixel];
        }

        for (size_t digit = 0; digit < MNIST_CLASS_COUNT; ++digit) {
            train_data_target.content[digit * sample_count + sample] =
                label_samples[sample].content[digit];
        }
    }

    // The packed matrices now own all training data; release per-sample arrays.
    free_matrix_array(image_samples, sample_count);
    free_matrix_array(label_samples, sample_count);

    printf("Loaded %zu MNIST training samples.\n", sample_count);
    printf("Input matrix:  %zu x %zu\n",
           train_data_input.rows, train_data_input.columns);
    printf("Target matrix: %zu x %zu\n",
           train_data_target.rows, train_data_target.columns);

    // Constructing the perceptron
    // 784 inputs -> 16 -> 16 -> 10 output classes.
    size_t layer_sizes[] = {784, 16, 16, 10};
    activation_function funcs[3];

    funcs[0] = construct_ReLU();
    funcs[1] = construct_ReLU();
    funcs[2] = construct_softmax();

    cost_function cost_func = construct_cross_entropy();

    float range[] = {-0.05f, 0.05f};

    perceptron p = perceptron_init(
        4,
        layer_sizes,
        funcs,
        cost_func,
        range
    );

    if (p.matrices == NULL || p.biases == NULL ||
        p.activation_functions == NULL) {
        fprintf(stderr, "Could not initialize perceptron.\n");

        destroy_perceptron(&p);
        free(train_data_input.content);
        free(train_data_target.content);

        return EXIT_FAILURE;
    }

    size_t batch_size = TRAIN_BATCH_SIZE;
    if (batch_size > sample_count) {
        batch_size = sample_count;
    }

        printf("Starting training: %zu epochs, batch size %zu, learning rate %.2f.\n",
           (size_t)TRAIN_EPOCHS, batch_size, TRAIN_LEARNING_RATE);

    back_propagation(
        &p,
        train_data_input,
        train_data_target,
        batch_size,
        TRAIN_LEARNING_RATE,
        TRAIN_EPOCHS
    );

    printf("Training finished.\n");

    // Load MNIST test dataset
    matrix *test_images = NULL;
    matrix *test_labels = NULL;
    size_t test_count = 0;

    if (!load_mnist_train(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            &test_images,
            &test_labels,
            &test_count)) {

        fprintf(stderr, "Could not load MNIST test dataset.\n");

        destroy_perceptron(&p);
        free(train_data_input.content);
        free(train_data_target.content);

        return EXIT_FAILURE;
    }

    // Evaluate accuracy using the existing forward_propagation() function.
    size_t correct = 0;

    for (size_t i = 0; i < test_count; i++) {
        matrix prediction = forward_propagation(&p, test_images[i]);

        if (prediction.content == NULL ||
            prediction.rows * prediction.columns != MNIST_CLASS_COUNT) {
            fprintf(stderr, "Prediction failed at sample %zu.\n", i);

            free(prediction.content);
            free_matrix_array(test_images, test_count);
            free_matrix_array(test_labels, test_count);

            destroy_perceptron(&p);
            free(train_data_input.content);
            free(train_data_target.content);

            return EXIT_FAILURE;
        }

        size_t predicted = argmax(
            prediction.content,
            prediction.rows * prediction.columns
        );
        size_t actual = argmax(test_labels[i].content, MNIST_CLASS_COUNT);

        if (predicted == actual) {
            correct++;
        }

        free(prediction.content);
    }

    float accuracy = 100.0f * (float)correct / (float)test_count;

    printf("\n===== MNIST TEST RESULTS =====\n");
    printf("Correct predictions: %zu / %zu\n", correct, test_count);
    printf("Accuracy: %.2f%%\n", accuracy);

    // Release test data
    free_matrix_array(test_images, test_count);
    free_matrix_array(test_labels, test_count);

    destroy_perceptron(&p);
    free(train_data_input.content);
    free(train_data_target.content);

    return EXIT_SUCCESS;
}
