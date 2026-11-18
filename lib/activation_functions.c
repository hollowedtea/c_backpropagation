#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "activation_functions.h"

static float ReLU(float *preprocess_layer, size_t layer_length, size_t current_node) {
    if (preprocess_layer[current_node] < 0) {
        return .0;
    }

    return preprocess_layer[current_node];
}

static matrix ReLU_jacobian(float *preprocess_layer, size_t layer_length) {
    matrix jacobian = {};
    matrix_init(&jacobian, layer_length, layer_length);

    memset(jacobian.content, 0, sizeof(float) * layer_length * layer_length);

    for (int i = 0; i < layer_length; i++) {
        jacobian.content[i * layer_length + i] = (preprocess_layer[i] <= 0) ? .0 : 1.0;
    }

    return jacobian;
}

static float softmax(float *preprocess_layer, size_t layer_length, size_t current_node) {
    // to avoid overflow
    float max_node = preprocess_layer[0];
    for (int i = 1; i < layer_length; i++) {
        if (preprocess_layer[i] > max_node) {
            max_node = preprocess_layer[i];
        }
    }

    // just regular softmax
    float result = expf(preprocess_layer[current_node] - max_node);
    float denominator = .0;

    for (int i = 0; i < layer_length; i++) {
        denominator += expf(preprocess_layer[i] - max_node);
    }

    return result/denominator;
}

static matrix softmax_jacobian(float *preprocess_layer, size_t layer_length) {
    matrix jacobian = {};
    matrix_init(&jacobian, layer_length, layer_length);

    float *s = (float *)malloc(sizeof(float) * layer_length);
    for (int i = 0; i < layer_length; i++) {
        s[i] = softmax(preprocess_layer, layer_length, i);
    }

    for (int i = 0; i < layer_length; i++) {
        for (int j = 0; j < layer_length; j++) {
            if (i == j) {
                jacobian.content[i * layer_length + j] = s[i] * (1 - s[i]);
                continue;
            }

            jacobian.content[i * layer_length + j] = -s[i]*s[j];
        }
    }


    free(s);
    return jacobian;
}

activation_function construct_ReLU() {
    return (activation_function){
        .function = &ReLU,
        .jacobian = &ReLU_jacobian
    };
};

activation_function construct_softmax() {
    return (activation_function){
        .function = &softmax,
        .jacobian = &softmax_jacobian,
    };
}
