#include "cost_functions.h"
#include "matrix.h"

#include <math.h>

static float MSE(float *output_layer, float *target, size_t layer_length) {
    float res = 0;

    for (int i = 0; i < layer_length; i++) {
        res += (output_layer[i] - target[i]) * (output_layer[i] - target[i]);
    }

    return res;
}

static matrix MSE_gradient(float *output_layer, float *target, size_t layer_length) {
    matrix gradient = {};
    matrix_init(&gradient, layer_length, 1);

    for (int i = 0; i < layer_length; i++) {
        gradient.content[i] = 2 * (output_layer[i] - target[i]);
    }

    return gradient;
}

static float cross_entropy(float *output_layer, float *target, size_t layer_length) {
    float res = 0;

    for (int i = 0; i < layer_length; i++) {
        res -= target[i] * logf(output_layer[i]);
    }

    return res;
}

static matrix cross_entropy_gradient(float *output_layer, float *target, size_t layer_length) {
    matrix gradient = {};
    matrix_init(&gradient, layer_length, 1);

    for (int i = 0; i < layer_length; i++) {
        gradient.content[i] = -target[i] / output_layer[i];
    }

    return gradient;
}

cost_function construct_MSE() {
    return (cost_function){
        .function = &MSE,
        .gradient = &MSE_gradient,
    };
}

cost_function construct_cross_entropy() {
    return (cost_function){
        .function = &cross_entropy,
        .gradient = &cross_entropy_gradient,
    };
}
