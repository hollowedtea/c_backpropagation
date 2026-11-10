#include <stdint.h>
#include <stdlib.h>
#include <math.h>

float ReLU(float *preprocess_layer, size_t layer_length, size_t current_neuron) {
    if (preprocess_layer[current_neuron] < 0) {
        return .0;
    }

    return preprocess_layer[current_neuron];
}

float ReLU_derivative(float *preprocess_layer, size_t layer_length, size_t current_neuron) {
    if (preprocess_layer[current_neuron] <= 0) {
        return .0;
    }

    return 1.0;
}

float softmax(float *preprocess_layer, size_t layer_length, size_t current_neuron) {
    float result = exp(preprocess_layer[current_neuron]);
    float denominator = .0;

    for (int i = 0; i < layer_length; i++) {
        denominator += exp(preprocess_layer[i]);
    }

    return result/denominator;
}
