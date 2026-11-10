#include "lib/matrix.h"
#include "lib/perceptron.h"
#include "lib/activation_functions.c"
#include <stdio.h>

int main(void) {
    size_t layer_sizes[] = {2, 3, 2};
    float (*funcs[2])(float *, size_t, size_t);

    funcs[0] = &ReLU;
    funcs[1] = &softmax;

    perceptron p = perceptron_init(3, layer_sizes, funcs);

    matrix input = {};
    matrix_init(&input, 1, 2);

    for (int i = 0; i < 2; i++) {
        input.content[i] = 1;
    }

    for (int i = 0; i < p.layers - 1; i++) {
        for (int j = 0; j < p.matrices[i].rows * p.matrices[i].columns; j++) {
            p.matrices[i].content[j] = (float)3/(i + j + 1);
        }
    }

    matrix output = forward_propagation(&p, input);

    for (int i = 0; i < output.rows * output.columns; i++) {
        printf("%.2f \n", output.content[i]);
    }
    
    free(p.matrices);
    free(p.biases);
    free(p.activation_functions);
    free(input.content);
    free(output.content);
}
