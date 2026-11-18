#include "lib/matrix.h"
#include "lib/cost_functions.h"
#include "lib/perceptron.h"
//#include "lib/activation_functions.c"
#include "lib/activation_functions.h"
#include <stdio.h>

int main(void) {
    size_t layer_sizes[] = {2, 3, 2};
    activation_function funcs[2];

    funcs[0] = construct_ReLU();
    funcs[1] = construct_softmax();

    float range[] = {-10., 10.};

    cost_function cost_func = construct_cross_entropy();

    perceptron p = perceptron_init(3, layer_sizes, funcs, cost_func, range);

    matrix input = {};
    matrix_init(&input, 2, 1);

    for (int i = 0; i < 2; i++) {
        input.content[i] = 1;
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
