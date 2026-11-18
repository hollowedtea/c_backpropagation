#include <stdio.h>
#include "matrix.h"

#ifndef _COST_FUNCTIONS
#define _COST_FUNCTIONS

typedef struct {
    float (*function)(float *output_layer, float *target, size_t layer_length);
    matrix (*gradient)(float *output_layer, float *target, size_t layer_length);
} cost_function;

cost_function construct_MSE();
cost_function construct_cross_entropy();

#endif
