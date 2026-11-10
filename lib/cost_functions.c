float MSE(float node, float target) {
    return (node - target)*(node - target);
}

float MSE_derivative(float node, float target) {
    return 2*(node - target);
}
