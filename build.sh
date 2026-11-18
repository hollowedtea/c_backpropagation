clang -c lib/matrix.c -o matrix.lib
clang -c lib/perceptron.c -o perceptron.lib
clang -c lib/activation_functions.c -o activation_functions.lib
clang -c lib/cost_functions.c -o cost_functions.lib
clang -lm main.c cost_functions.lib activation_functions.lib matrix.lib perceptron.lib -o main
