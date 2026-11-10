clang -c lib/matrix.c -o matrix.lib
clang -c lib/perceptron.c -o perceptron.lib
clang -lm main.c matrix.lib perceptron.lib -o main
