# C Backpropagation
A multilayer perceptron, matrix algebra and forward/back propagation implementation written in c. This repo was written just to boost understanding in back propagation, there is a huge room for optimization, and other features.

# Requirements
None. Just the clang compiler.

# Building
```bash
cmhod +x build.sh
./build.sh
```

# Running
If everything builded correctly, after running placeholder program by ```./main``` you'll get output similar to this
```
Loaded 60000 MNIST training samples.
Input matrix:  784 x 60000
Target matrix: 10 x 60000
Starting training: 10 epochs, batch size 64, learning rate 0.01.
epoch 9/10, sample 59968/60000
Training finished.

===== MNIST TEST RESULTS =====
Correct predictions: 8586 / 10000
Accuracy: 85.86%
```

Enjoy!
