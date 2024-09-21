# handwritingAI
Implementing an AI to read the common handwriting dataset.
It will have two layers each with 10 nodes.
I am using the MNIST dataset.

1. each letter is 28x28 resolution handwritten digits
2. each pixel will be denoted 0 for black and 255 for white
3. each letter will be added to a 784xM (28^2xM) matrix with each row being one example, then the matrix is Transposed
4. the input layer is simply A = x 
5. the first unactivated hidden layer will be a dot product of a weight array and the input layer plus a bias
6. the ReLu function is then applied to the same layer to activate it 

the ReLu function, or multiplying by the heaviside function, ensures the neural network is not linear

7. The unactivated second layer (output layer) will again be a dot product of a weight array and the input layer plus a bias
8. The softmax function will then be applied to the second layer to convert weights to percentages
9. The output will then be tested for correctness, and the weights and biases will be adjusted adequately (backwards propagation)

x = [x1, x2... xm]^T

A[0] = x

Z[1] = W[1]A[0] + b[1]
A[1] = Relu(Z[1])

Z[2] = W[2]A[1] + b[2]
A[2] = softmax(Z[2])

Y = maxIndex(A[2])

BackProp
ex, y = 4

Y = [0,0,0,1,0...]^T

dZ[2] = A[2] - Y
dW[2] = (1/m)(dZ[2]A[1])
db[2] = (1/m)Summation(dZ[2])

dZ[1] = W[2]^T dZ[2] * ReLu'(Z[1])
dW[1] = (1/m)(dZ[1]X^T)
db[1] = (1/m)Summation(dZ[1])

a = learning rate (hyper variable)

W[1] := W[1] - a*dW[1]
b[1] := b[1] - a*db[1]
W[2] := W[2] - a*dW[2]
b[2] := b[2] - a*db[2]

repeat!