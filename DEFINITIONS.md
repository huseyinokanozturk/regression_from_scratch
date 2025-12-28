I am writing this file to define the terms I will use in my regression from scratch project.    

Mathematical terms:
- x : feature
- y : target
- y^ : predicted target
- b : constant
- w : weight 
- alpha : learning rate
- J(b,w) : loss function (mean squared error)

The formula for the linear regression is: y^ = b + w * x 


My goal: Solve a one variable linear regression problem using:
- A neural network from scratch
- gradient descent 
- only using basic math libraries like exp, sqrt etc. 
- no numpy or pandas or any other library

Loss function: 
- I will use the mean squared error 
- mse formula:
    - J(b,w) = 1/2m * sum((y^ - y)^2)

Gradient Descent: I am using derivatives to make the algorithm understand if it should go up or down.

    Derivative of J with respect to b: 
    - der(J)/der(b) = 1/m * sum(y^ - y)

    Derivative of J with respect to w: 
    - der(J)/der(w) = 1/m * sum((y^ - y)*x)

Update rules:
    - w = w - (alpha * der(J)/der(w))
    - b = b - (alpha * der(J)/der(b))
    - We are using alpha here to control the results of the derivatives (derivative might be too big or too small) (example: alpha = 0.01)

For normalization:
    - x_norm = (x - x_min) / (x_max - x_min) 


TRAINING LOOP:

- Initialize b and w to random values (example: b = 0, w = 0) 
- For each iteration:
    - Calculate y^ using the formula y^ = b + w * x
    - Calculate the loss using the formula J(b,w) = 1/2m * sum((y^ - y)^2)
    - Calculate the gradient of the loss with respect to b and w
    - Update b and w using the update rules
- Repeat until the loss is minimized 
