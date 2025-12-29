from core.neuron import neuron
from loss.mse import mse
from optim.gradients import compute_gradients

# Initialize parameters
b = 0
w = 0

# a fake dataset 
# function: y = 2x + 1

x = [0, 1, 2, 3, 4]
y = [1, 3, 5, 7, 9]

# hyperparameters
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    
    # Forward pass
    y_hat = []
    for i in range(len(x)):
        y_hat.append(neuron(x[i], b, w))

    # Calculate the loss
    loss = mse(y, y_hat)

    # Backpropagation
    grad_b, grad_w =compute_gradients(x, y, y_hat)

    # Update
    b = b- (learning_rate * grad_b)
    w = w- (learning_rate * grad_w)

    # Print the loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

    if loss < 0.0001:
        break    

print(f"Final parameters: b = {b}, w = {w}")

# Plot the results
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, y_hat, color='red')
plt.show()

