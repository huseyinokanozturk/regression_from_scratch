'''
Basic loss function: Mean Squared Error (MSE)

formula: 
    - J(b,w) = 1/2m * sum((y^ - y)^2)

'''


def mse(y,y_hat):
    """
    y is a list of target values
    y_hat is a list of predicted values
    m is the number of samples
    """
    m = len(y)
    sum_loss = 0

    for i in range(m):
        sum_loss += (y_hat[i] - y[i])**2

    return sum_loss / (2*m)   