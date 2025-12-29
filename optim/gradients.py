"""
    Derivative of J with respect to b: 
    - der(J)/der(b) = 1/m * sum(y^ - y)

    Derivative of J with respect to w: 
    - der(J)/der(w) = 1/m * sum((y^ - y)*x)

"""

def compute_gradients(x, y, y_hat):
    """
    x : features
    y : targets
    y_hat : predictions
    """
    m = len(y)
    grad_b = 0
    grad_w = 0

    for i in range(m):
        grad_b += y_hat[i] - y[i]
        grad_w += (y_hat[i] - y[i]) * x[i]
    
    return grad_b / m, grad_w / m 