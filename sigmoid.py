import numpy as np
def sigmoid(x):
     """
    Compute the sigmoid of x, with x being a scalar or a numpy array
    """
     return 1/(1+np.exp(-x))

assert np.allclose(
    sigmoid(np.array([1, 2, 3])),
    [0.73105858, 0.88079708, 0.95257413]
)

assert np.allclose(
    sigmoid(np.array(2)),
    0.88079708
)

print("Both cases are passed")

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


assert np.allclose(
    sigmoid_derivative(np.array([1, 2, 3])),
    [0.19661193, 0.10499359, 0.04517666]
)

assert np.allclose(
    sigmoid_derivative(np.array(3)),
    0.04517666
)

print("Both cases are passed for sigmoid derivative")