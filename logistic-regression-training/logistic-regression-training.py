import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression using gradient descent.
    Returns (w, b)
    """
    
    # number of samples and features
    n_samples, n_features = X.shape
    
    # initialize parameters
    w = np.zeros(n_features)
    b = 0
    
    for i in range(steps):

        # Linear model
        z = X@w + b
        # sigmoid probabilities
        sigma = _sigmoid(z)
        #error
        error = sigma - y
        # calculate Gradients
        dw = (1/n_samples)*(X.T @ (error))
        db = np.mean(error)
        # final eqn
        w = w - lr*dw
        b = b - lr*db
        
    return w, b