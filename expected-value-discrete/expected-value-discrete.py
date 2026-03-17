import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    
    ex = 0.0
    sum = 0.0
    for i in range(len(x)):
        ex+=(x[i]*p[i])
        sum+=p[i]

    if sum != 1.0:
        raise ValueError('Not possible')
    
    # Write code here
    return ex
