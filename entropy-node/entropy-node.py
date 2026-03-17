import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    # mp = {}
    # total = 0.0
    # p = []

    # for i in y:
    #     if i in mp:
    #         mp[i] += 1
    #     else:
    #         mp[i] = 1
    #     total += 1

    # ans = 0.0
    # for j in mp.values():
    #     ans+=(j/total)*np.log2(j/total)
    val,counts = np.unique(y,return_counts=True)
    probs = counts/counts.sum()
    return -np.sum(probs*np.log2(probs))