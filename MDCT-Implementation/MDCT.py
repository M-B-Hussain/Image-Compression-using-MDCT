import numpy as np

def mdct(x, window):
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += x[n] * window[n] * np.cos((2*np.pi*(n+0.5)*k)/N)
        X[k] = sum
    return X