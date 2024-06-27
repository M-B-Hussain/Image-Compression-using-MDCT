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

def window_hamming(n):
   return 0.54 - 0.46 * np.cos(2*np.pi*n/(n-1))