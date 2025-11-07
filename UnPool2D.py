import numpy as np
from numba import njit,jit

@njit
def forward(X,stride):
    C, H, W = X.shape
    H_new, W_new = H * stride, W * stride
    upsampled = np.zeros((C, H_new, W_new))

    for c in range(C):
        for i in range(H_new):
            for j in range(W_new):
                # Map coordinates back to original image space
                x = i / stride
                y = j / stride

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, H - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, W - 1)

                dx = x - x0
                dy = y - y0

                # Bilinear interpolation
                top = (1 - dy) * X[c, x0, y0] + dy * X[c, x0, y1]
                bottom = (1 - dy) * X[c, x1, y0] + dy * X[c, x1, y1]
                value = (1 - dx) * top + dx * bottom

                upsampled[c, i, j] = value

    return upsampled
    
@jit(nopython=True)
def backward(X,kernel_size,stride):
    C, H, W = X.shape

    h_out = int((H - kernel_size) / stride + 1)
    w_out = int((W - kernel_size) / stride + 1)

    output = np.zeros((C, h_out, w_out))

    for c in range(C):
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                window = X[c, h_start:h_start + kernel_size, w_start:w_start + kernel_size]
                max_n=np.max(window)
                output[c, i, j] = max_n

    return output


class MaxUnPool2D:
    def __init__(self,kernel_size,stride):
        self.kernel_size=kernel_size
        self.stride=stride

    def forward(self,X,indices):
        return forward(X,self.stride)
    
    def backward(self,X,learning_rate,dummy):
        return backward(X,self.kernel_size,self.stride)