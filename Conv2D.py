import numpy as np
from numba import njit,jit

@jit(nopython=True)
def convolution(X,stride,kernel_size,out_channels,kernels,b):
        
        #X.shape -> (depth,h,w)
        rows=int((X.shape[1]-kernel_size)/stride + 1)
        cols=int((X.shape[2]-kernel_size)/stride + 1)

        output=np.zeros((out_channels,rows,cols)) #output after convolution
        
        for k in range(out_channels):
            for i in range(rows):
                for j in range(cols):
                    region = X[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                    output[k, i, j] = np.sum(region * kernels[k]) + b[k]


        return output

@jit(nopython=True)
def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

@jit(nopython=True)
def relu_derivative(x):
    return np.where(x > 0)

@njit
def manual_flip_kernels(kernels):
    out_ch, in_ch, kH, kW = kernels.shape
    flipped = np.empty_like(kernels)
    for o in range(out_ch):
        for i in range(in_ch):
            for h in range(kH):
                for w in range(kW):
                    flipped[o, i, h, w] = kernels[o, i, kH - 1 - h, kW - 1 - w]
    return flipped

@njit
def pad_dZ(dZ, pad_h, pad_w):
    C, H, W = dZ.shape
    padded = np.zeros((C, H + 2 * pad_h, W + 2 * pad_w), dtype=dZ.dtype)
    
    for c in range(C):
        for i in range(H):
            for j in range(W):
                padded[c, i + pad_h, j + pad_w] = dZ[c, i, j]
                
    return padded


@jit(nopython=True)
def backward_c(dZ,learning_rate,is_last_layer,kernels,b,X,pre_activation,kernel_size,out_channels,input_channels,stride):
        
        d_kernels = np.zeros_like(kernels) #to store derivative of each kernel
        d_b = np.zeros_like(b) #to store derivative of each bias
        d_X=np.zeros_like(X) #to store derivative of input to this layer
        
        if not is_last_layer:
            if dZ.shape!=pre_activation.shape:
                pre_activation=pre_activation[:,:dZ.shape[1],:dZ.shape[2]]
            dZ=dZ*leaky_relu_derivative(pre_activation)
        
        #padding dz for gradient of dX
        padded_dz = pad_dZ(dZ,kernel_size-1,kernel_size-1)

        
        rotated_kernels= manual_flip_kernels(kernels) #for derivative of dl/dx
        
        for k in range(out_channels):
            for c in range(input_channels):
                for i in range(dZ.shape[1]):
                    for j in range(dZ.shape[2]):
                        region = X[c, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                        d_kernels[k, c] += dZ[k, i, j] * region
            d_b[k] = np.sum(dZ[k])



        for c in range(input_channels):
            for k in range(out_channels):
                for i in range(d_X.shape[1]):
                    for j in range(d_X.shape[2]):
                        region = padded_dz[k, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                        if region.shape != (kernel_size, kernel_size):
                            continue
                        d_X[c, i, j] += np.sum(region * rotated_kernels[k, c])

    

        kernels-=learning_rate*d_kernels
        b-=learning_rate*d_b
      
        return d_X,kernels,b

class Conv2D:
    def __init__(self,input_channels,out_channels,kernel_size,stride,load,params):
        
        self.input_channels=input_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride

        #initialising weights
        if not load:
            scale = np.sqrt(2 / (input_channels * kernel_size * kernel_size))
            self.kernels=np.random.randn(out_channels,input_channels,kernel_size,kernel_size)*scale
            self.b=np.zeros(out_channels)
        else:
            #load saved parameters
            self.kernels=params['kernels'].copy()
            self.b=params['b'].copy()
        
    def get_params(self):
        return self.kernels,self.b
    def relu(self,x):
        return np.maximum(0,x)
    def leaky_relu(self,z):
        return np.where(z > 0, z, 0.01 * z)
    def forward(self,X,is_last_layer):
        
        self.X=X 
        self.z=convolution(X,self.stride,self.kernel_size,self.out_channels,self.kernels,self.b)
        self.pre_activation=self.z
        
        if not is_last_layer:
            self.z= self.relu(self.z)
        
        return self.z
    
    def backward(self,dZ,learning_rate,is_last_layer):
        dX,kernels,b=backward_c(dZ,learning_rate,is_last_layer,self.kernels,self.b,self.X,self.pre_activation
                              ,self.kernel_size,self.out_channels,self.input_channels,self.stride)
        self.kernels=kernels.copy()
        self.b=b.copy()
        return dX