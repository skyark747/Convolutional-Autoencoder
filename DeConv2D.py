import numpy as np
from numba import njit,jit

@jit(nopython=True)
def deconvolution(X,stride,kernel_size,out_channels,b,kernels):
        rows = int((X.shape[1] - 1) * stride + kernel_size)
        cols = int((X.shape[2] - 1) * stride + kernel_size)

        output = np.zeros((out_channels, rows, cols))

        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                x_slice = X[:, i, j]  
                for k in range(out_channels):
                    output[k, i*stride:i*stride+kernel_size,
                            j*stride:j*stride+kernel_size] += \
                            np.sum(x_slice[:, None, None] * kernels[k], axis=0)
                    output[k]+=b[k]
        return output

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

@jit(nopython=True)
def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)


@jit(nopython=True)
def backward_d(dZ, learning_rate,is_last_layer,kernels,b,X,pre_activation,out_channels,stride,kernel_size):
        d_kernels = np.zeros_like(kernels)
        d_b = np.zeros_like(b)
        d_X = np.zeros_like(X)

        rotated_kernels = manual_flip_kernels(kernels)
        
        if not is_last_layer: 
            dZ=dZ*leaky_relu_derivative(pre_activation)

        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                x_val = X[:, i, j]  # shape: (input_channels,)
                for k in range(out_channels):
                    region = dZ[k, i*stride:i*stride+kernel_size,
                                j*stride:j*stride+kernel_size]
                    
                    if region.shape != (kernel_size, kernel_size):
                        continue

                    d_kernels[k] += x_val[:, None, None] * region
                    d_X[:, i, j] += np.sum(np.sum(rotated_kernels[k] * region, axis=2), axis=1)

        d_b = np.sum(np.sum(dZ, axis=1), axis=1)

        # Update
        kernels-=learning_rate*d_kernels
        b-=learning_rate*d_b
        
        return d_X,kernels,b           

class ConvTranspose2D:
    def __init__(self,input_channels,out_channels,kernel_size,stride,load,params):
        
        self.input_channels=input_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride

        #initialising weights
        if not load:
            scale = np.sqrt(2 / (input_channels * kernel_size * kernel_size))
            self.kernels=np.random.randn(out_channels,input_channels,kernel_size,kernel_size)*scale
            self.b=np.zeros((out_channels,1))
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
        self.z=deconvolution(X,self.stride,self.kernel_size,self.out_channels,self.b,self.kernels)
        self.pre_activation=self.z
        if not is_last_layer:
            self.z=self.relu(self.z)
        
        return self.z
    
    
    def backward(self,dZ,learning_rate,is_last_layer):
        dX,kernels,b=backward_d(dZ,learning_rate,is_last_layer,self.kernels,self.b,self.X,self.pre_activation
                              ,self.out_channels,self.stride,self.kernel_size)
        
        self.kernels=kernels.copy()
        self.b=b.copy()
        
        return dX