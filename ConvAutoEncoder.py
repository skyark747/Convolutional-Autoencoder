from Conv2D import Conv2D
from DeConv2D import ConvTranspose2D
from MaxPool2D import MaxPool2D
from UnPool2D import MaxUnPool2D
import numpy as np


class Convautoencoder:
    def __init__(self,load,params_c,params_d):

        self.encoder = [
            Conv2D(input_channels=3,out_channels=16,kernel_size=3,stride=1,load=load,params=params_c[0]),  
            Conv2D(input_channels=16,out_channels=32,kernel_size=3,stride=1,load=load,params=params_c[1]), 
            MaxPool2D(2,2),
            Conv2D(input_channels=32,out_channels=64,kernel_size=3,stride=1,load=load,params=params_c[2]),             
        ]

        self.decoder = [

            ConvTranspose2D(input_channels=64,out_channels=32,kernel_size=3,stride=1,load=load,params=params_d[0]),
            MaxUnPool2D(2,2),
            ConvTranspose2D(input_channels=32,out_channels=16,kernel_size=3,stride=1,load=load,params=params_d[1]),
            ConvTranspose2D(input_channels=16,out_channels=3,kernel_size=3,stride=1,load=load,params=params_d[2]),
        ]

    def sigmoid(self,X):
        return 1 / (1 + np.exp(-X))
    
    
    def get_params(self):
        params_conv,params_deconv=[],[]
        for i,layer in enumerate(self.encoder):
            if not isinstance(layer,MaxPool2D):
                kernels,b=layer.get_params()
                params_conv.append({'kernels':kernels,'b':b})
                
        for i,layer in enumerate(self.decoder):
            if not isinstance(layer,MaxUnPool2D):
                kernels,b=layer.get_params()
                params_deconv.append({'kernels':kernels,'b':b})
                
                     
        return params_conv,params_deconv
    
    def forward(self, X):
    
        #for skip-conections
        self.encoder_activations = []
        self.indices=[] 
        
        # Encoder
        for layer in self.encoder:
            X=layer.forward(X,False)
            if isinstance(layer, Conv2D):
                self.encoder_activations.append(X.copy())
            
        # Decoder
        skip_index = len(self.encoder_activations) - 1
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, ConvTranspose2D):
                if skip_index >= 0:
                    skip = self.encoder_activations[skip_index]
                    
                    if X.shape != skip.shape:
                        X=skip[:,:skip.shape[1],:skip.shape[2]]
                    X += skip

                    skip_index -= 1
                
                if i!=len(self.decoder)-1:
                    X = layer.forward(X,False)
                else:
                    X = layer.forward(X,True)
            
        X = self.sigmoid(X)
        return X
    
    def backward(self, dz, learning_rate):

        # Decoder backward pass
        for i, layer in enumerate(reversed(self.decoder)):
            
                if i==0:
                    dz = layer.backward(dz, learning_rate,True)
                    
                else:
                    dz = layer.backward(dz, learning_rate,False)
                    
                    

        # Encoder backward pass
        for i, layer in enumerate(reversed(self.encoder)):

            dz = layer.backward(dz, learning_rate,False)
            #print(dz.max(),dz.min())

    def cal_err(self,z,gt):
        return np.mean((z-gt)**2)
    
    def sig_derivative(self,x):
        return x * (1 - x)

    def train(self,x,gt,learning_rate,n_samples):
        z=self.forward(x)

        if z.shape != gt.shape:
            gt=gt[:,:z.shape[1],:z.shape[2]] #clipping to match output dimensions if mismatch occurs a little

        train_loss= self.cal_err(z,gt) #mean-squared error

        dz=(2*(z-gt))/n_samples #derivative of MSE
        dz*=self.sig_derivative(z) 
        self.backward(dz,learning_rate)
        
        return train_loss
    
    def test(self, X):
        return self.forward(X)
    
    def val_test(self,X):

        z = self.forward(X)

        if z.shape != X.shape:
            X=X[:,:z.shape[1],:z.shape[2]] 

        val_loss= self.cal_err(z,X) 

        return val_loss
        