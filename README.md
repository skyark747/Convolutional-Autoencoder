# Convolutional-Autoencoder
Convolutional Autoencoder  implemented in numpy and optimized to run faster on cpu using numba.jit. The Autoencoder folows modular design i.e adjust params and layers according to need.

## dependencies
Anaconda/Miniconda

## Setting up the Environment
```
conda create -n AutoEncoder python=3.9
conda activate AutoEncoder
pip install -r requirements.txt
```

**Train the model**
```
conda activate AutoEncoder
python train.py
```

**Testing the model**
```
python test.py
```





