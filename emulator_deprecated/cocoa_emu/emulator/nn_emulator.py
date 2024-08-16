import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
import sys
from torchinfo import summary
from datetime import datetime
'''
Affine, \
                                  ResBlock, \
                                  ResBottle, \
                                  DenseBlock, \
                                  Attention, \
                                  Transformer, \
                                  True_Transformer, \
                                  Better_Attention, \
                                  Better_Transformer, \
                                  Better_ResBlock, \
                                  mamba_block, \
                                  SSM
'''
class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        #self.norm2 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        #self.act2 = #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        # first layer
        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        # middle layer
        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        # last layer
        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.Tanh()

        self.skip     = nn.Identity()#nn.Linear(size,size)
        self.act_skip = nn.Tanh()

    def forward(self, x):
        x_skip = self.act_skip(self.skip(x))

        o1 = self.act1(self.norm1(self.layer1(x)/np.sqrt(10)))
        o2 = self.act2(self.norm2(self.layer2(o1)/np.sqrt(10)))
        o3 = self.norm3(self.layer3(o2))
        o  = self.act3(o3+x_skip)

        return o

class DenseBlock(nn.Module):
    def __init__(self, size):
        super(DenseBlock, self).__init__()

        self.skip = nn.Identity()

        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, size)

        self.norm1 = torch.nn.BatchNorm1d(size)
        self.norm2 = torch.nn.BatchNorm1d(size)

        self.act1 = nn.Tanh()#nn.SiLU()#nn.PReLU()
        self.act2 = nn.Tanh()#nn.SiLU()#nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)
        o1    = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2    = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10)
        o     = torch.cat((o2,xskip),axis=1)
        return o

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = nn.Tanh() #activation_fcn(in_size)  #nn.Tanh()#nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = nn.Tanh() #activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)
        #x_norm = self.norm(x)
        #_x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        #_x = x.reshape(x.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        o1 = self.act(self.norm(torch.matmul(x,mat1)+self.bias1))
        o2 = torch.matmul(o1,mat2)+self.bias2  #self.act2(self.norm2(torch.matmul(o1,mat2)+self.bias2))
        o3 = self.act3(self.norm3(o2+x))
        return o3

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = -1*torch.mul(self.beta,x)
        inv = (1+torch.exp(exp)).pow_(-1)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

class True_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(True_Transformer, self).__init__()  
    
        self.int_dim      = in_size//n_partitions
        self.n_partitions = n_partitions
        self.linear       = nn.Linear(self.int_dim,self.int_dim)#ResBlock(self.int_dim,self.int_dim)#
        self.act          = nn.ReLU()
        self.norm         = torch.nn.BatchNorm1d(self.int_dim*n_partitions)

    def forward(self,x):
        batchsize = x.shape[0]
        out = torch.reshape(self.norm(x),(batchsize,self.n_partitions,self.int_dim))
        out = self.act(self.linear(out))
        out = torch.reshape(out,(batchsize,self.n_partitions*self.int_dim))
        return out+x
    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, invcov, mask=None, model=None, optim=None, device='cpu'):
        self.N_DIM = N_DIM
        self.model = model
        self.optim = optim
        self.device = device
        self.trained = False
        self.dv_fid = torch.Tensor(dv_fid)
        self.dv_std = torch.Tensor(dv_std)
        self.invcov = torch.Tensor(invcov)
        if mask is not None:
            self.mask = mask.astype(float)
        else:
            self.mask = np.ones(OUTPUT_DIM)        
        self.mask = torch.Tensor(self.mask)
        
        if (model==0):
            print("Using simply connected NN...")
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, OUTPUT_DIM),
                                Affine()
                                )
        elif(model==1):
            print("Using resnet model...")
            self.model = nn.Sequential(
                           nn.Linear(N_DIM, 128),
                           ResBlock(128, 256),
                           ResBlock(256, 256),
                           ResBlock(256, 256),
                           ResBlock(256, 512),
                           ResBlock(512, 512),
                           ResBlock(512, 512),
                           ResBlock(512, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           Affine(),
                           nn.PReLU(),
                           nn.Linear(1024, OUTPUT_DIM),
                           Affine()
                       )        
        elif(model==2):
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, OUTPUT_DIM),
                                Affine()
                                )
        elif(model==3):
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, OUTPUT_DIM),
                                Affine()
                                )
        elif(model==4):
            print("Using resnet model...")
            self.model = nn.Sequential(
                           nn.Linear(N_DIM, 1024),
                           nn.ReLU(),
                           ResBlock(1024, 512),
                           ResBlock(512, 256),
                           ResBlock(256, 128),
                           nn.Linear(128, 512),
                           nn.ReLU(),
                           nn.Linear(512, OUTPUT_DIM),
                           nn.ReLU(),
                           Affine()
                       )
        elif(model==5):
            print("Using Evan's model...")
            in_dim_res = 256
            n_channels = 32
            int_dim_trf = 1024
            self.model = nn.Sequential(
                            nn.Linear(N_DIM, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            nn.Linear(int_dim_res, int_dim_trf),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            nn.Linear(int_dim_trf,OUTPUT_DIM),
                            Affine()
                        )

        self.model.to(device)

        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)

    def do_pca(self, data_vector, N_PCA):
        self.N_PCA = N_PCA
        pca = PCA(self.N_PCA)
        pca.fit(data_vector)
        self.pca = pca
        pca_coeff = pca.transform(data_vector)
        return pca_coeff
    
    def do_inverse_pca(self, pca_coeff):
        return self.pca.inverse_transform(pca_coeff)
    
    def train(self, X, y, test_split=None, batch_size=32, n_epochs=100):
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True))
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True))
            self.y_mean = self.dv_fid
            self.y_std  = self.dv_std

        X_train = (X - self.X_mean) / self.X_std
#         y_train = y / self.dv_fid
        y_train = (y - self.y_mean) / self.y_std

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
        epoch_range = tqdm(range(n_epochs))
        
        losses = []
        loss = 100.
        for _ in epoch_range:
            for i, data in enumerate(trainloader):
                X_batch = data[0]
                y_batch = data[1]

                y_pred = self.model(X_batch)
                _d = (y_batch-y_pred)*self.mask*self.y_std
                _chi2 = (_d*torch.matmul(_d, self.invcov)).sum(-1)
                loss = torch.mean(_chi2)
                #loss = torch.mean(torch.abs(y_batch - y_pred) * self.mask)
                losses.append(loss)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            epoch_range.set_description('Loss: {0}'.format(loss))

        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach()
            X_std  = self.X_std.clone().detach()

            X_norm = (X - X_mean) / X_std
            y_pred = self.model.eval()(X_norm).cpu()
        
#         y_pred = y_pred * self.dv_fid
        y_pred = y_pred * self.y_std + self.y_mean

        return y_pred.numpy()

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['Y_mean'] = self.y_mean
            f['Y_std']  = self.y_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
        
    def load(self, filename):
        self.trained = True
        self.model = torch.load(filename)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.y_mean = torch.Tensor(f['Y_mean'][:])
            self.y_std  = torch.Tensor(f['Y_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
