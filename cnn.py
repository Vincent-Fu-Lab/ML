#%%
from projet_etu import Module
import numpy as np
from lin import Linear
from notlin import ReLU
from pipeline import *
#%%
class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride):
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride

        np.random.seed(0)
        self._parameters = 2 * np.random.rand(k_size, chan_in, chan_out) * 1e-2 # dim (k_size, chan_in, chan_out)
        self._gradient = np.zeros((k_size,chan_in,chan_out)) # dim : k_size, chan_in, chan_out

    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,(length-k_size)/stride + 1, chan_out)
        '''
        b,l,cin = X.shape

        res = np.zeros((b, (l-self.k_size)//self.stride + 1, self.chan_out))
        for n in range(b):
            X0 = X[n] # dim (length,chan_in)
            for f in range(self.chan_out): # for every filter
                W = self._parameters[:,:,f] # dim (k_size,chan_in)

                for i in range(0,res.shape[1]):
                    j = i*self.stride
                    X1 = X0[j:j+self.k_size] # dim (k_size,chan_in)
                    
                    res[n,i,f] = (X1 * W).sum()

        return res

    def backward_update_gradient(self, input, delta):
        '''
        params:
        -------
        input : dim (batch,length,chan_in)
        delta : dim (batch,(length-k_size)/stride + 1,chan_out)
        self._gradient : (k_size,chan_in,chan_out)
        
        return:
        -------
        None
        '''
        assert input.shape[2] == self.chan_in
        assert delta.shape[2] == self.chan_out
        assert delta.shape[1] == (input.shape[1]-self.k_size)//self.stride + 1
        assert delta.shape[0] == input.shape[0]
        b, length_out, chan_out = delta.shape
        #g = np.zeros((self.k_size,self.chan_in,self.chan_out))
        for n in range(b):
            X0 = input[n] # dim (length,chan_in)
            for z in range(chan_out):
                for i in range(length_out):
                    Xs = X0[i:i+self.k_size] # dim (k_size, chan_in)
                                             # derivative of o_i with respect of w is x
                    delta0 = delta[n,i,z]
                    self._gradient[:,:,z] += Xs*delta0
                    #g[:,:,z] += Xs*delta0
        #return g

    def backward_delta(self, input, delta):
        '''
        params:
        -------
        input : dim (batch,length,chan_in)
        delta : dim (batch,(length-k_size)/stride + 1,chan_out)
        self._parameters : dim (k_size, chan_in, chan_out)
        
        return:
        -------
        delta : dim(batch, length, chan_in)        
        '''    

        b, length_out, chan_out = delta.shape
        
        res = np.zeros(input.shape)

        for n in range(b):
            X0 = input[n] # dim (length,chan_in)
            for z in range(chan_out):
                Ws = self._parameters[:,:,z] # dim (k_size, chan_in)
                                             # derivative of o_i with respect of x is w
                for i in range(length_out):
                    delta0 = delta[n,i,z]
                    res[n,i:i+self.k_size,:] += Ws*delta0
        return res
    
    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
'''
a = np.array([[[3,4,8],[5,6,7],[3,3,9],[6,7,1]]])
b = np.array([[[1,2,3,6],[4,2,6,10],[10,11,12,5]]])
c1 = Conv1D(2,3,4,1)
#f = c1.forward(a)
print('c1',c1.backward_delta(a,b))
print('c1',c1.backward_update_gradient(a,b))

c2 = Conv1D2(2,3,4,1)
print('c2',c2.backward_delta(a,b))
print('c2',c2.backward_update_gradient(a,b))
'''
#%%
class MaxPool1D(Module):
    def __init__(self,k_size,stride):
        self.k_size = k_size
        self.stride = stride
        self.maxind = None
    
    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,(length-k_size)/stride + 1, chan_in)
        '''
        b,l,cin = X.shape

        res = np.zeros((b, (l-self.k_size)//self.stride + 1, cin))
        self.maxind = np.zeros(res.shape)

        for i in range(0, res.shape[1]):
            
            self.maxind[:,i] = np.argmax(X[:, (i * self.stride): (i * self.stride + self.k_size)], axis=1) + i * self.stride
            res[:,i,:] = np.max(X[:,(i*self.stride) : (i*self.stride + self.k_size)], axis=1)
        self.maxind = self.maxind.astype(int)
        return res

    def update_parameters(self, gradient_step=1e-3):
        '''
        no parameter
        '''        
        pass
        

    def backward_update_gradient(self, input, delta):
        '''
        MaxPool1D has no parameter.
        '''        
        pass

    def backward_delta(self, input, delta):
        '''
        There is no gradient with respect to non maximum values, since changing them slightly does not affect the output. 
        Further, the max is locally linear with slope 1, with respect to the input that actually achieves the max. 
        Thus, the gradient from the next layer is passed back to only that neuron which achieved the max. 
        All other neurons get zero gradient.

        chan_in == chan_out
        
        param:
        -------
        input: dim (batch,length,chan_in)
        delta: dim (batch,(length-k_size)//stride + 1,chan_in)
        
        return:
        -------
        delta, dim (batch,length,chan_in)
        '''
        b,l,c = input.shape        
        res = np.zeros((b,l,c))
        
        for n in range(b):
            X0 = input[n]
            ind = self.maxind[n]
            #print(ind)
            for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                    res[n,ind[i,j],j] = delta[n,i,j]
        
        return res
'''
a = np.array([[[3,4,8],[5,6,7],[3,3,9],[6,7,1]]])
b = np.array([[[1,2,3],[4,2,6],[10,11,12]]])
m1 = MaxPool1D(2,1)
m1.forward(a)
m1.backward_delta(a, b)
'''
#%%


class Flatten(Module):
    def __init__(self):
        pass
    
    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,length*chan_in)
        '''
        return X.reshape((len(X),-1))
        
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        param:
        ------
        input : resPool, dim (batch,length,chan_in)
        delta : delta of lin, dim (batch,length*chan_out)
        
        return:
        dim (batch,length,chan_in)
        '''
        return delta.reshape(input.shape)

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def update_parameters(self, gradient_step=1e-3):
        '''
        no parameter
        '''        
        pass    

#%%
uspsdatatrain = "data/USPS_train.txt"
uspsdatatest = "data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)

#%%
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

cov1d = Conv1D(3,1,32,1)
maxpool1d = MaxPool1D(2,2)
flatten = Flatten()
lin1 = Linear(4064,100)
relu = ReLU()
lin2 = Linear(100,10)

seq = Sequentiel(cov1d,
                maxpool1d,
                flatten,
                lin1,
                relu,
                lin2)

# preprocessing 
idx = np.random.choice(len(alltrainx), 200)
alltrainx = MinMaxScaler().fit_transform(alltrainx)[idx][:,:,np.newaxis]
alltrainy = OneHotEncoder(sparse = False).fit_transform(alltrainy.reshape(-1,1))[idx]

n_iter = 10
ep = 1e-5
sq, loss = mini_SGD(seq,alltrainx, alltrainy, batch_size=50, eps=ep, loss_fonction=Softmax_CELoss(), nb_iteration=n_iter)

#%%
idx = np.random.choice(len(alltestx), 200)
alltestx_transformed = MinMaxScaler().fit_transform(alltestx)[idx][:,:,np.newaxis]

outputs = sq.forward(alltestx_transformed)
yhat = Softmax().forward(outputs[-1])
yhat = np.argmax(yhat, axis=1)

print(zero_one_loss(yhat, alltesty[idx])) 

#plt.plot(range(n_iter),loss)

#plt.show()
# %%
