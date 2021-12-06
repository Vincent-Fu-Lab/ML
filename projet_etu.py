import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step = 1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis = 1) ** 2
    
    def backward(self, y, yhat):
        return - 2 * (y - yhat)
    

class Linear(Module):
    def __init__(self, input, output):
        super(Linear, self).__init__()
        self.input = input
        self.output = output
        self._parameters = np.random.randn(self.input, self.output)
        
    def zero_grad(self):
        self._gradient = np.zeros((self.input, self.output))
        
    def forward(self, X):
        return X @ self._parameters
    
    def backward_update_gradient(self, input, delta):
        self._gradient = self._gradient + input.T @ delta
        
    def backward_delta(self, input, delta):
        return delta @ self._parameters.T
    

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        
    def forward(self, X):
        return np.tanh(X) 
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)
    
    
class Sigmoide(Module):
    def __init__(self):
        super(Sigmoide, self).__init__()
        
    def forward(self, X):
        return 1 / (1 + np.exp(-X)) 
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        return delta * (np.exp(-input) / ((1 + np.exp(-input)) ** 2))
    
    
class Sequentiel:
    def __init__(self):
        self.net = []
        
    def append_modules(self, *args):
        for module in args:
            self.net.append(module)
        
    def net_forward(self, datax):
        forwards = [self.net[0].forward(datax)]
        
        for i in range(1, len(self.net)):
            forwards.append(self.net[i].forward(forwards[-1]))
            
        return forwards
    
    def net_update(self, datax, datay, loss, eps):
        for module in self.net:
            module.zero_grad()
        
        forwards = self.net_forward(datax)
        deltas = [loss.backward(datay, forwards[-1])]
        #print("i", deltas[0])
        
        for i in range(len(self.net) - 1, 0, -1):
            deltas = [self.net[i].backward_delta(forwards[i - 1], deltas[0])] + deltas
            #print(i, deltas[0])
        
        deltas = [self.net[0].backward_delta(datax, deltas[0])] + deltas
        self.net[0].backward_update_gradient(datax, deltas[1])
        
        for i in range(1, len(self.net)):
            self.net[i].backward_update_gradient(forwards[i - 1], deltas[i + 1])
            
        for module in self.net:
            module.update_parameters(eps)
            
        
        
            
            
class Optim:
    def __init__(self, net, loss = MSELoss, eps = 1e-3):
        self.net = net
        self.loss = loss()
        self.eps = eps
        
    def step(self, batch_x, batch_y):
        return self.net.net_update(batch_x, batch_y, self.loss, self.eps)
        

def SGD(net, datax, datay, batch_size, ite):
    o = Optim(net)
    
    for i in range(ite):
        inds = np.random.choice([i for i in range(len(datax))], size = batch_size)
        batch_x = datax[inds]
        batch_y = datay[inds]
        o.step(batch_x, batch_y)
        
    return net
    

class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        
    def forward(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis = 1).reshape(X.shape[0], 1)
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        return delta * self.forward(input) * (1 - self.forward(input))


class CELoss(Loss):
    def forward(self, y, yhat):
        return np.array([- yhat[i, y[i]] for i in range(len(y))])
    
    def backward(self, y, yhat):
        res = np.zeros(yhat.shape)
        
        for i in range(len(yhat)):
            res[i, int(y[i])] = -1
            
        return res
    
    
class CrossEntropy(Loss):
    def forward(self, y, yhat):
        return CELoss().forward(y, yhat) + np.log(np.sum(np.exp(yhat), axis = 1).reshape(yhat.shape[0], 1))
    
    def backward(self, y, yhat):
        return CELoss().backward(y, yhat) + np.exp(yhat) / np.sum(np.exp(yhat), axis = 1).reshape(yhat.shape[0], 1)
    
    
class BCE(Loss):
    def log_clip(self, arg):
        return np.array([[max(-100, np.log(arg[i, j] + 1e-8)) for j in range(arg.shape[1])] for i in range(arg.shape[0])])
    
    def forward(self, y, yhat):
        return - (y * self.log_clip(yhat) + (1 - y) * self.log_clip(1 - yhat))
    
    def backward(self, y, yhat):
        return - (y / (yhat + 1e-8) - (1 - y) / ((1 - yhat) + 1e-8))
    
    
class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        super(Conv1D, self).__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = 1e-1 * np.random.randn(self.k_size, self.chan_in, self.chan_out)
        
    def zero_grad(self):
        self._gradient = np.zeros((self.k_size, self.chan_in, self.chan_out))
        
    def forward(self, X):
        res = np.empty((X.shape[0], (X.shape[1] - self.k_size) // self.stride + 1, self.chan_out))
        p, q, r = res.shape
        
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    res[i, j, k] = np.sum(X[i, j * self.stride : j * self.stride + self.k_size] * self._parameters[:, :, k])
        
        return res
    
    def backward_update_gradient(self, input, delta):
        p, q, r = delta.shape 
        
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    self._gradient[:, :, k] = input[i, j * self.stride : j * self.stride + self.k_size] * delta[i, j, k]
                    
                    
    def backward_delta(self, input, delta):
        res = np.zeros(input.shape)
        p, q, r = delta.shape
        
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    res[i, j * self.stride : j * self.stride + self.k_size] = self._parameters[:, :, k] * delta[i, j, k]
                    
        return res
    

class MaxPool1D(Module):                    
    def __init__(self, k_size, stride):
        super(MaxPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride
        self.max_inds = None
        
    def forward(self, X):
        res = np.empty((X.shape[0], (X.shape[1] - self.k_size) // self.stride + 1, X.shape[2]))
        self.max_inds = np.empty(res.shape)
        p, q, r = res.shape
        
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    res[i, j, k] = np.max(X[i, j * self.stride : j * self.stride + self.k_size, k])
                    self.max_inds[i, j, k] = np.argmax(X[i, j * self.stride : j * self.stride + self.k_size, k]) + j * self.stride
        
        return res
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        res = np.empty(input.shape)
        p, q, r = delta.shape
        
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    res[i, int(self.max_inds[i, j, k]) ,k] = delta[i, j, k]
                    
        return res


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, X):
        p, q, r = X.shape
        return X.reshape(p, q * r)
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)  
    

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, X):
        return (X > 0) * X
    
    def update_parameters(self, gradient_step = None):
        pass
    
    def backward_update_gradient(self, input = None, delta = None):
        pass
    
    def backward_delta(self, input, delta):
        return (input > 0) * np.ones(input.shape) * delta
    
    
 
# usefull fonctions for test process    
    
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
   
    
if __name__ == '__main__':
    """
    # linear 
    np.random.seed(1)

    datax = np.random.randn(20,10)
    datay = np.random.choice([-1,1],20,replace=True)
    dataymulti = np.random.choice(range(10),20,replace=True)
    linear = Linear(10,1)
    mse = MSELoss()
    linear.zero_grad()
    x = []
    y = []
    
    for i in range(500):
        res_lin = linear.forward(datax)
        res_mse = mse.forward(datay.reshape(-1,1), res_lin)
        y.append(np.sum(res_mse))
        x.append(i)
        delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
        linear.backward_update_gradient(datax,delta_mse)
        grad_lin = linear._gradient
        delta_lin = linear.backward_delta(datax,delta_mse)
        linear.update_parameters(gradient_step=1.5e-7)
    
    plt.plot(x, y)
    plt.xlabel("timestep")
    plt.ylabel("sum error")
    plt.title("Linear Regression")
    plt.grid()
    plt.show()
    
    
    # binary classification
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 6
    pos = 9
    datax,datay = get_usps([neg, pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg, pos],alltestx,alltesty)
    datay[datay == neg] = 0
    datay[datay == pos] = 1
    testy[testy == neg] = 0
    testy[testy == pos] = 1
    
    linear1 = Linear(256, 64)
    linear2 = Linear(64, 1)
    mse = MSELoss()
    tanh = Tanh()
    sigmoide = Sigmoide()
    linear1.zero_grad()
    linear2.zero_grad()

    res = []
    pred = []
    x = []
    
    for i in range(100):
        res_lin = linear1.forward(datax)
        res_tanh = tanh.forward(res_lin)
        res_lin2 = linear2.forward(res_tanh)
        res_sin = sigmoide.forward(res_lin2)
        res.append(np.sum(mse.forward(datay.reshape(-1,1), res_sin)))
        pred.append(np.sum(res_sin == datay.reshape(-1,1)) / len(datay))
        x.append(i)
        delta_mse = mse.backward(datay.reshape(-1,1), res_sin)
        delta_sin = sigmoide.backward_delta(res_lin2, delta_mse)
        delta_lin2 = linear2.backward_delta(res_tanh, delta_sin)
        delta_tanh = tanh.backward_delta(res_lin, delta_lin2)
        delta_lin = linear1.backward_delta(datax, delta_tanh)
        linear1.backward_update_gradient(datax, delta_tanh)
        tanh.backward_update_gradient(res_lin, delta_lin2)
        linear2.backward_update_gradient(res_tanh, delta_sin)
        sigmoide.backward_update_gradient(res_lin2, delta_mse)
        linear1.update_parameters(gradient_step=1e-4)
        linear2.update_parameters(gradient_step=1e-4)
        
    plt.plot(x, res)
    plt.title("Binary Classification")
    plt.xlabel("timestep")
    plt.ylabel("sum error")
    plt.grid()
    plt.show()
    plt.plot(x, pred)
    plt.title("Predition score")
    plt.xlabel("timestep")
    plt.ylabel("score")
    plt.grid()
    plt.show()
    
    # with sequentiel
    res = []
    pred = []
    x = []

    s = Sequentiel()
    s.append_modules(Linear(256, 64), Tanh(), Linear(64, 1), Sigmoide())
    o = Optim(s, MSELoss, eps=1e-4)
    for i in range(500):
        x.append(i)
        o.step(datax, datay.reshape(datay.shape[0], 1))
        res.append(np.sum(np.abs(s.net_forward(datax)[-1] - datay.reshape(-1, 1)) < 1e-5) / len(datay))

    plt.plot(x, res, label = "|pred - label| < $10^{-5}$")
    plt.title("Predition score")
    plt.xlabel("timestep")
    plt.ylabel("score")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # multi-class classification
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    datax,datay = get_usps([i for i in range(10)],alltrainx,alltrainy)
    testx,testy = get_usps([i for i in range(10)],alltestx,alltesty)
    
    res = []
    pred = []
    x = []
    
    s = Sequentiel()
    s.append_modules(Linear(256, 128), Tanh(), Linear(128, 64), Tanh(), Linear(64, 10), Softmax())
    o = Optim(s, CrossEntropy, 1e-3)
    o.step(datax, datay)
    for i in range(100):
        x.append(i)
        o.step(datax, datay)
        res.append(np.sum(CrossEntropy().forward(datay.reshape(-1, 1), s.net_forward(datax)[-1])))
        pred.append(len(np.where((np.argmax(s.net_forward(datax)[-1], axis = 1) == datay) == True)[0]) / len(datax))
        
    plt.plot(x, res)
    plt.title("Multi-class Classification")
    plt.xlabel("timestep")
    plt.ylabel("sum error")
    plt.grid()
    plt.show()
    plt.plot(x, pred)
    plt.title("Predition score")
    plt.xlabel("timestep")
    plt.ylabel("score")
    plt.grid()
    plt.show()
    
    """
    
    """
    # visualization 
    s1 = Sequentiel()
    s1.append_modules(Linear(256, 100), Tanh(), Linear(100, 3), Tanh(), Linear(3, 100), Tanh(), Linear(100, 256), Sigmoide())
    o1 = Optim(s1, MSELoss, 1e-4)
    for i in range(1000):
        o1.step(datax, datax)
    
    for k in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]:
        show_usps(datax[k])
        plt.show()
        show_usps(s1.net_forward(datax)[-1][k])
        plt.show()

    """
    
    
    # convolution
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    idx = np.random.choice(len(alltrainx), 50)
    alltrainx = MinMaxScaler().fit_transform(alltrainx)[idx][:,:,np.newaxis]
    alltrainy = np.argmax(OneHotEncoder(sparse = False).fit_transform(alltrainy.reshape(-1,1))[idx], axis = 1)
    
    x = []
    res = []
    pred = []
    
    s2 = Sequentiel()
    s2.append_modules(Conv1D(3, 1, 32, 1), MaxPool1D(2, 2), Flatten(), Linear(4064, 100), ReLU(), Linear(100, 10), Softmax())
    o2 = Optim(s2, CrossEntropy, 1e-2)
    for i in range(10):
        print(i)
        x.append(i)
        o2.step(alltrainx, alltrainy)
        res.append(np.sum(CrossEntropy().forward(alltrainy, s2.net_forward(alltrainx)[-1])))
        pred.append(np.sum(np.argmax(s2.net_forward(alltrainx)[-1], axis = 1) == alltrainy) / len(alltrainx))

    plt.plot(x, res)
    plt.title("Picture classification with convolution")
    plt.xlabel("timestep")
    plt.ylabel("sum error")
    plt.grid()
    plt.show()
    plt.plot(x, pred)
    plt.title("Predition score")
    plt.xlabel("timestep")
    plt.ylabel("score")
    plt.grid()
    plt.show()
    
    
    
    
    
        
        
        
























