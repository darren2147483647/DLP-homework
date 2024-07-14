import numpy as np
#一些必要的函式
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(sigmoid(x),1.0-sigmoid(x))
#
class layer():
    def __init__(self,sample_size,input_size,output_size):#z=xw+b y=6(z)
        self.x=np.random.randn(sample_size,input_size)
        self.w=np.random.randn(input_size,output_size)
        self.b=np.random.randn(1,output_size)
        self.b=np.tile(self.b, (sample_size, 1))
        self.z=np.random.randn(sample_size,output_size)
        self.y=np.random.randn(sample_size,output_size)
        self.n=sample_size
    def forword(self,input):
        self.x=input
        self.xw=np.matmul(self.x,self.w)
        self.b_expend=np.tile(self.b, (self.n, 1))
        self.z=self.xw+self.b
        self.y=sigmoid(self.z)
        return self.y
    def back(self,gradient):
        tmp=derivative_sigmoid(self.z)*gradient
        self.w_gradient=np.matmul(self.x.T,tmp)
        self.b_gradient=tmp.sum(axis=0, keepdims=True)
        return np.matmul(tmp,self.w.T)
    def update(self,lr=0.01):
        self.w-=lr*self.w_gradient
        self.b-=lr*np.tile(self.b_gradient, (self.n, 1))