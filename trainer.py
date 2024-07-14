import numpy as np
from model import model
#一些必要的函式
def show_result(x,y,pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title("Predict result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()
#
class trainer():
    def __init__(self,x,y,lr=0.01):
        self.x=x
        self.y=y
        self.lr=lr
        self.a=model(x)
    def train(self,epoch=100):
        for i in range(epoch):
            self.y_pred=self.a.forward()
            self.mse=np.sum((self.y_pred-self.y)**2)/self.x.shape[0]
            self.grad=(self.y_pred-self.y)*2.0/self.x.shape[0]
            self.a.back(self.grad)
            self.a.upd(self.lr)
            if i%1==0:
                print(f"epoch={i}: loss={self.mse}")
    def print(self,all=False):
        if all:
            print(self.a.l3.z[0,0],self.a.l3.y[0,0])
        print(f"accuracy: {np.sum(self.y==(self.a.y_pred>0.5))}/{self.x.shape[0]}")
        show_result(self.x,self.y,self.a.y_pred>0.5)