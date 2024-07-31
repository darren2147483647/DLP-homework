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
def show_loss(loss_history, span):
    import matplotlib.pyplot as plt
    """
    Plots the loss history over training epochs.
    
    Parameters:
    loss_history (list): A list of loss values recorded during training.
    span (int): The span over which to plot the loss history.
                Defaults to 100, but can be adjusted to reflect the actual length of training epochs.
    """
    epochs = list(range(0, len(loss_history)*span,span))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()

class trainer():
    def __init__(self,x,y,lr=0.01):
        self.x=x
        self.y=y
        self.lr=lr
        self.a=model(x)
        self.span=10
        self.loss_history=[]
    def train(self,epoch=100):
        for i in range(epoch):
            self.y_pred=self.a.forward()
            self.mse=np.sum((self.y_pred-self.y)**2)/self.x.shape[0]
            self.hit=np.sum(self.y==(self.a.y_pred>0.5))
            self.accuracy=self.hit/self.x.shape[0]
            self.grad=(self.y_pred-self.y)*2.0/self.x.shape[0]
            if self.hit==self.x.shape[0]:
                for i in range(self.x.shape[0]):
                    print(f"Iter{i} |\tGround truth: {self.y[i,0]} |\tprediction: {self.y_pred[i,0]}|")
                print(f"loss={self.mse} accuracy={self.accuracy*100.0}%")
                break
            self.a.back(self.grad)
            self.a.upd(self.lr)
            if i%self.span==0:
                print(f"epoch {i} loss : {self.mse}")
                self.loss_history.append(self.mse)
            
    def print(self):
        print(f"accuracy: {self.hit}/{self.x.shape[0]}")
        show_result(self.x,self.y,self.a.y_pred>0.5)
        show_loss(self.loss_history,self.span)
    def test(self,x,y,show=False):
        self.y_pred=self.a.test(x)
        self.hit=np.sum(y==(self.y_pred>0.5))
        self.accuracy=self.hit/x.shape[0]
        for i in range(x.shape[0]):
            print(f"Iter{i} |\tGround truth: {y[i,0]} |\tprediction: {self.y_pred[i,0]}|")
        print(f"loss={self.mse} accuracy={self.accuracy*100.0}%")
        if show:
            show_result(x,y,self.y_pred>0.5)
