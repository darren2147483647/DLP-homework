import numpy as np
from layer import layer
#
class model():
    def __init__(self,x):
        self.x=x
        self.n=self.x.shape[0]
        self.l1=layer(self.n,2,100)
        self.l2=layer(self.n,100,100)
        self.l3=layer(self.n,100,1)
    def forward(self):
        tmp = self.l1.forword(self.x)
        tmp = self.l2.forword(tmp)
        self.y_pred = self.l3.forword(tmp)
        return self.y_pred
    def back(self,grad):
        tmp = self.l3.back(grad)
        tmp = self.l2.back(tmp)
        tmp = self.l1.back(tmp)
    def upd(self,lr=0.01):
        self.l1.update(lr)
        self.l2.update(lr)
        self.l3.update(lr)
