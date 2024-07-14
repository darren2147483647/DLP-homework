import numpy as np
from trainer import trainer
#一些必要的函式
def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

#設定資料
task=1
x,y=np.array([[0.5,0.8],[0.6,0.2],[0.5,0.3],[0.1,0.7]]),np.array([0,1,1,0]).reshape(4,1)
if task==1:
    x,y=generate_linear(n=100)
elif task==2:
    x,y=generate_XOR_easy()
#
if __name__=="__main__":
    nn=trainer(x,y,0.1)
    nn.train(100000)
    nn.print(1)